"""ctypes binding + drop-in wrappers for the C++ MCMC kernels.

Low-level API works on flat ``(B, L*L)`` arrays:
    Ising  -> int8 in {0, 1}
    Potts  -> int32 in {0 .. q-1}

The ``*_compat`` functions mimic the signatures of ``utils_ising`` /
``utils_potts`` so they can be swapped in without touching call sites.
"""

from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
BUILD_DIR = _HERE / "build"
_LIB_NAMES = ("mdns_samplers.dll", "libmdns_samplers.so", "libmdns_samplers.dylib")

ISING_KINDS = ("ising_mh", "ising_sw")
POTTS_KINDS = ("potts_mh", "potts_glauber", "potts_sw")
KINDS = ISING_KINDS + POTTS_KINDS

_SIG = [
    ctypes.c_void_p,  # S
    ctypes.c_int,     # B
    ctypes.c_int,     # L
    ctypes.c_int,     # q
    ctypes.c_double,  # beta
    ctypes.c_double,  # J
    ctypes.c_double,  # h
    ctypes.c_int,     # sweeps
    ctypes.c_void_p,  # e_series (may be NULL)
    ctypes.c_uint64,  # seed
]

_lib: ctypes.CDLL | None = None


class LibNotFound(RuntimeError):
    pass


def lib_path() -> Path:
    for cand in [BUILD_DIR / n for n in _LIB_NAMES] + [_HERE / n for n in _LIB_NAMES]:
        if cand.exists():
            return cand
    raise LibNotFound(
        f"native sampler library not built; run `pwsh scripts/build_cpp.ps1 -Force` "
        f"(looked in {BUILD_DIR})"
    )


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        L = ctypes.CDLL(str(lib_path()))
        for kind in KINDS:
            f = getattr(L, f"mdns_{kind}")
            f.argtypes = _SIG
            f.restype = None
        L.mdns_set_threads.argtypes = [ctypes.c_int]
        L.mdns_set_threads.restype = None
        L.mdns_threads.argtypes = []
        L.mdns_threads.restype = ctypes.c_int
        L.mdns_version.argtypes = []
        L.mdns_version.restype = ctypes.c_char_p
        _lib = L
    return _lib


def version() -> str:
    return lib().mdns_version().decode()


def threads() -> int:
    return int(lib().mdns_threads())


def set_threads(n: int) -> None:
    lib().mdns_set_threads(int(n))


def available() -> bool:
    try:
        lib()
        return True
    except LibNotFound:
        return False


# --------------------------------------------------------------- low level
def run(kind: str, S: np.ndarray, *, beta: float, J: float = 1.0, h: float = 0.0,
        q: int = 0, sweeps: int = 1, seed: int = 0, record_energy: bool = True,
        threads: int | None = None) -> np.ndarray | None:
    """Advance ``S`` in place by ``sweeps`` sweeps.

    One sweep == L*L single-site attempts (MH / Glauber) or one full cluster
    sweep (Swendsen-Wang).

    Returns per-chain energy series ``(B, sweeps)`` or ``None``.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown sampler {kind!r}; expected one of {KINDS}")
    if S.ndim != 2:
        raise ValueError(f"S must be 2-D (B, L*L), got shape {S.shape}")
    B, D = S.shape
    L = int(round(math.isqrt(D)))
    if L * L != D:
        raise ValueError(f"L*L of {D} is not a perfect square")
    if kind in ISING_KINDS:
        if S.dtype != np.int8:
            raise TypeError(f"{kind} needs int8 spins in {{0,1}}, got {S.dtype}")
    else:
        if S.dtype != np.int32:
            raise TypeError(f"{kind} needs int32 states, got {S.dtype}")
        if q < 2:
            raise ValueError("q must be >= 2 for Potts samplers")
    if not S.flags["C_CONTIGUOUS"]:
        raise ValueError("S must be C-contiguous")

    if threads:
        set_threads(threads)

    e = np.empty((B, int(sweeps)), dtype=np.float64) if record_energy else None
    getattr(lib(), f"mdns_{kind}")(
        ctypes.c_void_p(S.ctypes.data),
        int(B), int(L), int(q),
        float(beta), float(J), float(h),
        int(sweeps),
        ctypes.c_void_p(None if e is None else e.ctypes.data),
        ctypes.c_uint64(int(seed)),
    )
    return e


def init_state(kind: str, B: int, L: int, q: int = 3, seed: int | None = None) -> np.ndarray:
    """Random initial configuration."""
    rng = np.random.default_rng(seed)
    D = L * L
    if kind in ISING_KINDS:
        return rng.integers(0, 2, size=(B, D), dtype=np.int8)
    if q < 2:
        raise ValueError("q must be >= 2")
    return rng.integers(0, q, size=(B, D), dtype=np.int32)


def collect(kind: str, *, L: int, B: int, beta: float, J: float = 1.0, h: float = 0.0,
            q: int = 3, burn_in: int = 1000, n_collect: int = 32,
            collect_every: int = 100, seed: int = 0, threads: int | None = None,
            init: np.ndarray | None = None):
    """Run a chain and return ``(samples [n_collect*B, D], energies [n_collect*B])``."""
    S = init_state(kind, B, L, q, seed) if init is None else np.array(init, copy=True)
    if burn_in > 0:
        run(kind, S, beta=beta, J=J, h=h, q=q, sweeps=burn_in, seed=seed, record_energy=False,
            threads=threads)
    out, energies = [], []
    for k in range(n_collect):
        e = run(kind, S, beta=beta, J=J, h=h, q=q, sweeps=collect_every,
                seed=seed + k + 1, record_energy=True, threads=threads)
        out.append(S.copy())
        energies.append(e[:, -1])
    D = L * L
    return np.concatenate(out, axis=0).reshape(n_collect * B, D), np.concatenate(energies)


# ----------------------------------------------------- drop-in replacements
def ising2d_mh_compat(L: int, J: float = 1, h: float = 0, beta: float = .5, B: int = 256,
                      num_collect: int = 20000, burn_in: int = 10000,
                      collect_every: int = 1000, init=None, seed: int = 0,
                      threads: int | None = None) -> np.ndarray:
    """Same contract as ``utils_ising.ising2d_mh`` (returns spins in {-1,+1})."""
    # upstream counts steps in single-site updates; convert to sweeps
    D = L * L
    samples, _ = collect("ising_mh", L=L, B=B, beta=beta, J=J, h=h,
                         burn_in=max(1, burn_in // D),
                         n_collect=max(1, num_collect),
                         collect_every=max(1, collect_every // D),
                         seed=seed, threads=threads, init=init)
    return (samples.astype(np.int8) * 2 - 1)


def ising2d_swendsen_wang_compat(L: int, J: float = 1, beta: float = .5, B: int = 256,
                                 num_collect: int = 20000, burn_in: int = 10000,
                                 collect_every: int = 1000, init=None, seed: int = 0,
                                 threads: int | None = None) -> np.ndarray:
    """Same contract as ``utils_ising.ising2d_swendsen_wang`` (spins in {-1,+1})."""
    samples, _ = collect("ising_sw", L=L, B=B, beta=beta, J=J,
                         burn_in=burn_in, n_collect=num_collect,
                         collect_every=collect_every, seed=seed, threads=threads, init=init)
    return (samples.astype(np.int8) * 2 - 1)


def potts2d_mh_compat(L: int, J: float = 1, h: float = 0, beta: float = .5, q: int = 3,
                      B: int = 256, num_collect: int = 20000, burn_in: int = 10000,
                      collect_every: int = 1000, init=None, seed: int = 0,
                      threads: int | None = None) -> np.ndarray:
    D = L * L
    samples, _ = collect("potts_mh", L=L, B=B, beta=beta, J=J, q=q,
                         burn_in=max(1, burn_in // D), n_collect=max(1, num_collect),
                         collect_every=max(1, collect_every // D), seed=seed,
                         threads=threads, init=init)
    return samples.astype(np.int64)


def potts2d_glauber_compat(L: int, J: float = 1, h: float = 0, beta: float = .5, q: int = 3,
                           B: int = 256, num_collect: int = 20000, burn_in: int = 10000,
                           collect_every: int = 1000, init=None, seed: int = 0,
                           threads: int | None = None) -> np.ndarray:
    D = L * L
    samples, _ = collect("potts_glauber", L=L, B=B, beta=beta, J=J, q=q,
                         burn_in=max(1, burn_in // D), n_collect=max(1, num_collect),
                         collect_every=max(1, collect_every // D), seed=seed,
                         threads=threads, init=init)
    return samples.astype(np.int64)


def potts2d_swendsen_wang_compat(L: int, J: float = 1, beta: float = .5, q: int = 3,
                                 B: int = 256, num_collect: int = 20000,
                                 burn_in: int = 10000, collect_every: int = 1000,
                                 init=None, seed: int = 0,
                                 threads: int | None = None) -> np.ndarray:
    samples, _ = collect("potts_sw", L=L, B=B, beta=beta, J=J, q=q,
                         burn_in=burn_in, n_collect=num_collect,
                         collect_every=collect_every, seed=seed, threads=threads, init=init)
    return samples.astype(np.int64)
