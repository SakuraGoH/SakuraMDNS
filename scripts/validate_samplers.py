"""Validate the C++ samplers against exact enumerable distributions.

  Ising  L=4  -> 2^16 = 65_536 configurations (exact)
  Potts  L=3  -> 3^9  = 19_683 configurations (exact)

For each kernel we (a) check the reported energy equals the reference
Hamiltonian on the produced configuration and (b) compare the empirical
energy distribution to the exact one via total-variation distance.

Usage:  python scripts/validate_samplers.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mdns.samplers import cpp_ext  # noqa: E402
from mdns.physics import (  # noqa: E402
    get_all_potts_configs, ising2d_get_all_configs, ising2d_ham, potts2d_ham)


def exact_energy_pmf(model: str, L: int, beta: float, J: float, h: float, q: int):
    if model == "ising":
        cfg = ising2d_get_all_configs(L, device="cpu").float()  # {-1,+1}
        H = ising2d_ham(cfg, J=J, h=h)
    else:
        cfg = get_all_potts_configs(L, q).to(torch.int32)
        H = potts2d_ham(cfg, J=J, q=q)
    logp = (-beta * H).double()
    p = torch.softmax(logp, dim=0).numpy()
    energy = H.numpy().round(6)
    vals, inv = np.unique(energy, return_inverse=True)
    pmf = np.bincount(inv, weights=p, minlength=len(vals))
    return vals, pmf


def empirical_energy_pmf(energies: np.ndarray, vals: np.ndarray):
    idx = np.clip(np.searchsorted(vals, energies.round(6)), 0, len(vals) - 1)
    counts = np.bincount(idx, minlength=len(vals)).astype(np.float64)
    return counts / counts.sum()


def check(model: str, kind: str, L: int, beta: float, J: float = 1.0, h: float = 0.0,
          q: int = 3, B: int = 1024, n_collect: int = 40, collect_every: int = 25,
          burn_in: int = 400, seed: int = 0, tv_tol: float = 0.03):
    vals, pmf = exact_energy_pmf(model, L, beta, J, h, q)
    samples, energies = cpp_ext.collect(kind, L=L, B=B, beta=beta, J=J, h=h, q=q,
                                        burn_in=burn_in, n_collect=n_collect,
                                        collect_every=collect_every, seed=seed)
    # (a) Hamiltonian consistency between C++ and the reference implementation
    if model == "ising":
        spins = torch.from_numpy(samples.astype(np.float32) * 2.0 - 1.0)
        ref = ising2d_ham(spins, J=J, h=h).numpy()
    else:
        ref = potts2d_ham(torch.from_numpy(samples.astype(np.int32)), J=J, q=q).numpy()
    ham_err = float(np.max(np.abs(energies - ref)))

    # (b) distributional agreement
    emp = empirical_energy_pmf(energies, vals)
    tv = 0.5 * float(np.abs(emp - pmf).sum())

    # (c) label / spin-flip symmetry of a single site.  The energy distribution
    # is blind to a biased *proposal* (a uniform cyclic drift leaves energies
    # almost unchanged) but a single-site marginal is not: every Potts state
    # must appear with probability 1/q, and the mean Ising spin must vanish.
    if model == "potts":
        marg = np.bincount(samples[:, 0].astype(np.int64), minlength=q) / samples.shape[0]
        label_dev = float(np.abs(marg - 1.0 / q).max())
    else:
        mag = float((samples.astype(np.float64) * 2.0 - 1.0).mean())
        label_dev = abs(mag)

    ok = (ham_err < 1e-4) and (tv < tv_tol) and (label_dev < 0.02)
    return {
        "model": model, "sampler": kind, "L": L, "beta": beta, "q": q if model == "potts" else None,
        "n_samples": int(samples.shape[0]), "hamiltonian_max_err": ham_err,
        "tv_energy": round(tv, 5), "tv_tol": tv_tol,
        "label_symmetry_dev": round(label_dev, 5), "pass": bool(ok),
    }


def main() -> int:
    if not cpp_ext.available():
        print("native library missing; run scripts/build_cpp.ps1")
        return 2

    print(f"library: {cpp_ext.version()}  threads={cpp_ext.threads()}")
    cases = [
        # Ising at criticality, L=4 (exact)
        dict(model="ising", kind="ising_mh", L=4, beta=0.4407),
        dict(model="ising", kind="ising_sw", L=4, beta=0.4407),
        dict(model="ising", kind="ising_sw", L=4, beta=0.28),
        # Potts q=3, L=3 (exact), critical and ordered
        dict(model="potts", kind="potts_mh", L=3, beta=1.005, q=3),
        dict(model="potts", kind="potts_glauber", L=3, beta=1.005, q=3),
        dict(model="potts", kind="potts_sw", L=3, beta=1.005, q=3),
        dict(model="potts", kind="potts_sw", L=3, beta=1.2, q=3),
        dict(model="potts", kind="potts_glauber", L=3, beta=0.5, q=3),
    ]

    results = []
    for c in cases:
        r = check(**c)
        results.append(r)
        flag = "PASS" if r["pass"] else "FAIL"
        print(f"[{flag}] {r['sampler']:14s} L={r['L']} beta={r['beta']:<7} "
              f"ham_err={r['hamiltonian_max_err']:.2e} TV={r['tv_energy']:.4f} "
              f"label_dev={r['label_symmetry_dev']:.4f} (n={r['n_samples']})")

    Path(ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "validate_samplers.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    n_fail = sum(not r["pass"] for r in results)
    print(f"\n{len(results) - n_fail}/{len(results)} passed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
