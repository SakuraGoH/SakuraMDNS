"""Physics of the 2-D Ising and q-state Potts models (periodic boundaries).

Only *observables* live here -- Hamiltonians, magnetisation, correlations,
exact enumeration on tiny lattices, plotting.  Sampling is done by the C++
kernels in ``mdns.samplers.cpp_ext`` (fast) or ``mdns.samplers.gpu``.

The slow pure-python Metropolis / Glauber / Swendsen-Wang implementations that
this module replaced were removed: they are 50-500x slower than the C++ kernels
for identical transition kernels (see results/RESULTS_item1.md) and nothing in
the training loop needs them.

Conventions
-----------
Ising  ``S`` holds spins in {-1, +1}; the *token* convention used by the model
       is {0, 1}, converted with ``2*S-1`` / ``(S+1)//2``.
       H = -J * sum_{<ij>} s_i s_j - h * sum_i s_i
Potts  ``S`` holds states in {0 .. q-1}.
       H = -J * (#edges whose endpoints carry the same state)
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

# Palette used throughout the figures (paper colours).
ISING_COLORS = ["#313342", "#DEB4B2"]
POTTS_COLORS = ["#540D6E", "#EE4266", "#FFD23F"]


def _lattice_side(n: int) -> int:
    L = int(round(math.isqrt(int(n))))
    if L * L != int(n):
        raise ValueError(f"{n} sites is not a perfect square")
    return L


# =========================================================== Ising =========
def ising2d_ham(S: torch.Tensor, J: float = 1.0, h: float = 0.0) -> torch.Tensor:
    """H = -J sum_<ij> s_i s_j - h sum_i s_i for a batch of configurations.

    Args:
        S: (B, L*L) spins in {-1, +1}
    Returns:
        (B,) energies
    """
    B, D = S.shape
    L = _lattice_side(D)
    S = S.view(B, L, L)
    Sx = torch.roll(S, shifts=-1, dims=1)
    Sy = torch.roll(S, shifts=-1, dims=2)
    interaction = -J * torch.sum(S * (Sx + Sy), dim=(1, 2))
    magnetic = -h * torch.sum(S, dim=(1, 2))
    return interaction + magnetic


def ising2d_get_all_configs(L: int = 4, device="cpu") -> torch.Tensor:
    """All 2^(L^2) configurations in increasing binary order, spins in {-1,+1}."""
    B = 2 ** (L * L)
    if L * L > 20:
        raise ValueError("refusing to enumerate more than 2^20 configurations")
    bits = torch.arange(L * L - 1, -1, -1, device=device)
    return (((torch.arange(B, device=device)[:, None] >> bits) & 1) * 2 - 1).to(torch.int8)


def ising2d_part_func(L: int = 4, beta: float = .5, J: float = 1.0, h: float = 0.0,
                      device="cpu") -> float:
    """Exact partition function by enumeration (L <= 4)."""
    configs = ising2d_get_all_configs(L, device=device)
    return torch.exp(-beta * ising2d_ham(configs.float(), J=J, h=h)).sum().item()


def ising2d_get_pmf(L: int, J: float = 1.0, h: float = 0.0, beta: float = 1.0,
                    device="cpu") -> torch.Tensor:
    """Exact pmf over all configurations (same ordering as get_all_configs)."""
    configs = ising2d_get_all_configs(L, device=device).float()
    return (-beta * ising2d_ham(configs, J=J, h=h)).softmax(dim=0)


def ising2d_mag(S: torch.Tensor) -> torch.Tensor:
    """(B,) magnetisation per configuration (spins in {-1,+1})."""
    return S.float().mean(dim=1)


def ising2d_row_col_mag(S: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """(B, L) magnetisation of every row (axis=0) or column (axis=1)."""
    B, D = S.shape
    L = _lattice_side(D)
    return S.view(B, L, L).float().mean(dim=axis + 1)


def ising2d_2pt_corr(S: torch.Tensor, rx: int, ry: int) -> torch.Tensor:
    """(B,) <s_i s_{i+rx}> + <s_i s_{i+ry}> averaged over sites."""
    B, D = S.shape
    L = _lattice_side(D)
    S = S.view(B, L, L)
    Sx = torch.roll(S, shifts=-rx, dims=1)
    Sy = torch.roll(S, shifts=-ry, dims=2)
    return (S * (Sx + Sy)).float().view(B, -1).mean(dim=1)


def ising_2pt_corr_direction(S, r: int, use_x: bool = True, use_y: bool = False,
                             symmetric: bool = True) -> float:
    """Scalar 2-point correlation at distance ``r`` along one axis.

    ``symmetric=True`` averages the +r and -r displacements (the upstream helper
    only used +r, which biases the estimate on a finite torus).
    """
    if not isinstance(S, torch.Tensor):
        S = torch.from_numpy(np.asarray(S))
    S = S.reshape(-1, _lattice_side(S.shape[-1]), _lattice_side(S.shape[-1])).float()
    total = None
    for axis, use in ((1, use_x), (2, use_y)):
        if not use:
            continue
        nb = torch.roll(S, shifts=-r, dims=axis)
        if symmetric and r:
            nb = 0.5 * (nb + torch.roll(S, shifts=r, dims=axis))
        term = (S * nb).mean()
        total = term if total is None else total + term
    if total is None:
        raise ValueError("enable at least one direction")
    return total


def ising2d_emp_dist(samples: torch.Tensor) -> torch.Tensor:
    """Empirical pmf over all 2^(L^2) configurations, same ordering as above."""
    if not isinstance(samples, torch.Tensor):
        samples = torch.from_numpy(np.asarray(samples))
    B, N = samples.shape
    bin_samples = ((samples + 1) // 2).to(torch.int32)
    bits = torch.arange(N - 1, -1, -1, device=samples.device)
    indices = (bin_samples.long() << bits).sum(dim=1)
    counts = torch.bincount(indices, minlength=2 ** N)
    return counts.float() / B


# =========================================================== Potts =========
def potts2d_ham(S: torch.Tensor, J: float = 1.0, q: int = 3) -> torch.Tensor:
    """H = -J * (#edges with equal endpoints) for a batch of configurations.

    Args:
        S: (B, L*L) states in {0 .. q-1}
    Returns:
        (B,) energies
    """
    B, D = S.shape
    L = _lattice_side(D)
    S = S.view(B, L, L)
    equal = ((S == torch.roll(S, shifts=-1, dims=2)).sum(dim=(1, 2))
             + (S == torch.roll(S, shifts=-1, dims=1)).sum(dim=(1, 2)))
    return -J * equal.to(torch.float32)


def get_all_potts_configs(L: int, q: int) -> torch.Tensor:
    """All q^(L^2) configurations, row-major, values in {0 .. q-1}."""
    if q ** (L * L) > 5e7:
        raise ValueError(f"refusing to enumerate q^(L^2) = {q ** (L * L):.2e} configurations")
    values = [np.arange(q) for _ in range(L * L)]
    configs = np.array(np.meshgrid(*values)).T.reshape(-1, L * L)
    return torch.from_numpy(configs)


def _potts_dominant_fraction(S: np.ndarray, q: int) -> np.ndarray:
    B = S.shape[0]
    most = np.zeros(B, dtype=np.int64)
    for b in range(B):
        most[b] = np.argmax(np.bincount(S[b].ravel(), minlength=q))
    return np.array([np.mean(S[b] == most[b]) for b in range(B)])


def potts2d_magnetization_all(S, q: int) -> np.ndarray:
    """(B,) fraction of sites in each configuration's most frequent state."""
    if isinstance(S, torch.Tensor):
        S = S.detach().cpu().numpy()
    if S.ndim == 2:
        S = S.reshape(S.shape[0], _lattice_side(S.shape[1]), _lattice_side(S.shape[1]))
    return _potts_dominant_fraction(S, q)


def potts2d_magnetization_ij(S, q: int) -> np.ndarray:
    """(L, L) site-resolved magnetisation (q*p_max-1)/(q-1) over the batch."""
    if isinstance(S, torch.Tensor):
        S = S.detach().cpu().numpy()
    if S.ndim == 2:
        L = _lattice_side(S.shape[1])
        S = S.reshape(S.shape[0], L, L)
    B, L, _ = S.shape
    flat = S.reshape(B, L * L)
    mag = np.zeros(L * L)
    for i in range(L * L):
        states = flat[:, i]
        top = np.bincount(states, minlength=q).argmax()
        mag[i] = (q * np.mean(states == top) - 1) / (q - 1)
    return mag.reshape(L, L)


def potts2d_magnetization(S, q: int, row: int | None = None,
                          col: int | None = None) -> float:
    """Magnetisation of one row or column, averaged over the batch."""
    if row is None and col is None:
        raise ValueError("specify row= or col=")
    if isinstance(S, torch.Tensor):
        S = S.detach().cpu().numpy()
    if S.ndim == 2:
        L = _lattice_side(S.shape[1])
        S = S.reshape(S.shape[0], L, L)
    line = S[:, row, :] if row is not None else S[:, :, col]
    counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=q), 1, line)
    top = counts.argmax(axis=1)
    return float(np.mean((q * np.mean(line == top[:, None], axis=1) - 1) / (q - 1)))


def potts_2pt_corr_direction(S, q: int, r: int = 1, use_x: bool = True,
                             use_y: bool = False, symmetric: bool = True) -> float:
    """Normalised Potts correlation C(r) = (<d(s_i,s_{i+r})> - 1/q) / (1 - 1/q)."""
    if not isinstance(S, torch.Tensor):
        S = torch.from_numpy(np.asarray(S))
    if S.ndim == 2:
        L = _lattice_side(S.shape[1])
        S = S.reshape(S.shape[0], L, L)
    total, n = None, 0
    for axis, use in ((1, use_x), (2, use_y)):
        if not use:
            continue
        nb = torch.roll(S, shifts=-r, dims=axis)
        if symmetric and r:
            nb = 0.5 * (torch.roll(S, shifts=r, dims=axis) + nb)
        term = (S == nb).float().mean()
        total = term if total is None else total + term
        n += 1
    if total is None:
        raise ValueError("enable at least one direction")
    return float(total / n - 1.0 / q) / (1.0 - 1.0 / q)


# =========================================================== plots =========
def visualize_ising(S, k_x: int, k_y: int):
    """Grid of k_x*k_y Ising configurations; accepts tokens in {0,1} or spins."""
    if isinstance(S, torch.Tensor):
        S = S.detach().cpu().numpy()
    S = np.asarray(S)
    if S.ndim == 2:
        L = _lattice_side(S.shape[1])
        S = S.reshape(-1, L, L)
    if S.max() <= 1:
        S = 2.0 * S - 1.0
    if k_x * k_y != S.shape[0]:
        raise ValueError(f"k_x*k_y={k_x * k_y} != number of samples {S.shape[0]}")
    fig, axes = plt.subplots(k_x, k_y, figsize=(1.5 * k_x, 1.5 * k_y), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    cmap = ListedColormap(ISING_COLORS)
    for i in range(S.shape[0]):
        axes[i].imshow(S[i], cmap=cmap, vmin=-1, vmax=1, origin="lower", interpolation="nearest")
        axes[i].axis("off")
    plt.tight_layout()
    return fig


def visualize_potts(S, q: int, k_x: int, k_y: int):
    """Grid of k_x*k_y Potts configurations (states 0..q-1)."""
    if isinstance(S, torch.Tensor):
        S = S.detach().cpu().numpy()
    S = np.asarray(S)
    if S.ndim == 2:
        L = _lattice_side(S.shape[1])
        S = S.reshape(-1, L, L)
    if k_x * k_y != S.shape[0]:
        raise ValueError(f"k_x*k_y={k_x * k_y} != number of samples {S.shape[0]}")
    fig, axes = plt.subplots(k_x, k_y, figsize=(1.5 * k_x, 1.5 * k_y), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    cmap = ListedColormap(POTTS_COLORS[:q])
    for i in range(S.shape[0]):
        axes[i].imshow(S[i], cmap=cmap, vmin=0, vmax=q - 1, origin="lower",
                       interpolation="nearest")
        axes[i].axis("off")
    plt.tight_layout()
    return fig
