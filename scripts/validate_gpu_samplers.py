"""Validate the GPU (XPU) samplers against exact distributions + CPU reference.

  * exact energy distribution on tiny lattices (Ising L=4, Potts L=3)
  * single-site label symmetry (catches biased proposals, which the energy
    distribution alone can miss)
  * L=16 equilibration cross-check against the C++ Swendsen-Wang kernel

Usage:  python scripts/validate_gpu_samplers.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mdns import device as dev_mod  # noqa: E402
from mdns.samplers import cpp_ext, gpu  # noqa: E402
from mdns.physics import ising2d_ham, potts2d_ham  # noqa: E402
from validate_samplers import empirical_energy_pmf, exact_energy_pmf  # noqa: E402


def gpu_check(kind, *, model, L, beta, J=1.0, h=0.0, q=3, B=1024, n_collect=40,
              collect_every=25, burn_in=400, seed=0, tv_tol=0.03, device=None):
    vals, pmf = exact_energy_pmf(model, L, beta, J, h, q)
    samples, energies, _ = gpu.collect(kind, L=L, B=B, beta=beta, J=J, h=h, q=q,
                                       burn_in=burn_in, n_collect=n_collect,
                                       collect_every=collect_every, device=device, seed=seed)
    if model == "ising":
        ref = ising2d_ham(torch.from_numpy(samples.astype(np.float32) * 2.0 - 1.0), J=J, h=h).numpy()
    else:
        ref = potts2d_ham(torch.from_numpy(samples.astype(np.int32)), J=J, q=q).numpy()
    ham_err = float(np.max(np.abs(energies - ref)))
    emp = empirical_energy_pmf(energies, vals)
    tv = 0.5 * float(np.abs(emp - pmf).sum())
    if model == "potts":
        marg = np.bincount(samples[:, 0].astype(np.int64), minlength=q) / samples.shape[0]
        label_dev = float(np.abs(marg - 1.0 / q).max())
    else:
        label_dev = abs(float((samples.astype(np.float64) * 2 - 1).mean()))
    ok = ham_err < 1e-4 and tv < tv_tol and label_dev < 0.02
    return {"sampler": kind, "model": model, "L": L, "beta": beta, "n": int(samples.shape[0]),
            "hamiltonian_max_err": ham_err, "tv_energy": round(tv, 5),
            "label_symmetry_dev": round(label_dev, 5), "pass": bool(ok)}


def equil_check(kind, *, model, L, beta, q, B=256, burn=2000, n_collect=40, every=25,
                device=None):
    """Compare mean energy/site with the C++ Swendsen-Wang reference."""
    cpp_ext.set_threads(4)
    ref_kind = "ising_sw" if model == "ising" else "potts_sw"
    _, e_ref = cpp_ext.collect(ref_kind, L=L, B=B, beta=beta, J=1.0, h=0.0, q=q,
                               burn_in=burn, n_collect=n_collect, collect_every=every, seed=11)
    _, e, _ = gpu.collect(kind, L=L, B=B, beta=beta, J=1.0, h=0.0, q=q, burn_in=burn,
                          n_collect=n_collect, collect_every=every, device=device, seed=11)
    d = L * L
    ref = float(e_ref.mean() / d)
    got = float(e.mean() / d)
    se = float(np.std(e / d) / np.sqrt(e.size) + np.std(e_ref / d) / np.sqrt(e_ref.size))
    ok = abs(got - ref) <= max(4 * se, 2e-3)
    return {"sampler": kind, "model": model, "beta": beta, "reference": round(ref, 5),
            "gpu": round(got, 5), "delta": round(got - ref, 5), "pass": bool(ok)}


def cross_check(kind, *, model, L, beta, q=3, B=512, burn=1000, n_collect=40, every=25,
                device=None, tol=0.05):
    """Compare the GPU energy distribution against the C++ Swendsen-Wang kernel.

    Needed for configurations where exact enumeration is out of reach, e.g.
    Potts q=3 on an even lattice (3^16 configurations).
    """
    cpp_ext.set_threads(4)
    ref_kind = "ising_sw" if model == "ising" else "potts_sw"
    _, e_ref = cpp_ext.collect(ref_kind, L=L, B=B, beta=beta, J=1.0, h=0.0, q=q,
                               burn_in=burn, n_collect=n_collect, collect_every=every, seed=11)
    _, e, _ = gpu.collect(kind, L=L, B=B, beta=beta, J=1.0, h=0.0, q=q, burn_in=burn,
                          n_collect=n_collect, collect_every=every, device=device, seed=11)
    e_ref, e = np.round(e_ref, 4), np.round(e, 4)
    vals = np.unique(np.concatenate([e_ref, e]))
    h1 = np.bincount(np.searchsorted(vals, e_ref), minlength=len(vals)).astype(float)
    h2 = np.bincount(np.searchsorted(vals, e), minlength=len(vals)).astype(float)
    h1 /= h1.sum()
    h2 /= h2.sum()
    tv = 0.5 * float(np.abs(h1 - h2).sum())
    d = L * L
    return {"sampler": kind, "model": model, "L": L, "beta": beta,
            "reference": ref_kind, "tv_vs_cpp": round(tv, 5),
            "mean_e_cpp": round(float(e_ref.mean() / d), 5),
            "mean_e_gpu": round(float(e.mean() / d), 5),
            "pass": bool(tv < tol)}


def main() -> int:
    dev = dev_mod.resolve("auto")
    print(f"device: {dev_mod.describe(dev)}")
    print(f"torch: {torch.__version__}")

    results, cross, equil = [], [], []
    # exact enumeration is only affordable on tiny lattices; checkerboard needs
    # an even L, so Potts exact tests use Wolff (valid for any L).
    for c in [
        dict(kind="ising_cb", model="ising", L=4, beta=0.4407),
        dict(kind="ising_wolff", model="ising", L=4, beta=0.4407),
        dict(kind="ising_wolff", model="ising", L=4, beta=0.6),
        dict(kind="potts_wolff", model="potts", L=3, beta=1.005, q=3),
        dict(kind="potts_wolff", model="potts", L=3, beta=1.2, q=3),
        dict(kind="potts_wolff", model="potts", L=3, beta=0.5, q=3),
    ]:
        r = gpu_check(device=dev, burn_in=300, n_collect=30, collect_every=20, **c)
        results.append(r)
        print(f"[{'PASS' if r['pass'] else 'FAIL'}] {r['sampler']:14s} L={r['L']} "
              f"beta={r['beta']:<7} ham_err={r['hamiltonian_max_err']:.2e} "
              f"TV={r['tv_energy']:.4f} label_dev={r['label_symmetry_dev']:.4f}")

    print("\ncross-check vs C++ SW (even L, exact enumeration infeasible):")
    for c in [
        dict(kind="potts_cb", model="potts", L=4, beta=1.005, q=3),
        dict(kind="potts_wolff", model="potts", L=4, beta=1.005, q=3),
        dict(kind="potts_cb", model="potts", L=8, beta=1.2, q=3),
        dict(kind="potts_wolff", model="potts", L=8, beta=1.2, q=3),
    ]:
        r = cross_check(device=dev, **c)
        cross.append(r)
        print(f"  [{'PASS' if r['pass'] else 'FAIL'}] {r['sampler']:14s} L={r['L']} "
              f"beta={r['beta']:<7} TV={r['tv_vs_cpp']:.4f} "
              f"meanE cpp={r['mean_e_cpp']:.5f} gpu={r['mean_e_gpu']:.5f}")

    print("\nL=16 equilibration vs C++ SW:")
    for c in [
        dict(kind="ising_cb", model="ising", L=16, beta=0.4407, q=0),
        dict(kind="ising_wolff", model="ising", L=16, beta=0.4407, q=0),
        dict(kind="potts_cb", model="potts", L=16, beta=1.2, q=3),
        dict(kind="potts_wolff", model="potts", L=16, beta=1.2, q=3),
        dict(kind="potts_wolff", model="potts", L=16, beta=1.005, q=3),
    ]:
        r = equil_check(device=dev, B=128, burn=400, n_collect=20, every=10, **c)
        equil.append(r)
        print(f"  [{'PASS' if r['pass'] else 'FAIL'}] {r['sampler']:14s} beta={r['beta']:<7} "
              f"cpp={r['reference']:.5f} gpu={r['gpu']:.5f} delta={r['delta']:+.5f}")

    Path(ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "validate_gpu_samplers.json", "w", encoding="utf-8") as f:
        json.dump({"exact": results, "cross_check": cross, "equilibration_l16": equil},
                  f, indent=2)

    allr = results + cross + equil
    fails = sum(not r["pass"] for r in allr)
    print(f"\n{len(allr) - fails}/{len(allr)} passed")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
