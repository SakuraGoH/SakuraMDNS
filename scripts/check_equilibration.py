"""L=16 equilibration cross-check: do all kernels reach the same energy?

The L=4 / L=3 exact tests validate the *transition kernels* against the true
distribution, but they say nothing about whether a chain has equilibrated at
the lattice sizes we actually train on.  Here we run every sampler for the same
wall-clock-ish budget from a random start and compare the mean energy per site
against Swendsen-Wang, which is ergodic on the relevant time scale.

A disagreement of local updates (MH / Glauber) with SW in the ordered phase is
not a bug: it is critical slowing down, and it is exactly why cluster
algorithms are needed there.

Usage:  python scripts/check_equilibration.py
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

from mdns.samplers import cpp_ext  # noqa: E402

BURN = 2000
N_COLLECT = 100
EVERY = 10
B = 256


def energy_per_site(kind: str, *, model: str, L: int, beta: float, q: int) -> tuple[float, float]:
    cpp_ext.set_threads(4)
    _, e = cpp_ext.collect(kind, L=L, B=B, beta=beta, J=1.0, h=0.0, q=q,
                           burn_in=BURN, n_collect=N_COLLECT, collect_every=EVERY, seed=11)
    d = L * L
    e = e / d
    return float(e.mean()), float(e.std(ddof=1) / np.sqrt(e.size))


def main() -> int:
    if not cpp_ext.available():
        print("native library missing; run scripts/build_cpp.ps1")
        return 2

    out = {"settings": {"burn_in_sweeps": BURN, "n_collect": N_COLLECT,
                        "collect_every": EVERY, "B": B}, "models": {}}

    for model, beta, kinds, q in [
        ("ising", 0.4407, ["ising_mh", "ising_sw"], 0),
        ("ising", 0.6, ["ising_mh", "ising_sw"], 0),
        ("potts", 1.2, ["potts_mh", "potts_glauber", "potts_sw"], 3),
        ("potts", 1.005, ["potts_mh", "potts_glauber", "potts_sw"], 3),
    ]:
        L = 16
        ref_kind = kinds[-1]
        ref, ref_err = energy_per_site(ref_kind, model=model, L=L, beta=beta, q=q)
        print(f"\n{model} L={L} beta={beta}  reference ({ref_kind}) = {ref:.5f} +- {ref_err:.5f}")
        entry = {"beta": beta, "q": q or None, "reference_kind": ref_kind,
                 "reference_energy_per_site": round(ref, 6), "kernels": {}}
        for kind in kinds:
            mean, err = energy_per_site(kind, model=model, L=L, beta=beta, q=q)
            delta = mean - ref
            ok = abs(delta) <= max(4 * (err + ref_err), 2e-3)
            entry["kernels"][kind] = {"energy_per_site": round(mean, 6),
                                      "stderr": round(err, 6),
                                      "delta_vs_reference": round(delta, 6),
                                      "consistent": bool(ok)}
            print(f"  {kind:14s} {mean:.5f} +- {err:.5f}   delta={delta:+.5f}   "
                  f"{'consistent' if ok else 'MISMATCH'}")
        out["models"][f"{model}_beta{beta}"] = entry

    Path(ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "equilibration_l16.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/equilibration_l16.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
