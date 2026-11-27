"""Muon vs AdamW on the same denoising proxy task.

Protocol (identical to scripts/bench_architectures.py so the numbers are
comparable): ground-truth Potts configurations from the C++ Swendsen-Wang
kernel, mask each site with prob lambda ~ U(0,1), cross-entropy on masked sites,
evaluate at fixed lambda=0.75 on a held-out set.

Muon and AdamW have different natural learning-rate scales, so we first run a
short per-optimizer LR sweep and then compare the two best configurations at a
fixed step count.  Comparing at a single arbitrary LR would just measure which
optimizer happened to like that number.

Usage:  python scripts/bench_optimizers.py [--quick]
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bench_architectures import BATCH, BETA, evaluate, make_data, train_loss  # noqa: E402
from mdns import device as dev_mod  # noqa: E402
from mdns.models.registry import build  # noqa: E402
from mdns.optim import build_optimizer, describe, split_params  # noqa: E402
from mdns.optim.muon import zeropower_via_newtonschulz5  # noqa: E402

PRESET = "res-s"
Q = 3
STEPS_COMPARE = 300
STEPS_SWEEP = 120


def orthogonality_check(dev) -> dict:
    """Singular values of the orthogonalised update must be ~1."""
    torch.manual_seed(0)
    G = torch.randn(64, 32, device=dev)
    O = zeropower_via_newtonschulz5(G, steps=5).float()
    sv = torch.linalg.svdvals(O)
    return {"min_sv": float(sv.min()), "max_sv": float(sv.max()),
            "mean_sv": float(sv.mean()), "sv_std": float(sv.std())}


def run(model, opt, tr, va, steps: int, seed: int = 1234):
    dev = tr.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    n = tr.shape[0]
    dev_mod.synchronize(dev)
    t0 = time.perf_counter()
    first = last = float("nan")
    for i in range(steps):
        idx = torch.randint(0, n, (BATCH,), device=dev, generator=gen)
        loss = train_loss(model, tr[idx], gen, mask_token=Q)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.detach())
        if i == 0:
            first = last
    dev_mod.synchronize(dev)
    dt = time.perf_counter() - t0
    val_gen = torch.Generator(device=dev)
    val_gen.manual_seed(4242)
    return {"steps": steps, "seconds": round(dt, 2), "first_loss": round(first, 4),
            "last_loss": round(last, 4), "val_nll": round(evaluate(model, va, val_gen, Q, 0.75), 4),
            "steps_per_s": round(steps / dt, 2)}


def make(preset: str, dev, kind: str, lr: float, adamw_lr: float = 1e-3):
    m = build(preset, L=16, vocab_size=Q + 1, device=dev)
    opt = build_optimizer(m, kind=kind, lr=lr, adamw_lr=adamw_lr, momentum=0.95, ns_steps=5)
    return m, opt


def main() -> int:
    quick = "--quick" in sys.argv
    dev = dev_mod.resolve("auto")
    steps_sweep = 60 if quick else STEPS_SWEEP
    steps_cmp = 120 if quick else STEPS_COMPARE
    print(f"device: {dev_mod.describe(dev)} | preset={PRESET} L=16 q={Q} beta={BETA}")
    print(f"orthogonality of the Newton-Schulz output: {orthogonality_check(dev)}")

    tr, va = make_data(16, Q, BETA, n_train=8192, n_val=2048, device=dev)
    m0, _ = make(PRESET, dev, "muon", 0.02)
    print(f"param split: {describe(m0)}")
    del m0

    # ---- per-optimizer LR sweep -------------------------------------------
    grid = {"adamw": [5e-4, 1e-3, 2e-3, 4e-3], "muon": [5e-3, 1e-2, 2e-2, 4e-2]}
    sweep = []
    for kind, lrs in grid.items():
        for lr in lrs:
            m, opt = make(PRESET, dev, kind, lr)
            r = run(m, opt, tr, va, steps_sweep)
            sweep.append({"optimizer": kind, "lr": lr, **r})
            print(f"  {kind:6s} lr={lr:<7g} val_nll={r['val_nll']:.4f} "
                  f"({r['steps_per_s']:.1f} steps/s)")
            del m, opt

    best = {}
    for kind in grid:
        cands = [r for r in sweep if r["optimizer"] == kind]
        best[kind] = min(cands, key=lambda r: r["val_nll"])
    print(f"\nbest LR: adamw={best['adamw']['lr']:g} muon={best['muon']['lr']:g}")

    # ---- head-to-head at fixed steps --------------------------------------
    final = []
    for kind in ("adamw", "muon"):
        m, opt = make(PRESET, dev, kind, best[kind]["lr"])
        r = run(m, opt, tr, va, steps_cmp)
        final.append({"optimizer": kind, "lr": best[kind]["lr"], **r})
        print(f"  {kind:6s} lr={best[kind]['lr']:<7g} steps={r['steps']} "
              f"val_nll={r['val_nll']:.4f} first={r['first_loss']:.3f} last={r['last_loss']:.3f} "
              f"{r['steps_per_s']:.1f} steps/s")
        del m, opt

    base = [r for r in final if r["optimizer"] == "adamw"][0]
    mu = [r for r in final if r["optimizer"] == "muon"][0]
    print(f"\nMuon vs AdamW @ {steps_cmp} steps: "
          f"val_nll {mu['val_nll']:.4f} vs {base['val_nll']:.4f} "
          f"(delta {mu['val_nll'] - base['val_nll']:+.4f}); "
          f"speed {mu['steps_per_s']:.1f} vs {base['steps_per_s']:.1f} steps/s")

    Path(ROOT / "results").mkdir(exist_ok=True)
    out = {"device": dev_mod.describe(dev), "preset": PRESET, "L": 16, "q": Q,
           "beta": BETA, "orthogonality": orthogonality_check(dev),
           "param_split": describe(build(PRESET, L=16, vocab_size=Q + 1, device=dev)),
           "lr_sweep": sweep, "final": final}
    p = ROOT / "results" / "bench_optimizers.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
