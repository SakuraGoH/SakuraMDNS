"""Architecture benchmark + compression-ratio study.

Two questions:
  1. what does each registered preset cost (params, forward FLOPs, throughput)?
  2. given a fixed *wall-clock* training budget, which compression ratio gives
     the best denoising quality per second?

Part 2 is a *proxy* study, not the MDNS objective.  We take ground-truth Potts
configurations from the C++ Swendsen-Wang kernel (fast, exact-validated) and
train each architecture as a masked denoising autoencoder: mask each site with
prob lambda ~ U(0,1), predict the clean token, cross-entropy on masked sites.

Evaluation uses *fixed* high masking rates (0.5 / 0.75 / 0.9).  At beta=1.2 the
lattice is nearly fully ordered and low masking rates give a saturated NLL of
~0 (a single visible neighbour already determines the site), which cannot rank
architectures.  At the critical beta=1.005 with 75-90% of sites hidden, the
prediction genuinely requires global inference -- that is what we score.

Usage:  python scripts/bench_architectures.py [--quick] [--L 16]
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mdns import device as dev_mod  # noqa: E402
from mdns.models.registry import build, list_presets  # noqa: E402
from mdns.samplers import cpp_ext  # noqa: E402

TIME_BUDGET_S = 30.0
LR = 1e-3
BATCH = 64
N_TRAIN = 8192
N_VAL = 2048
EVAL_LAMBDAS = (0.5, 0.75, 0.9)
PRIMARY_LAMBDA = 0.75
BETA = 1.005          # critical: maximal fluctuations, non-saturated predictions


# ------------------------------------------------------------------- model cost
def forward_flops(model, x) -> int:
    try:
        from torch.utils.flop_counter import FlopCounterMode

        with FlopCounterMode(display=False) as fc:
            with torch.no_grad():
                model(x)
        return int(fc.get_total_flops())
    except Exception:
        return -1


def throughput(model, x, iters: int = 20) -> float:
    dev = x.device
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(3):
        loss = model(x).float().logsumexp(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    dev_mod.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        loss = model(x).float().logsumexp(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    dev_mod.synchronize(dev)
    return x.shape[0] * iters / (time.perf_counter() - t0)


# ------------------------------------------------------------------ proxy data
def make_data(L: int, q: int, beta: float, n_train: int, n_val: int, device):
    cpp_ext.set_threads(4)
    need = n_train + n_val
    n_collect = max(4, math.ceil(need / 1024))
    samples, _ = cpp_ext.collect("potts_sw", L=L, B=1024, beta=beta, J=1.0, q=q,
                                 burn_in=400, n_collect=n_collect, collect_every=10, seed=5)
    samples = samples[:need].astype(np.int64)
    assert samples.shape[0] == need, (samples.shape, need)
    tr = torch.from_numpy(samples[:n_train]).to(device)
    va = torch.from_numpy(samples[n_train:need]).to(device)
    return tr, va


def train_loss(model, batch, gen, mask_token: int) -> torch.Tensor:
    B, D = batch.shape
    lam = torch.rand(B, 1, device=batch.device, generator=gen)
    masked = torch.rand(batch.shape, device=batch.device, generator=gen) < lam
    perturbed = torch.where(masked, torch.full_like(batch, mask_token), batch)
    logits = model(perturbed)
    ce = -torch.gather(logits, -1, batch[..., None]).squeeze(-1)
    return (ce * masked).sum() / masked.sum().clamp_min(1)


@torch.no_grad()
def evaluate(model, va, gen, mask_token: int, lam: float, reps: int = 2) -> float:
    model.eval()
    vals = []
    for _ in range(reps):
        masked = torch.rand(va.shape, device=va.device, generator=gen) < lam
        perturbed = torch.where(masked, torch.full_like(va, mask_token), va)
        logits = model(perturbed)
        ce = -torch.gather(logits, -1, va[..., None]).squeeze(-1)
        vals.append(float((ce * masked).sum() / masked.sum().clamp_min(1)))
    model.train()
    return float(np.mean(vals))


def train_steps(model, tr, va, q: int, steps: int, seed: int = 1234):
    """Fixed *number of steps*: throughput on this box swings 10x between runs,
    so a fixed-time budget would rank architectures by measurement noise."""
    dev = tr.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    n = tr.shape[0]
    dev_mod.synchronize(dev)
    t0 = time.perf_counter()
    last = float("nan")
    for _ in range(steps):
        idx = torch.randint(0, n, (BATCH,), device=dev, generator=gen)
        loss = train_loss(model, tr[idx], gen, mask_token=q)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.detach())
    dev_mod.synchronize(dev)
    dt = time.perf_counter() - t0
    val_gen = torch.Generator(device=dev)
    val_gen.manual_seed(4242)          # identical mask stream for every architecture
    nll = {f"nll_lam{l}": round(evaluate(model, va, val_gen, q, l), 4) for l in EVAL_LAMBDAS}
    nll["val_nll"] = nll[f"nll_lam{PRIMARY_LAMBDA}"]
    return {"steps": steps, "seconds": round(dt, 2), "train_nll": round(last, 4),
            "steps_per_s": round(steps / max(dt, 1e-9), 2), **nll}


# ------------------------------------------------------------------------ main
def main() -> int:
    quick = "--quick" in sys.argv
    L = 16
    for i, a in enumerate(sys.argv):
        if a == "--L":
            L = int(sys.argv[i + 1])
    q, N = 3, 4
    dev = dev_mod.resolve("auto")
    steps = 100 if quick else 300
    budget = steps
    print(f"device: {dev_mod.describe(dev)} | L={L} q={q} beta={BETA} "
          f"time_budget={budget}s batch={BATCH}")

    tr, va = make_data(L, q, BETA, N_TRAIN, N_VAL, dev)
    print(f"proxy data: train={tuple(tr.shape)} val={tuple(va.shape)} "
          f"(ground truth: C++ Swendsen-Wang)")

    x = torch.randint(0, N, (BATCH, L * L), device=dev)
    rows = []
    quality_presets = ["res-s", "res-m", "vit-64x6", "nbt-1", "nbt-2", "nbt-4",
                       "nbt-8", "nbt-4-noattn"]
    for name in list_presets():
        m = build(name, L=L, vocab_size=N, device=dev)
        nparam = m.num_params()
        flops = forward_flops(m, x)
        thr = max(throughput(m, x, iters=6 if quick else 12) for _ in range(3))
        r = (train_steps(m, tr, va, q, steps=steps) if name in quality_presets
             else {"steps": 0, "seconds": 0.0, "train_nll": float("nan"),
                   "steps_per_s": 0.0, **{f"nll_lam{l}": float("nan") for l in EVAL_LAMBDAS},
                   "val_nll": float("nan")})
        rows.append({"preset": name, "params": nparam, "fwd_flops": flops,
                     "fwd_bwd_samples_per_s": round(thr, 1), "config": m.config(), **r})
        print(f"  {name:14s} params={nparam/1e6:7.4f}M flops={flops/1e9:8.3f}G "
              f"thr={thr:8.1f}/s steps={r['steps']:5d} "
              f"nll(0.5/0.75/0.9)={r['nll_lam0.5']:.3f}/{r['nll_lam0.75']:.3f}/"
              f"{r['nll_lam0.9']:.3f}")

    Path(ROOT / "results").mkdir(exist_ok=True)
    out = {"device": dev_mod.describe(dev), "L": L, "q": q, "beta": BETA,
           "train_steps": steps, "batch": BATCH, "lr": LR,
           "primary_lambda": PRIMARY_LAMBDA, "rows": rows}
    p = ROOT / "results" / "bench_architectures.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {p}")

    best = min(rows, key=lambda r: r["val_nll"])
    best_eff = min(rows, key=lambda r: r["val_nll"] / math.sqrt(max(r["params"], 1) / 1e6))
    rows = [r for r in rows if r["steps"] > 0]
    print(f"\nbest val NLL(lam=0.75): {best['preset']} "
          f"({best['val_nll']:.4f}, {best['params']/1e6:.4f}M params, {best['steps']} steps)")
    print(f"best val NLL per sqrt(params): {best_eff['preset']} "
          f"({best_eff['val_nll']:.4f}, {best_eff['params']/1e6:.4f}M)")

    print("\n| preset | params (M) | fwd GFLOPs | samples/s | steps | NLL@0.5 | NLL@0.75 | NLL@0.9 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: r["val_nll"]):
        print(f"| {r['preset']} | {r['params']/1e6:.4f} | {r['fwd_flops']/1e9:.3f} "
              f"| {r['fwd_bwd_samples_per_s']:.0f} | {r['steps']} | {r['nll_lam0.5']:.4f} "
              f"| {r['nll_lam0.75']:.4f} | {r['nll_lam0.9']:.4f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
