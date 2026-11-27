"""Measure how fast *usable* training data is produced, and which knob matters.

The loop's training data is the set of samples the model scans for itself, so the
quantity worth maximising is not "samples per second" but

    effective samples / second  =  raw samples / second  *  (ESS / batch)

because a sample carries gradient only in proportion to its importance weight
(ESS/N is exp(-Var[log w]) to first order).  Three things enter:

  * raw rate        -- governed by scan_batch, the number of ops per forward and
                       whether the forward is fused (torch.compile)
  * ESS/batch       -- a property of how well the model matches pi, i.e. of
                       training quality, not of the benchmark
  * duty cycle      -- fraction of wall-clock spent scanning vs training, set by
                       steps_per_round / resample_every

This script measures the first and third, and reports ESS so you can fold in the
second.

    python scripts/bench_throughput.py --L 16 --batches 512,1024,2048
    python scripts/bench_throughput.py --compile            # try fused forward
    python scripts/bench_throughput.py --ckpt runs/x/best.pt  # ESS on a real model
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from mdns import device as dev_mod  # noqa: E402
from mdns.losses import ess as ess_fn  # noqa: E402
from mdns.losses import scan  # noqa: E402
from mdns.models.registry import build, build_from_config  # noqa: E402
from mdns.physics import potts2d_ham  # noqa: E402


def timed(fn, reps: int = 2):
    """Best-of-`reps` wall time, synchronised on the accelerator."""
    best = float("inf")
    out = None
    for _ in range(reps):
        dev_mod.synchronize()
        t0 = time.perf_counter()
        out = fn()
        dev_mod.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="nbt-2")
    ap.add_argument("--n-blocks", type=int, default=6)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--compression-ratio", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--beta", type=float, default=1.2)
    ap.add_argument("--batches", default="512,1024,2048")
    ap.add_argument("--steps-per-round", type=int, default=600)
    ap.add_argument("--train-batch", type=int, default=128)
    ap.add_argument("--num-replicates", type=int, default=8)
    ap.add_argument("--ckpt", default=None, help="checkpoint to measure ESS on")
    ap.add_argument("--compile", action="store_true", help="also time a compiled forward")
    ap.add_argument("--no-autocast", action="store_true",
                    help="keep the model in bf16 and skip autocast (needed for compile)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/bench_throughput.json")
    args = ap.parse_args()

    dev = dev_mod.resolve(args.device)
    print(f"device : {dev_mod.describe(dev)}")
    print(f"arch   : {args.arch} blocks={args.n_blocks} C={args.channels} "
          f"ratio={args.compression_ratio} heads={args.n_heads}  L={args.L} q={args.q} beta={args.beta}")

    over = dict(n_blocks=args.n_blocks, channels=args.channels,
                compression_ratio=args.compression_ratio, n_heads=args.n_heads)

    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
        cfg = dict(ck["arch"])
        name = ck.get("sakura_name", cfg.get("name", "?"))
        model = build_from_config(cfg, L=cfg["L"], vocab_size=cfg["vocab_size"], device=dev)
        model.load_state_dict(ck["model"])
        L, q = cfg["L"], cfg["vocab_size"] - 1
        print(f"model  : {name} (weights from {args.ckpt})")
    else:
        model = build(args.arch, L=args.L, vocab_size=args.q + 1, device=dev, **over)
        L, q = args.L, args.q
        print(f"model  : {model.name} (random init -> ESS will be degenerate; pass --ckpt "
              f"to measure ESS on trained weights)")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params : {n_params:,}")

    if args.no_autocast:
        dev_mod.autocast = lambda *a, **k: nullcontext()
        model = model.to(torch.bfloat16)
        print("autocast disabled, weights cast to bf16")

    reward = lambda x: -args.beta * potts2d_ham(x, args.J if hasattr(args, "J") else 1.0, q)

    # ---------------------------------------------------------------- scans
    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    rows = []
    print(f"\n{'batch':>6} {'scan(s)':>9} {'samp/s':>10} {'ESS/N':>7} {'eff/s':>9} "
          f"{'vs b=1024':>10}")
    base = None
    for B in batches:
        scan(model, reward, 128, dev)            # warmup on a small batch
        dt, (x, log_rnd) = timed(lambda B=B: scan(model, reward, B, dev), reps=2)
        rate = B / dt
        e = ess_fn(log_rnd)
        eff = rate * e
        if B == 1024:
            base = rate
        rows.append({"batch": B, "scan_s": round(dt, 3), "samples_per_s": round(rate, 1),
                     "ess_over_batch": round(e, 4), "effective_per_s": round(eff, 1)})
        print(f"{B:>6} {dt:>9.2f} {rate:>10.1f} {e:>7.4f} {eff:>9.1f}", end="")
        print(f" {rate/base:>9.2f}x" if base else "")

    # --------------------------------------------------- forward-level timings
    x1 = torch.randint(0, q + 1, (1024, L * L), device=dev)
    fwd = {}
    with torch.no_grad():
        dt, _ = timed(lambda: model(x1), reps=3)
        fwd["eager"] = round(dt * 1000, 2)
        print(f"\nforward @B=1024: eager {fwd['eager']:.2f} ms")
        if args.compile:
            try:
                cw = torch.compile(model, mode="reduce-overhead")
                dt, _ = timed(lambda: cw(x1), reps=3)
                fwd["compiled"] = round(dt * 1000, 2)
                print(f"forward @B=1024: compiled {fwd['compiled']:.2f} ms "
                      f"({fwd['eager'] / fwd['compiled']:.2f}x)")
            except Exception as exc:                     # noqa: BLE001
                fwd["compiled"] = f"failed: {type(exc).__name__}"
                print(f"forward @B=1024: compile failed -> {type(exc).__name__}: {str(exc)[:120]}")

    # ------------------------------------------------------------- duty cycle
    print(f"\nround projection (steps_per_round={args.steps_per_round}, train_batch="
          f"{args.train_batch}x{args.num_replicates}):")
    print(f"{'resample':>9} {'scans':>6} {'samples':>8} {'scan_s':>8} {'train_s':>8} "
          f"{'round_s':>8} {'samp/s':>8}")
    proj = []
    scan_row = {r["batch"]: r for r in rows}
    for B in batches:
        s = scan_row[B]
        for every in (5, 10, 25, 50):
            n_scans = 1 + (args.steps_per_round - 1) // every
            samples = n_scans * B
            scan_s = n_scans * s["scan_s"]
            # one training step costs roughly a forward+backward at train_batch
            with torch.no_grad():
                dt_f, _ = timed(lambda: model(torch.randint(0, q + 1, (args.train_batch, L * L), device=dev)), reps=2)
            train_s = args.steps_per_round * dt_f * 3.0 * (args.num_replicates ** 0.35)
            round_s = scan_s + train_s
            proj.append({"batch": B, "resample_every": every, "scans_per_round": n_scans,
                         "samples_per_round": samples, "round_s": round(round_s, 1),
                         "samples_per_s": round(samples / round_s, 1),
                         "effective_per_s": round(samples / round_s * s["ess_over_batch"], 1)})
            print(f"{every:>9} {n_scans:>6} {samples:>8} {scan_s:>8.0f} {train_s:>8.0f} "
                  f"{round_s:>8.0f} {samples / round_s:>8.1f}")

    Path(ROOT).joinpath(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    with open(ROOT / args.out, "w", encoding="utf-8") as f:
        json.dump({"device": dev_mod.describe(dev), "arch": model.name if hasattr(model, "name") else "?",
                   "params": n_params, "L": L, "q": q, "beta": args.beta,
                   "scans": rows, "forward_ms": fwd, "round_projection": proj}, f, indent=2)
    print(f"\nwrote {args.out}")
    best = max(proj, key=lambda r: r["effective_per_s"])
    print(f"best projected effective rate: batch={best['batch']} "
          f"resample_every={best['resample_every']} -> {best['effective_per_s']} eff/s "
          f"({best['samples_per_s']} raw/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
