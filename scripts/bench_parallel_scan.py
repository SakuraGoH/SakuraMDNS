"""Can the GPU be filled better than one sequential scan at a time?

Three questions, one run:

A. Does `torch.compile` make the forward faster?  (Autocast is disabled for the
   compiled variant, which is what broke it before: the autocast context inside
   `LatticeNet.logits` makes dynamo re-trace endlessly.)
B. Does a bigger `scan_batch` raise throughput?  (Measured before as linear, i.e.
   no gain -- re-checked here.)
C. Do several *concurrent* scan workers scale?  Each scan is 256 sequential
   forwards, but different scans are independent, so N workers can overlap.  If
   one forward occupies only a few SMs, N of them should nearly add up.

Modes
-----
driver :  python scripts/bench_parallel_scan.py --L 16 --workers 1,2,4
worker :  python scripts/bench_parallel_scan.py --worker --scans 3   (internal)

The driver prints a table; `--json out.json` saves it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def get_model(args, dev, compile_it: bool = False, bf16: bool = False):
    import torch

    from mdns import device as dev_mod
    from mdns.models.registry import build

    over = dict(n_blocks=args.n_blocks, channels=args.channels,
                compression_ratio=args.compression_ratio, n_heads=args.n_heads)
    model = build(args.arch, L=args.L, vocab_size=args.q + 1, device=dev, **over)
    if bf16:
        # remove the autocast context from the forward path, then keep weights bf16
        from contextlib import nullcontext

        dev_mod.autocast = lambda *a, **k: nullcontext()
        model = model.to(torch.bfloat16)
    if compile_it:
        model = torch.compile(model)
    return model


def worker_main(args) -> int:
    """One scan worker: run `scans` scans and report the wall time on stdout."""
    import time as _t

    import torch

    from mdns import device as dev_mod
    from mdns.losses import scan
    from mdns.physics import potts2d_ham

    dev = dev_mod.resolve(args.device)
    torch.manual_seed(args.seed)
    model = get_model(args, dev)
    model.eval()
    reward = lambda x: -args.beta * potts2d_ham(x, 1.0, args.q)
    with torch.no_grad():
        scan(model, reward, 64, dev)                      # warmup
        dev_mod.synchronize()
        t0 = _t.perf_counter()
        for _ in range(args.scans):
            scan(model, reward, args.batch, dev)
        dev_mod.synchronize()
        dt = _t.perf_counter() - t0
    print(f"RESULT {args.scans * args.batch} {dt:.3f}")
    return 0


def driver_main(args) -> int:
    import torch

    from mdns import device as dev_mod
    from mdns.losses import scan
    from mdns.physics import potts2d_ham

    dev = dev_mod.resolve(args.device)
    print(f"device: {dev_mod.describe(dev)}")
    model = get_model(args, dev)
    print(f"model : {model.name if hasattr(model, 'name') else args.arch}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    reward = lambda x: -args.beta * potts2d_ham(x, 1.0, args.q)
    out = {"device": dev_mod.describe(dev), "L": args.L, "q": args.q, "beta": args.beta,
           "arch": args.arch, "blocks": args.n_blocks, "channels": args.channels,
           "ratio": args.compression_ratio, "heads": args.n_heads}

    # ---- A: forward variants ---------------------------------------------
    print("\n[A] forward @ B=1024 (per call)")
    x = torch.randint(0, args.q + 1, (1024, args.L ** 2), device=dev)

    def time_fwd(m, n=3):
        with torch.no_grad():
            for _ in range(2):
                m(x)
            dev_mod.synchronize()
            t = time.perf_counter()
            for _ in range(n):
                m(x)
            dev_mod.synchronize()
        return (time.perf_counter() - t) / n * 1000

    fwd = {"eager_autocast": round(time_fwd(model), 2)}
    print(f"  eager + autocast        : {fwd['eager_autocast']:8.2f} ms")
    if args.compile:
        try:
            model = None
            torch.cuda.empty_cache()
            m2 = get_model(args, dev, bf16=True)
            fwd["eager_bf16"] = round(time_fwd(m2), 2)
            print(f"  eager, bf16, no autocast: {fwd['eager_bf16']:8.2f} ms")
            m3 = get_model(args, dev, compile_it=True, bf16=True)
            fwd["compiled_bf16"] = round(time_fwd(m3), 2)
            print(f"  compiled, bf16          : {fwd['compiled_bf16']:8.2f} ms "
                  f"({fwd['eager_autocast'] / fwd['compiled_bf16']:.2f}x vs eager)")
            del m2, m3
        except Exception as exc:                       # noqa: BLE001
            fwd["compiled_bf16"] = f"failed: {type(exc).__name__}: {str(exc)[:150]}"
            print(f"  compile failed: {type(exc).__name__}: {str(exc)[:200]}")
    out["forward_ms"] = fwd
    model = None
    torch.cuda.empty_cache()

    # ---- B: single-process scan vs batch --------------------------------
    print("\n[B] single-process scan")
    model = get_model(args, dev)
    rows = []
    for B in [int(v) for v in args.batches.split(",") if v.strip()]:
        with torch.no_grad():
            scan(model, reward, 64, dev)
            dev_mod.synchronize()
            t = time.perf_counter()
            scan(model, reward, B, dev)
            dev_mod.synchronize()
            dt = time.perf_counter() - t
        rows.append({"batch": B, "scan_s": round(dt, 2), "samples_per_s": round(B / dt, 1),
                     "ms_per_sample": round(dt / B * 1000, 2)})
        print(f"  batch={B:<5} scan {dt:7.2f} s   {B/dt:7.1f} samples/s   "
              f"{dt/B*1000:5.2f} ms/sample")
    out["single"] = rows
    model = None
    torch.cuda.empty_cache()

    # ---- C: concurrent workers ------------------------------------------
    print("\n[C] concurrent scan workers (each does its own independent scans)")
    conc = []
    for K in [int(v) for v in args.workers.split(",") if v.strip()]:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--scans", str(args.scans), "--batch", str(args.batch),
               "--arch", args.arch, "--n-blocks", str(args.n_blocks),
               "--channels", str(args.channels),
               "--compression-ratio", str(args.compression_ratio),
               "--n-heads", str(args.n_heads), "--L", str(args.L),
               "--q", str(args.q), "--beta", str(args.beta)]
        procs = []
        t0 = time.perf_counter()
        for k in range(K):
            env = dict(os.environ)
            procs.append(subprocess.Popen(cmd + ["--seed", str(1000 + k)],
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          text=True, env=env))
        samples = 0.0
        for p in procs:
            so, se = p.communicate()
            for line in so.splitlines():
                if line.startswith("RESULT"):
                    samples += float(line.split()[1])
        wall = time.perf_counter() - t0
        rate = samples / wall if wall else 0.0
        conc.append({"workers": K, "wall_s": round(wall, 2), "samples": int(samples),
                     "samples_per_s": round(rate, 1)})
        print(f"  K={K:<3} wall {wall:7.2f} s   {samples:8.0f} samples   "
              f"{rate:7.1f} samples/s")
    out["concurrent"] = conc
    if conc:
        base = next((c for c in conc if c["workers"] == 1), None)
        if base and base["samples_per_s"]:
            print("\n  scaling vs K=1:")
            for c in conc:
                print(f"    K={c['workers']:<3} {c['samples_per_s']/base['samples_per_s']:.2f}x")
            out["scaling"] = {c["workers"]: round(c["samples_per_s"] / base["samples_per_s"], 3)
                              for c in conc}

    if args.json:
        Path(ROOT).joinpath(Path(args.json).parent).mkdir(parents=True, exist_ok=True)
        with open(ROOT / args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


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
    ap.add_argument("--batches", default="1024,2048,4096")
    ap.add_argument("--workers", default="1,2,4")
    ap.add_argument("--scans", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1024, help="batch used by workers")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json", default="results/bench_parallel_scan.json")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    return worker_main(args) if args.worker else driver_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
