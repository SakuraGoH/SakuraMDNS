"""Time the upstream MDNS training step against ours, on the same GPU.

The original project (train_potts.py) used RopeViT embed_dim=128, depth=4,
heads=4, batch_size=256, resample_every_n_step=10, wdce_num_replicates=8, AdamW.
Our config is nbt-2 (6 blocks / 64 ch / ratio 2 / 4 heads), scan_batch=1024,
resample_every=25, 600 steps/round, Muon, compiled.

Per optimizer step:
    original = scan(B=256) / 10          + train_step(256 x 8 rows)
    ours     = 24 * scan(B=1024) / 600   + train_step(128 x 8 rows)

Run on the 4080S:  python scripts/bench_upstream_step.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from mdns import device as dev_mod  # noqa: E402
from mdns.losses import masked_denoise_loss, scan  # noqa: E402
from mdns.models.registry import build  # noqa: E402
from mdns.models.ropevit import get_rope_vit_model  # noqa: E402
from mdns.optim import build_optimizer  # noqa: E402
from mdns.physics import potts2d_ham  # noqa: E402

L, Q, BETA = 16, 3, 1.2
D = L * L


def timeit(fn, reps=2, warm=1):
    for _ in range(warm):
        fn()
    dev_mod.synchronize()
    best = float("inf")
    for _ in range(reps):
        dev_mod.synchronize()
        t = time.perf_counter()
        fn()
        dev_mod.synchronize()
        best = min(best, time.perf_counter() - t)
    return best


def measure(tag, model, scan_batch, train_rows, num_replicates, compile_scan=False,
            compile_train=False):
    dev = next(model.parameters()).device
    reward = lambda x: -BETA * potts2d_ham(x, 1.0, Q)
    n_params = sum(p.numel() for p in model.parameters())
    out = {"tag": tag, "params": n_params, "scan_batch": scan_batch,
           "train_rows": train_rows}
    print(f"\n--- {tag}  params={n_params:,}")

    # forward at the two batch sizes of interest
    fwd = {}
    with torch.no_grad():
        for B in (256, 1024):
            x = torch.randint(0, Q + 1, (B, D), device=dev)
            dt = timeit(lambda: model(x), reps=3)
            fwd[B] = round(dt * 1000, 2)
            print(f"    forward  B={B:<5} {dt*1000:8.1f} ms")
    out["forward_ms"] = fwd

    # scan
    if compile_scan:
        import copy
        sm = copy.deepcopy(model).to(torch.bfloat16)
        sm.autocast_enabled = False
        sm.eval()
        sm = torch.compile(sm)
        scan(sm, reward, scan_batch, dev)                    # warm at the real shape
        dt = timeit(lambda: scan(sm, reward, scan_batch, dev), reps=2, warm=0)
        del sm
    else:
        dt = timeit(lambda: scan(model, reward, scan_batch, dev), reps=2, warm=1)
    out["scan_s"] = round(dt, 2)
    print(f"    scan     B={scan_batch:<5} {dt:8.2f} s   ({scan_batch/dt:6.1f} samples/s)")

    # training step: `train_rows` samples x num_replicates
    x = torch.randint(0, Q + 1, (train_rows, D), device=dev)
    w = torch.zeros(train_rows, device=dev)
    opt = build_optimizer(model, kind="muon", lr=0.01)
    if compile_train:
        model.autocast_enabled = False
        cl = torch.compile(masked_denoise_loss, dynamic=False)

        def step():
            with dev_mod.autocast(dev, enabled=True):
                loss = cl(model, w, x, num_replicates)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        step()
    else:
        def step():
            loss = masked_denoise_loss(model, w, x, num_replicates)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        step()
    dt = timeit(step, reps=4, warm=1)
    out["train_step_s"] = round(dt, 4)
    print(f"    train    {train_rows}x{num_replicates} rows  {dt*1000:8.1f} ms")
    return out


def main() -> int:
    dev = dev_mod.resolve("auto")
    print(f"device: {dev_mod.describe(dev)}  L={L} q={Q} beta={BETA}")

    res = {}

    # ---- upstream MDNS: RopeViT 128 x 4, batch 256, resample every 10 steps
    rope = get_rope_vit_model(L, embed_dim=128, depth=4, num_heads=4, vocab_size=Q + 1,
                              dtype="bfloat16", device=str(dev))
    res["upstream"] = measure("upstream RopeViT 128x4 (eager, as shipped)", rope,
                              scan_batch=256, train_rows=256, num_replicates=8)
    del rope
    torch.cuda.empty_cache()

    # ---- ours: nbt-2, batch 1024, resample every 25 steps, compiled
    ours = build("nbt-2", L=L, vocab_size=Q + 1, device=dev, n_blocks=6, channels=64,
                 compression_ratio=2, n_heads=4)
    res["ours"] = measure("ours nbt-2x6 (scan+train compiled)", ours,
                          scan_batch=1024, train_rows=128, num_replicates=8,
                          compile_scan=True, compile_train=True)

    # ---- per optimizer step, at each project's own cadence
    up, me = res["upstream"], res["ours"]
    up_per_step = up["scan_s"] / 10.0 + up["train_step_s"]        # 1 scan per 10 steps
    me_per_step = 24 * me["scan_s"] / 600.0 + me["train_step_s"]  # 24 scans per 600 steps
    up_samples_per_step = 256 / 10.0
    me_samples_per_step = 1024 * 24 / 600.0

    res["summary"] = {
        "upstream_s_per_step": round(up_per_step, 4),
        "ours_s_per_step": round(me_per_step, 4),
        "step_speedup": round(up_per_step / me_per_step, 2),
        "upstream_samples_per_step": round(up_samples_per_step, 2),
        "ours_samples_per_step": round(me_samples_per_step, 2),
        "upstream_samples_per_s": round(up_samples_per_step / up_per_step, 1),
        "ours_samples_per_s": round(me_samples_per_step / me_per_step, 1),
        "samples_per_s_speedup": round((me_samples_per_step / me_per_step) /
                                       (up_samples_per_step / up_per_step), 2),
    }
    print("\n=== per optimizer step ===")
    print(f"  upstream : scan {up['scan_s']:.2f}s/10 + train {up['train_step_s']*1000:.0f}ms"
          f" = {up_per_step*1000:.0f} ms/step   ({up_samples_per_step:.1f} samples/step,"
          f" {up_samples_per_step/up_per_step:.1f} samples/s)")
    print(f"  ours     : scan {me['scan_s']:.2f}s*24/600 + train {me['train_step_s']*1000:.0f}ms"
          f" = {me_per_step*1000:.0f} ms/step   ({me_samples_per_step:.1f} samples/step,"
          f" {me_samples_per_step/me_per_step:.1f} samples/s)")
    print(f"  --> {up_per_step/me_per_step:.2f}x per optimizer step, "
          f"{(me_samples_per_step/me_per_step)/(up_samples_per_step/up_per_step):.2f}x "
          f"in samples/second")

    Path(ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "bench_upstream_step.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("\nwrote results/bench_upstream_step.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
