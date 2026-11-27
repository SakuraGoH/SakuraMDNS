"""Run the closed loop: scan -> write data -> train -> gate -> promote -> rescan.

Examples
--------
small smoke run (default, L=8, ~10 s):
    python scripts/run_closed_loop.py

the real thing (L=16, longer):
    python scripts/run_closed_loop.py --L 16 --rounds 10 --steps-per-round 300 ^
        --arch nbt-2 --optimizer muon --out-dir runs/potts_L16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mdns import device as dev_mod  # noqa: E402
from mdns.loop import LoopConfig, run_loop  # noqa: E402
from mdns.models.registry import list_presets  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="MDNS closed training loop")
    p.add_argument("--arch", default="res-s", choices=sorted(list_presets()))
    p.add_argument("--L", type=int, default=8)
    p.add_argument("--q", type=int, default=3)
    p.add_argument("--beta", type=float, default=1.2)
    p.add_argument("--beta-start", type=float, default=0.5,
                   help="starting beta for annealing (cheap region, keeps IS weights alive)")
    p.add_argument("--no-anneal", action="store_true")
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--rounds", type=int, default=6,
                   help="total rounds; 0 = run until stopped externally (unbounded)")
    p.add_argument("--anneal-rounds", type=int, default=0,
                   help="rounds to reach --beta (0 = rounds-1, or 60 when unbounded)")
    p.add_argument("--keep-shards", type=int, default=0,
                   help="keep only the newest N round_*.npz (0 = keep all)")
    p.add_argument("--stop-after-rejects", type=int, default=0,
                   help="stop after N consecutive gate rejections (0 = never)")
    p.add_argument("--lr-reduce-on-reject", type=float, default=0.0,
                   help="multiply LR by this factor when a round is rejected (0 = off)")
    p.add_argument("--max-hours", type=float, default=0.0,
                   help="stop cleanly after this many hours (0 = run all --rounds)")
    # architecture overrides on top of --arch
    p.add_argument("--channels", type=int, default=None, help="stem/trunk width")
    p.add_argument("--n-blocks", type=int, default=None)
    p.add_argument("--compression-ratio", type=int, default=None, help="NBT bottleneck ratio")
    p.add_argument("--n-inner", type=int, default=None, help="nested inner blocks per NBT block")
    p.add_argument("--n-heads", type=int, default=None, help="attention heads")
    p.add_argument("--attn-pos", default=None, choices=["none", "sincos"])
    p.add_argument("--no-attention", action="store_true", help="drop the transformer block")
    p.add_argument("--steps-per-round", type=int, default=100)
    p.add_argument("--scan-batch", type=int, default=128)
    p.add_argument("--train-batch", type=int, default=128)
    p.add_argument("--num-replicates", type=int, default=8)
    p.add_argument("--resample-every", type=int, default=10,
                   help="re-scan every N optimizer steps (paper default 10)")
    p.add_argument("--optimizer", default="muon", choices=["muon", "adamw"])
    p.add_argument("--lr", type=float, default=None,
                   help="default: 1e-2 for muon, 1e-3 for adamw")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--gate-mode", default="holdout_nll", choices=["holdout_nll", "ess"])
    p.add_argument("--gate-min-improvement", type=float, default=1e-4)
    p.add_argument("--gate-patience", type=int, default=0)
    p.add_argument("--gate-ref-size", type=int, default=512)
    p.add_argument("--gate-ref-refresh", type=int, default=0,
                   help="re-draw the gate reference every N rounds (0 = never; "
                        "recommended 20-50 for very long runs)")
    p.add_argument("--replay", action="store_true",
                   help="train on all previous shards (off-policy; see mdns/loop.py)")
    p.add_argument("--device", default="auto")
    p.add_argument("--out-dir", default="runs/loop_default")
    p.add_argument("--resume", default=None,
                   help="continue from a best.pt (rounds count from the resumed round)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    lr = args.lr if args.lr is not None else (1e-2 if args.optimizer == "muon" else 1e-3)
    overrides = {}
    for k in ("channels", "n_blocks", "compression_ratio", "n_inner", "n_heads", "attn_pos"):
        v = getattr(args, k, None)
        if v is not None:
            overrides[k] = v
    if args.no_attention:
        overrides["use_attention"] = False
    cfg = LoopConfig(arch=args.arch, arch_overrides=overrides, max_hours=args.max_hours, L=args.L, q=args.q, beta=args.beta, J=args.J,
                     rounds=args.rounds, steps_per_round=args.steps_per_round,
                     anneal=not args.no_anneal, beta_start=args.beta_start,
                     scan_batch=args.scan_batch, train_batch=args.train_batch,
                     num_replicates=args.num_replicates, resample_every=args.resample_every, optimizer=args.optimizer, lr=lr,
                     ema_decay=args.ema_decay, gate_mode=args.gate_mode,
                     gate_min_improvement=args.gate_min_improvement,
                     gate_patience=args.gate_patience, gate_ref_size=args.gate_ref_size,
                     replay=args.replay, out_dir=args.out_dir, seed=args.seed,
                     resume=args.resume, anneal_rounds=args.anneal_rounds,
                     keep_shards=args.keep_shards,
                     stop_after_rejects=args.stop_after_rejects,
                     lr_reduce_on_reject=args.lr_reduce_on_reject,
                     gate_ref_refresh=args.gate_ref_refresh)

    dev = dev_mod.resolve(args.device)
    print(f"device: {dev_mod.describe(dev)}")
    metrics = run_loop(cfg, device=dev)

    print(json.dumps({"rounds": len(metrics["history"]),
                      "promoted": sum(h["accepted"] for h in metrics["history"]),
                      "best_score": metrics["best_score"],
                      "checkpoint": metrics["best_checkpoint"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
