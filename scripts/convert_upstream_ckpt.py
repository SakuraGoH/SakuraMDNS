"""Convert an upstream MDNS checkpoint into this repo's checkpoint format.

    python scripts/convert_upstream_ckpt.py \
        --src checkpoints/L_16_potts/potts_low.pth \
        --out results/weights/upstream_ropevit_potts_low.pt

The upstream `*.pth` (as shipped in `checkpoints/`) is a dict with
`model_state_dict`, `ema_state_dict`, `losses`, `ess_eval` and `cfg`.  Its
`model_state_dict` loads *strictly* into `get_rope_vit_model(L, 128, depth=4,
heads=4, vocab=q+1)`, and its `ema_state_dict` uses the same
decay/num_updates/shadow_params layout as `mdns.losses.EMA`, so both restore
without any shape surgery.

The output is a normal `best.pt`, which means the upstream model can be scored
and plotted by the same tools as ours:

    python scripts/eval_checkpoint.py --ckpt results/weights/upstream_ropevit_potts_low.pt \
        --beta 1.2 --n 4096 --out results/eval_upstream
    python scripts/compare_models.py \
        --specs "upstream RopeViT=results/weights/upstream_ropevit_potts_low.pt" \
                "b8c64=results/weights/b8c64_round024.pt" --n 4096
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from mdns.losses import EMA  # noqa: E402
from mdns.models.registry import build  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--tag", default=None, help="sakura_name to record")
    args = ap.parse_args()

    d = torch.load(args.src, map_location="cpu", weights_only=False)
    cfg = d["cfg"]
    L, q = int(cfg["L"]), int(cfg["q"])
    print(f"upstream ckpt: L={L} q={q} beta={cfg.get('beta')} "
          f"epochs={cfg.get('num_epochs')} batch={cfg.get('batch_size')} "
          f"resample_every={cfg.get('resample_every_n_step')}")
    losses = d.get("losses") or []
    ess = d.get("ess_eval") or []
    if losses:
        print(f"  recorded final loss={float(losses[-1]):.4f}")
    if ess:
        print(f"  recorded final ess_eval={float(ess[-1]):.4f}")

    # build() attaches a .config(), which get_rope_vit_model alone does not
    model = build("vit-64x6", L=L, vocab_size=q + 1, device="cpu",
                  channels=args.embed_dim, n_blocks=args.depth, n_heads=args.num_heads)
    model.load_state_dict(d["model_state_dict"], strict=True)   # verified exact
    ema = EMA(list(model.parameters()), decay=0.999)
    esd = d.get("ema_state_dict") or {}
    ours = [p for p in model.parameters() if p.requires_grad]
    sp = esd.get("shadow_params")
    if sp is not None and len(sp) == len(ours) and all(a.shape == b.shape for a, b in zip(ours, sp)):
        # upstream stores the EMA as a bare list of tensors; ours as {"shadow": [...]}
        ema.shadow = [t.detach().clone() for t in sp]
        ema.decay = float(esd.get("decay", 0.999))
        ema.num_updates = int(esd.get("num_updates", 0))
        ema.copy_to(list(model.parameters()))                   # evaluate the EMA weights
        print(f"  EMA weights restored ({len(sp)} tensors, decay={ema.decay})")
    elif isinstance(esd, dict) and esd.get("shadow"):
        ema.load_state_dict(esd)
        ema.copy_to(list(model.parameters()))
        print("  EMA weights restored (native format)")
    else:
        print(f"  EMA skipped ({len(sp) if sp is not None else 0} shadow tensors vs {len(ours)} params)")
    model.eval()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"upstream-RopeViT{args.embed_dim}x{args.depth}"
    torch.save({"arch": model.config(), "model": model.state_dict(),
                "ema": ema.state_dict(), "round": -1, "score": None,
                "sakura_name": tag, "total_samples": int(cfg.get("num_epochs", 0)) * int(cfg.get("batch_size", 0)),
                "warmup": True, "upstream": True}, out)
    n = sum(p.numel() for p in model.parameters())
    print(f"wrote {out}  ({out.stat().st_size/1e6:.2f} MB)  params={n:,}  name={tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
