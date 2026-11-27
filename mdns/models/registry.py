"""Model registry: named architecture presets with every knob exposed.

The point is that a training run should be reproducible from one string, and
that architecture sweeps (block count, stem width, bottleneck width,
compression ratio, attention width/heads) never require editing a training
script again.

Preset fields
-------------
block              "res" (pure residual) | "nbt" (nested bottleneck + optional attention)
channels           stem / main-trunk width
n_blocks           number of residual blocks
compression_ratio  for nbt: mid width = channels // compression_ratio
n_inner            for nbt: number of nested 3x3 blocks at the compressed width
use_attention      for nbt: insert a Transformer block at the bottleneck
n_heads            attention heads (at the compressed width)
mlp_ratio          transformer MLP expansion
attn_pos           "none" (translation equivariant, default for a periodic lattice)
                   | "sincos" (fixed 2-D sinusoidal positional encoding)
embed_dim          token embedding width (defaults to `channels`)
dtype              autograd autocast dtype

Usage
-----
>>> from mdns.models.registry import build, list_presets
>>> list_presets()
>>> model = build("nbt-4", L=16, vocab_size=4, device="xpu:0")
>>> model.num_params()
>>> model.config()
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .. import device as dev_mod
from .latticenet import LatticeNet

# ------------------------------------------------------------------ presets
PRESETS: Dict[str, Dict[str, Any]] = {
    # ---- pure residual baselines (what Item 9 uses: no attention at all) ----
    "res-s": dict(block="res", channels=48, n_blocks=4),
    "res-m": dict(block="res", channels=64, n_blocks=6),
    "res-l": dict(block="res", channels=96, n_blocks=8),
    # ---- nested bottleneck + Transformer, compression-ratio sweep ----------
    "nbt-1": dict(block="nbt", channels=64, n_blocks=4, compression_ratio=1,
                  n_inner=2, use_attention=True, n_heads=4),
    "nbt-2": dict(block="nbt", channels=64, n_blocks=4, compression_ratio=2,
                  n_inner=2, use_attention=True, n_heads=4),
    "nbt-4": dict(block="nbt", channels=64, n_blocks=4, compression_ratio=4,
                  n_inner=2, use_attention=True, n_heads=4),
    "nbt-8": dict(block="nbt", channels=64, n_blocks=4, compression_ratio=8,
                  n_inner=2, use_attention=True, n_heads=4),
    # ---- ablations ---------------------------------------------------------
    "nbt-4-noattn": dict(block="nbt", channels=64, n_blocks=4, compression_ratio=4,
                         n_inner=2, use_attention=False, n_heads=4),
    "nbt-4-deep": dict(block="nbt", channels=64, n_blocks=6, compression_ratio=4,
                       n_inner=3, use_attention=True, n_heads=4),
    "nbt-4-sincos": dict(block="nbt", channels=64, n_blocks=4, compression_ratio=4,
                         n_inner=2, use_attention=True, n_heads=4, attn_pos="sincos"),
    # ---- the paper's original ViT, kept as a reference baseline ------------
    "vit-64x6": dict(block="vit", channels=64, n_blocks=6, n_heads=4),
}

DEFAULT_PRESET = "res-s"   # fastest in samples/s at equal quality on XPU, see
                           # results/RESULTS_item3_6.md


class UnknownPreset(KeyError):
    pass


def list_presets() -> Dict[str, Dict[str, Any]]:
    return {k: dict(v) for k, v in PRESETS.items()}


def describe(name: str) -> str:
    cfg = get_config(name)
    bits = [f"{k}={v}" for k, v in cfg.items() if k != "arch"]
    return f"{name}: " + ", ".join(bits)


def get_config(name: str, **overrides) -> Dict[str, Any]:
    if name not in PRESETS:
        raise UnknownPreset(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    cfg = dict(PRESETS[name])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg.setdefault("arch", "latticenet")
    return cfg


# ------------------------------------------------------------------- build
def build(name: str = DEFAULT_PRESET, *, L: int, vocab_size: int, device=None,
          **overrides) -> torch.nn.Module:
    """Instantiate a registered architecture on ``device``."""
    dev = dev_mod.resolve(device)
    cfg = get_config(name, **overrides)
    cfg.pop("arch", None)

    if cfg.get("block") == "vit":
        # legacy baseline: the original RopeVIT from the paper.  Its `forward`
        # log-softmaxes an in-place slice, which breaks autograd through the
        # loss, so we rebind forward to an out-of-place equivalent.
        from .ropevit import get_rope_vit_model  # lazily imported: pulls in timm

        m = get_rope_vit_model(
            L, embed_dim=cfg.get("channels", 64), depth=cfg.get("n_blocks", 6),
            num_heads=cfg.get("n_heads", 4), vocab_size=vocab_size,
            dtype=cfg.get("dtype", "bfloat16"), device=str(dev))

        def _forward(x, _m=m):
            z = _m.logits(x)
            out = torch.empty_like(z)
            out[..., :-1] = z[..., :-1].log_softmax(dim=-1)
            out[..., -1:] = z[..., -1:]
            return out

        m.forward = _forward
        # normalise the introspection interface so callers need no special case
        m.num_params = lambda: sum(p.numel() for p in m.parameters())
        m.config = lambda: {"arch": "ropevit", "L": L, "vocab_size": vocab_size,
                            "channels": cfg.get("channels", 64),
                            "n_blocks": cfg.get("n_blocks", 6),
                            "block": "vit", "compression_ratio": 1,
                            "use_attention": True, "dtype": cfg.get("dtype", "bfloat16")}
        return m

    model = LatticeNet(L=L, vocab_size=vocab_size, **cfg)
    return model.to(dev)


def build_from_config(cfg: Dict[str, Any], *, L: int, vocab_size: int, device=None):
    """Rebuild exactly what ``model.config()`` recorded (used when loading ckpts)."""
    dev = dev_mod.resolve(device)
    cfg = dict(cfg)
    arch = cfg.pop("arch", "latticenet")
    if arch == "latticenet":
        cfg.pop("L", None)
        cfg.pop("vocab_size", None)
        model = LatticeNet(L=L, vocab_size=vocab_size, **cfg)
        return model.to(dev)
    # legacy ViT checkpoints
    from .ropevit import get_rope_vit_model

    return get_rope_vit_model(L, embed_dim=cfg.get("channels", 64),
                              depth=cfg.get("n_blocks", 6), num_heads=cfg.get("n_heads", 4),
                              vocab_size=vocab_size, dtype=cfg.get("dtype", "bfloat16"),
                              device=str(dev))


def summary(name: str) -> str:
    return describe(name)
