"""LatticeNet: residual / NBT(Transformer) hybrid backbone for masked diffusion.

Why a new architecture at all
-----------------------------
The original MDNS model is a ViT over a length-D token sequence: every block is
global attention, which costs O(D^2 * C) and cannot exploit the fact that the
input *is* a 2-D lattice with periodic boundaries.  For L=16..32 the attention
matrices are small but the per-layer overhead dominates on an iGPU, and the
parameter budget is spent on position-agnostic attention rather than on local
(physically dominant) interactions.

LatticeNet keeps the same interface -- ``model(x) -> (B, D, N)`` log-probs,
``model.logits(x)`` raw, ``model.vocab_size``/``model.length`` -- so it drops
into ``utils_train.rnd`` / ``loss_wdce`` unchanged, but the backbone is

    token embedding -> circular-padded conv stem
        -> [ ResBlock | NBTBlock ] x n_blocks
        -> GroupNorm -> 1x1 conv head -> N logits/site

``ResBlock``  classic two-3x3 residual block at full width.
``NBTBlock``  nested bottleneck residual block:
                 1x1 compress C -> C/ratio
                 `n_inner` depthwise-separable 3x3 blocks at the compressed width
                 optional Transformer block at the compressed width (global attention)
                 1x1 expand back to C
              ``ratio`` is the *compression ratio*; everything expensive runs at
              C/ratio, which is where the parameter/FLOP savings come from.

All convs use circular padding because the physics uses periodic boundaries --
zero padding would invent a boundary the Hamiltonian does not have.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import device as dev_mod


# ------------------------------------------------------------------- naming
def sakura_name(*, block: str, n_blocks: int, channels: int, n_heads: int = 0,
                compression_ratio: int = 1, use_attention: bool = False) -> str:
    """Canonical model name: ``sakura-b<blocks>c<channels>[h<heads>][nbt<ratio>][tf]``.

    The sample counter is appended by the training loop as ``-s<total samples>``,
    producing e.g. ``sakura-b6c64h4nbt2tf-s1245184``.

    This string is stored *inside* the checkpoint (``arch["name"]`` and the
    top-level ``sakura_name``), so renaming ``best.pt`` never changes the
    identity of the weights.
    """
    if block == "nbt":
        name = f"sakura-b{n_blocks}c{channels}h{n_heads if use_attention else 0}nbt{compression_ratio}"
        if use_attention:
            name += "tf"
        return name
    if block == "res":
        return f"sakura-b{n_blocks}c{channels}"
    return f"sakura-{block}-b{n_blocks}c{channels}h{n_heads}"


# ------------------------------------------------------------------ helpers
def _groups(channels: int) -> int:
    g = math.gcd(channels, 8)
    return g if g > 1 else 1


def circular_conv3x3(cin: int, cout: int, groups: int = 1) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 3, padding=1, padding_mode="circular", groups=groups,
                     bias=True)


def trunc_normal_(t: torch.Tensor, std: float = 0.02) -> None:
    with torch.no_grad():
        t.normal_(0.0, std)


class NormAct(nn.Module):
    """GroupNorm + GELU.  GroupNorm (not BatchNorm) keeps bf16 stable at batch 256."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(x))


# ------------------------------------------------------------------- blocks
class ResBlock(nn.Module):
    """Two-3x3 residual block at full width (the 'pure residual' baseline)."""

    def __init__(self, channels: int, ls_init: float = 1e-4):
        super().__init__()
        self.na1 = NormAct(channels)
        self.conv1 = circular_conv3x3(channels, channels)
        self.na2 = NormAct(channels)
        self.conv2 = circular_conv3x3(channels, channels)
        self.gamma = nn.Parameter(ls_init * torch.ones(1, channels, 1, 1))

    def forward(self, x):
        h = self.conv1(self.na1(x))
        h = self.conv2(self.na2(h))
        return x + self.gamma * h


class MidBlock(nn.Module):
    """Depthwise-separable 3x3 at the compressed width (the 'nested' inner block)."""

    def __init__(self, mid: int):
        super().__init__()
        self.na = NormAct(mid)
        self.dw = circular_conv3x3(mid, mid, groups=mid)
        self.pw = nn.Conv2d(mid, mid, 1, bias=True)

    def forward(self, x):
        return x + self.pw(self.dw(self.na(x)))


def sincos_2d(dim: int, L: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """(1, L*L, dim) fixed 2-D sinusoidal encoding, for optional attention PE."""
    device = torch.device("cpu") if device is None else device
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, 2, device=device).float() / half)
    grid = torch.arange(L, device=device).float()
    ang = grid[:, None] * freqs[None, :]                     # (L, half/2)
    emb_x = torch.cat([ang.sin(), ang.cos()], dim=-1)        # (L, half)
    emb_y = emb_x
    pe = torch.cat([emb_y[:, None, :].expand(L, L, half),
                    emb_x[None, :, :].expand(L, L, half)], dim=-1)
    pe = pe.reshape(1, L * L, 2 * half)
    if pe.shape[-1] < dim:
        pe = F.pad(pe, (0, dim - pe.shape[-1]))
    return pe.to(dtype)


class AttnBlock(nn.Module):
    """Pre-LN Transformer block operating on the (B, D, C) token view."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 2.0,
                 ls_init: float = 1e-4, pos: str = "none", L: int = 16):
        super().__init__()
        n_heads = max(1, min(n_heads, dim // 4))
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.gamma1 = nn.Parameter(ls_init * torch.ones(dim))
        self.gamma2 = nn.Parameter(ls_init * torch.ones(dim))
        self.pos = pos
        if pos == "sincos":
            self.register_buffer("pe", sincos_2d(dim, L), persistent=False)
        else:
            self.pe = None

    def forward(self, x):  # x: (B, C, L, L)
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)                     # (B, D, C)
        if self.pe is not None:
            t = t + self.pe.to(t.dtype)
        qkv = self.qkv(self.norm1(t))
        q, k, v = qkv.view(B, -1, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).reshape(B, -1, C)
        t = t + self.gamma1 * self.proj(y)
        t = t + self.gamma2 * self.mlp(self.norm2(t))
        return t.transpose(1, 2).reshape(B, C, H, W)


class NBTBlock(nn.Module):
    """Nested bottleneck residual block with optional Transformer at the bottleneck."""

    def __init__(self, channels: int, ratio: int = 4, n_inner: int = 2,
                 use_attention: bool = True, n_heads: int = 4, mlp_ratio: float = 2.0,
                 L: int = 16, attn_pos: str = "none", ls_init: float = 1e-4):
        super().__init__()
        ratio = max(1, int(ratio))
        mid = max(4, channels // ratio)
        self.mid = mid
        self.ratio = ratio
        self.na = NormAct(channels)
        self.reduce = nn.Conv2d(channels, mid, 1, bias=True)
        self.inner = nn.ModuleList([MidBlock(mid) for _ in range(max(1, n_inner))])
        self.attn = (AttnBlock(mid, n_heads, mlp_ratio, ls_init, attn_pos, L)
                     if use_attention else None)
        self.expand = nn.Conv2d(mid, channels, 1, bias=True)
        self.gamma = nn.Parameter(ls_init * torch.ones(1, channels, 1, 1))

    def forward(self, x):
        h = self.reduce(self.na(x))
        for blk in self.inner:
            h = blk(h)
        if self.attn is not None:
            h = self.attn(h)
        h = self.expand(h)
        return x + self.gamma * h


# -------------------------------------------------------------------- model
class LatticeNet(nn.Module):
    """Masked-diffusion denoiser on an L x L periodic lattice.

    Interface-compatible with the original RopeVIT: ``logits(x)``, ``forward(x)``,
    ``vocab_size``, ``length``.
    """

    def __init__(self, L: int, vocab_size: int, channels: int = 64, n_blocks: int = 4,
                 block: str = "res", compression_ratio: int = 4, n_inner: int = 2,
                 use_attention: bool = True, n_heads: int = 4, mlp_ratio: float = 2.0,
                 attn_pos: str = "none", embed_dim: Optional[int] = None,
                 ls_init: float = 1e-4, dtype: str = "bfloat16",
                 autocast: bool = True):
        super().__init__()
        if block not in ("res", "nbt"):
            raise ValueError(f"unknown block type {block!r}; expected 'res' or 'nbt'")
        self.L = int(L)
        self.length = self.L * self.L
        self.vocab_size = int(vocab_size)
        self.block_type = block
        self.channels = int(channels)
        self.compression_ratio = int(compression_ratio) if block == "nbt" else 1
        self.use_attention = bool(use_attention) and block == "nbt"
        self.dtype_name = dtype
        # torch.compile cannot trace through the autocast context below: dynamo
        # re-traces endlessly (measured: 1.2 s per call instead of 13 ms).  A
        # compiled inference copy is built with autocast=False and bf16 weights.
        self.autocast_enabled = bool(autocast)

        emb = int(embed_dim or channels)
        self.embed_dim = emb
        # identity travels with the weights, not with the file name
        self.name = sakura_name(block=block, n_blocks=int(n_blocks),
                                channels=self.channels, n_heads=n_heads,
                                compression_ratio=self.compression_ratio,
                                use_attention=self.use_attention)
        self.embed = nn.Embedding(self.vocab_size, emb)
        self.stem = (nn.Conv2d(emb, channels, 1, bias=True) if emb != channels
                     else nn.Identity())

        blocks = []
        for _ in range(int(n_blocks)):
            if block == "res":
                blocks.append(ResBlock(channels, ls_init))
            else:
                blocks.append(NBTBlock(channels, self.compression_ratio, n_inner,
                                       use_attention, n_heads, mlp_ratio, self.L,
                                       attn_pos, ls_init))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.head = nn.Conv2d(channels, self.vocab_size, 1, bias=True)

        self.apply(self._init_weights)
        for m in self.modules():
            if isinstance(m, (ResBlock, NBTBlock)):
                nn.init.zeros_(m.gamma)
            if isinstance(m, AttnBlock):
                nn.init.zeros_(m.gamma1)
                nn.init.zeros_(m.gamma2)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ---- forward
    def _features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h = self.embed(x).view(B, self.L, self.L, self.embed_dim).permute(0, 3, 1, 2)
        h = self.stem(h)
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D) int in [0, vocab) -> raw logits (B, D, vocab)."""
        ctx = dev_mod.autocast(x.device, enabled=True) if self.autocast_enabled \
            else nullcontext()
        with ctx:
            h = self._features(x)
            h = self.head(h)
        return h.permute(0, 2, 3, 1).reshape(x.shape[0], self.length, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Same convention as RopeVIT: first vocab-1 entries log-softmaxed out-of-place."""
        z = self.logits(x)
        out = torch.empty_like(z)
        out[..., :-1] = z[..., :-1].log_softmax(dim=-1)
        out[..., -1:] = z[..., -1:]
        return out

    # ---- introspection
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config(self) -> dict:
        return {"arch": "latticenet", "name": self.name,
                "L": self.L, "vocab_size": self.vocab_size,
                "channels": self.channels, "n_blocks": len(self.blocks),
                "block": self.block_type, "compression_ratio": self.compression_ratio,
                "autocast": self.autocast_enabled,
                "use_attention": self.use_attention, "dtype": self.dtype_name}
