"""Muon: momentum SGD + Newton-Schulz orthogonalisation of the update.

Reference: Keller Jordan et al., "Muon: An optimizer for hidden layers in neural
networks" (2024).  The idea is that the update to a weight *matrix* should be
(approximately) an orthogonal matrix rather than a raw gradient: this
normalises the update's spectral scale so one learning rate works across layers
of very different conditioning.

Practicalities handled here
--------------------------
* Muon only applies to matrices.  Conv kernels are 4-D, so they are flattened
  to ``(out_channels, in_channels * k * k)`` and reshaped back afterwards.
* Parameters where orthogonalisation is meaningless or harmful -- embeddings,
  output head, normalisation scales, LayerScale gammas, biases -- are routed to
  AdamW inside the *same* optimizer object, so callers keep a single
  ``.step()`` / ``.state_dict()`` and checkpoints stay simple.
* Newton-Schulz runs in bf16: it is a fixed-point iteration on a normalised
  matrix, so precision beyond bf16 buys nothing and bf16 is ~2x faster on XPU.

Usage
-----
>>> from mdns.optim import build_optimizer
>>> opt = build_optimizer(model, kind="muon", lr=0.02, adamw_lr=1e-3)
>>> opt.step()

or manually::

>>> from mdns.optim.muon import Muon, split_params
>>> muon_params, other_params = split_params(model)
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch


# --------------------------------------------------------------- core kernel
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the orthogonal factor of ``G`` (i.e. U V^T of its SVD).

    Quintic Newton-Schulz iteration with the coefficients from the Muon
    reference implementation.  Requires ``G`` to be 2-D.
    """
    if G.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {tuple(G.shape)}")
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16)
    X = X / (X.norm() + eps)           # spectral norm <= 1 so the iteration converges
    transposed = X.size(0) > X.size(1)  # iterate on the smaller Gram matrix
    if transposed:
        X = X.mT
    for _ in range(int(steps)):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


# ---------------------------------------------------------------- optimiser
class Muon(torch.optim.Optimizer):
    """Muon for matrix parameters, AdamW for everything else.

    Param groups select the branch with ``use_muon`` (default True).  A group
    with ``use_muon=False`` behaves exactly like AdamW with ``adamw_lr``.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0,
                 adamw_lr: float = 1e-3, adamw_betas: Sequence[float] = (0.9, 0.999),
                 adamw_eps: float = 1e-8):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        weight_decay=weight_decay, adamw_lr=adamw_lr,
                        betas=tuple(adamw_betas), eps=adamw_eps, use_muon=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            use_muon = bool(group.get("use_muon", True))
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            ns_steps = int(group["ns_steps"])
            wd = float(group["weight_decay"])
            lr = float(group["lr"])
            adamw_lr = float(group["adamw_lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if use_muon and g.ndim >= 2:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    upd = g.add(buf, alpha=momentum) if nesterov else buf

                    shape = upd.shape
                    mat = upd.reshape(shape[0], -1)
                    mat = zeropower_via_newtonschulz5(mat, steps=ns_steps)
                    # scale by sqrt(rows/cols) as in the reference implementation
                    mat = mat * max(1.0, shape[0] / mat.size(1)) ** 0.5
                    upd = mat.reshape(shape).to(p.dtype)

                    if wd:
                        p.mul_(1.0 - lr * wd)
                    p.add_(upd, alpha=-lr)
                else:
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(g)
                        state["exp_avg_sq"] = torch.zeros_like(g)
                    state["step"] += 1
                    t = state["step"]
                    m, v = state["exp_avg"], state["exp_avg_sq"]
                    m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                    bias1 = 1.0 - beta1 ** t
                    bias2 = 1.0 - beta2 ** t
                    denom = (v.sqrt() / (bias2 ** 0.5)).add_(eps)
                    if wd:
                        p.mul_(1.0 - adamw_lr * wd)
                    p.addcdiv_(m, denom, value=-adamw_lr / bias1)
        return loss

    @torch.no_grad()
    def orthogonalise_(self, p: torch.Tensor) -> None:
        """In-place orthogonalisation, exposed for tests."""
        mat = p.reshape(p.shape[0], -1)
        out = zeropower_via_newtonschulz5(mat)
        p.copy_(out.reshape_as(p).to(p.dtype))


# ---------------------------------------------------------------- splitting
_OTHER_KEYWORDS = ("embed", "head", "norm", "gamma", "bias", "stem")


def split_params(model: torch.nn.Module) -> Tuple[List[torch.nn.Parameter],
                                                 List[torch.nn.Parameter]]:
    """Split into (matrix params for Muon, everything else for AdamW)."""
    muon, other = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_matrix_like = p.ndim >= 2
        excluded = any(k in name for k in _OTHER_KEYWORDS)
        (muon if (is_matrix_like and not excluded) else other).append(p)
    return muon, other


def build_optimizer(model: torch.nn.Module, kind: str = "muon", *, lr: float | None = None,
                    weight_decay: float = 0.0, momentum: float = 0.95,
                    ns_steps: int = 5, adamw_lr: float = 1e-3,
                    betas: Sequence[float] = (0.9, 0.999)):
    """Return a torch optimizer for ``model``.

    kind="adamw" -> plain AdamW on every parameter (the baseline)
    kind="muon"  -> Muon on matrix params, AdamW on the rest
    """
    kind = kind.lower()
    if kind == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr if lr is not None else 1e-3,
                                 betas=tuple(betas), weight_decay=weight_decay)
    if kind != "muon":
        raise ValueError(f"unknown optimizer {kind!r}; expected 'muon' or 'adamw'")

    muon_params, other_params = split_params(model)
    if not muon_params:
        # degenerate: nothing matrix-like, fall back to AdamW
        return torch.optim.AdamW(other_params, lr=adamw_lr, weight_decay=weight_decay)
    groups = [
        dict(params=muon_params, use_muon=True, lr=0.02 if lr is None else lr,
             momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay),
        dict(params=other_params, use_muon=False, lr=adamw_lr, weight_decay=weight_decay,
             betas=tuple(betas)),
    ]
    return Muon(groups, lr=0.02 if lr is None else lr, adamw_lr=adamw_lr,
                weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps)


def describe(model: torch.nn.Module) -> str:
    muon, other = split_params(model)
    n_m = sum(p.numel() for p in muon)
    n_o = sum(p.numel() for p in other)
    return (f"muon {len(muon)} tensors / {n_m/1e6:.4f}M params, "
            f"adamw {len(other)} tensors / {n_o/1e6:.4f}M params "
            f"({n_m / max(n_m + n_o, 1):.1%} of weights are orthogonalised)")
