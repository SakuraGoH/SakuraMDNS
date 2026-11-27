"""Device-agnostic MDNS training primitives.

Moved out of the legacy ``utils_train.py`` so the closed loop does not depend on
the old tree (which hardcodes CUDA in places) and so the sampler / loss paths
share one implementation.

The three pieces:

``scan``                run the model as an any-order unmasking sampler and
                        accumulate the trajectory log-weight
                        log pi(x) - log q_theta(x) (up to a constant).
``ess``                 effective sample size of those log-weights, in [0, 1]
                        after normalisation.  ESS -> 1 when q_theta -> pi, which
                        is why it is logged every round.
``masked_denoise_loss`` the weighted denoising cross-entropy (WDCE) objective:
                        masked positions only, weight 1/lambda * softmax(log_rnd).
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ sampling
def sample_categorical_logits(logits: torch.Tensor, dtype=torch.float64) -> torch.Tensor:
    """Gumbel-max sample; logits need not be normalised."""
    gumbel = -(1e-10 - (torch.rand_like(logits, dtype=dtype) + 1e-10).log()).log()
    return (logits.to(dtype) + gumbel).argmax(dim=-1)


@torch.no_grad()
def scan(model, reward_fn: Callable[[torch.Tensor], torch.Tensor], batch_size: int,
         device) -> tuple[torch.Tensor, torch.Tensor]:
    """Any-order autoregressive (masked diffusion) sampling.

    Returns ``(x [B, D] int64, log_rnd [B])`` where ``log_rnd`` accumulates
    ``log(1/(N-1)) - log p_theta`` along the trajectory and finally adds r(x).
    It is used both as an importance weight for training and as the ESS
    diagnostic.
    """
    if hasattr(model, "module"):
        model = model.module
    N = model.vocab_size
    D = model.length
    mask = N - 1
    x = torch.full((batch_size, D), mask, dtype=torch.int64, device=device)
    rows = torch.arange(batch_size, device=device)
    order = torch.rand(x.shape, device=device).argsort(dim=-1)
    log_rnd = torch.zeros(batch_size, device=device)
    for d in range(D - 1, -1, -1):
        logp = model(x)[:, :, :-1]                       # log-probs, [B, D, N-1]
        pos = order[:, d]
        update = sample_categorical_logits(logp[rows, pos])
        x[rows, pos] = update
        log_rnd += -math.log(N - 1) - logp[rows, pos, update]
    log_rnd += reward_fn(x)
    return x, log_rnd


def ess(log_rnd: torch.Tensor, normalize: bool = True) -> float:
    """Effective sample size of the (self-normalised) importance weights."""
    w = log_rnd.detach().double().softmax(dim=-1)
    value = 1.0 / (w ** 2).sum().item()
    return value / log_rnd.shape[0] if normalize else value


# --------------------------------------------------------------------- loss
def masked_denoise_loss(model, log_rnd: torch.Tensor, x: torch.Tensor,
                        num_replicates: int = 8, max_weight: float = 1e5) -> torch.Tensor:
    """Weighted denoising cross entropy (the MDNS objective).

    ``x`` are samples produced by ``scan``; ``log_rnd`` their importance weights.
    Each sample is replicated ``num_replicates`` times with an independent mask
    ratio lambda ~ U(0,1) and weighted by softmax(log_rnd) / lambda.
    """
    if hasattr(model, "module"):
        model = model.module
    mask_token = model.vocab_size - 1
    if log_rnd.shape[0] != x.shape[0]:
        raise ValueError(f"log_rnd has {log_rnd.shape[0]} entries but x has {x.shape[0]} rows; "
                         "weights must correspond to the scanned samples one-to-one")
    B, D = x.shape
    batch = x.repeat_interleave(num_replicates, dim=0)                 # [B*R, D]
    # NOTE: keep these in float.  Casting to ``x.dtype`` looks natural but x is
    # an int64 token tensor, so the weights silently truncate to 0 and the loss
    # (and every gradient) becomes exactly zero -- the loop then "trains" but
    # never improves, which is easy to miss.
    weights = log_rnd.detach().float().softmax(dim=-1).repeat_interleave(num_replicates, dim=0)
    lam = torch.rand(batch.shape[0], device=batch.device)
    lam_w = (1.0 / lam).clamp(max=max_weight)                          # 1/lambda
    masked = torch.rand(*batch.shape, device=batch.device) < lam[..., None]
    perturbed = torch.where(masked, torch.full_like(batch, mask_token), batch)
    logp = model(perturbed)
    ce = -torch.gather(logp, -1, batch[..., None]).squeeze(-1)         # [B*R, D]
    per_sample = (ce.float() * masked).sum(dim=-1)      # per_sample >= 0
    # Signs: minimise the weighted CE.  `ce` is already -log p, so there is NO
    # extra minus here.  (An extra minus silently *maximises* the cross-entropy;
    # the gate caught that, because the hold-out NLL got worse every round.)
    return (per_sample * lam_w * weights).mean()


# ---------------------------------------------------------------------- ema
class EMA:
    """Minimal exponential moving average of model parameters."""

    def __init__(self, parameters, decay: float = 0.9999):
        self.decay = decay
        self.shadow = [p.detach().clone() for p in parameters if p.requires_grad]
        self._stored = None

    def update(self, parameters) -> None:
        with torch.no_grad():
            params = [p for p in parameters if p.requires_grad]
            for s, p in zip(self.shadow, params):
                s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def store(self, parameters) -> None:
        self._stored = [p.detach().clone() for p in parameters]

    def copy_to(self, parameters) -> None:
        # filter first: self.shadow only holds requires_grad tensors, so zipping
        # against an unfiltered parameter list would silently misalign them
        params = [p for p in parameters if p.requires_grad]
        with torch.no_grad():
            for s, p in zip(self.shadow, params):
                p.copy_(s)

    def restore(self, parameters) -> None:
        if self._stored is None:
            raise RuntimeError("restore() without a prior store()")
        with torch.no_grad():
            for s, p in zip(self._stored, parameters):
                p.copy_(s)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd: dict) -> None:
        self.decay = sd["decay"]
        self.shadow = sd["shadow"]


@torch.no_grad()
def holdout_nll(model, reference: torch.Tensor, mask_lambda: float = 0.75,
                reps: int = 3, seed: int = 4242) -> float:
    """Mean CE of the clean token at masked positions, on held-out configs.

    Kept here so the gate, the loop and the benchmarks all score identically --
    an earlier private copy inside Gatekeeper silently returned ln(vocab_size)
    for every model, which made the gate reject everything.
    """
    mask_token = model.vocab_size - 1
    gen = torch.Generator(device=reference.device)
    gen.manual_seed(int(seed))
    vals = []
    for _ in range(int(reps)):
        masked = torch.rand(reference.shape, device=reference.device, generator=gen) < mask_lambda
        perturbed = torch.where(masked, torch.full_like(reference, mask_token), reference)
        # Score in float32 from the raw logits: the model runs under bf16
        # autocast, and bf16 log-probs quantise the NLL onto a 0.0078 grid,
        # which is too coarse for the gate to see a 0.1% improvement.
        logits = model.logits(perturbed).float()
        logp = logits[..., :-1].log_softmax(dim=-1)
        ce = -torch.gather(logp, -1, reference[..., None]).squeeze(-1)
        vals.append(float((ce * masked).sum() / masked.sum().clamp_min(1)))
    return float(sum(vals) / len(vals))
