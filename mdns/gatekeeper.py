"""Gatekeeper: only promote new weights if they are measurably better.

Why not use ESS as the gate
---------------------------
ESS is the natural *self-contained* progress signal (it reaches 1 iff the
sampler matches the target), but it is not sufficient: a collapsed model that
always emits one configuration produces *constant* importance weights, so ESS
also goes to 1.  Gating on ESS alone would happily promote a collapsed model.

Theory aside, the gate must use a metric that a degenerate model cannot game.
A proper scoring rule on held-out reference configurations does exactly that:
the mean log-likelihood of the clean token at masked positions.  Collapsing onto
one configuration scores catastrophically on held-out data.

    score = -mean_CE   (higher is better)

Reference configurations come from the exact-validated C++ Swendsen-Wang kernel
and are generated **once** per loop; they are never used for training, only for
gating, so they do not leak into the model.

Modes
-----
``holdout_nll``  (default, recommended) proper scoring rule on held-out data
``ess``          self-contained ESS of a fresh scan -- documented above as gameable
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class GateDecision:
    accepted: bool
    new_score: float
    best_score: float
    delta: float
    reason: str

    def as_dict(self) -> dict:
        return {"accepted": self.accepted, "new_score": round(self.new_score, 6),
                "best_score": round(self.best_score, 6), "delta": round(self.delta, 6),
                "reason": self.reason}


class Gatekeeper:
    """Decides whether a candidate checkpoint replaces the incumbent.

    Parameters
    ----------
    mode            "holdout_nll" | "ess"
    min_improvement require this much absolute improvement to accept
    patience        accept "no improvement" this many consecutive times before
                    giving up (0 = strict)
    """

    def __init__(self, mode: str = "holdout_nll", min_improvement: float = 1e-4,
                 patience: int = 0, eval_reps: int = 3, mask_lambda: float = 0.75,
                 seed: int = 4242):
        if mode not in ("holdout_nll", "ess"):
            raise ValueError(f"unknown gate mode {mode!r}")
        self.mode = mode
        self.min_improvement = float(min_improvement)
        self.patience = int(patience)
        self.eval_reps = int(eval_reps)
        self.mask_lambda = float(mask_lambda)
        self.seed = int(seed)
        self.best_score: Optional[float] = None
        self.strikes = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------------ score
    @torch.no_grad()
    def score(self, model, *, reference: Optional[torch.Tensor] = None,
              reward_fn=None, scan_fn=None, batch_size: int = 256) -> float:
        """Higher is better.  Deterministic for a fixed seed and reference."""
        model.eval()
        try:
            if self.mode == "holdout_nll":
                if reference is None:
                    raise ValueError("holdout_nll mode needs a reference tensor")
                return -self._holdout_nll(model, reference)
            if scan_fn is None or reward_fn is None:
                raise ValueError("ess mode needs scan_fn and reward_fn")
            from .losses import ess as ess_fn

            vals = []
            for r in range(self.eval_reps):
                _, log_rnd = scan_fn(model, reward_fn, batch_size, reference.device
                                     if reference is not None else None)
                vals.append(ess_fn(log_rnd))
            return float(sum(vals) / len(vals))
        finally:
            model.train()

    def _holdout_nll(self, model, reference: torch.Tensor) -> float:
        from .losses import holdout_nll

        return holdout_nll(model, reference, mask_lambda=self.mask_lambda,
                           reps=self.eval_reps, seed=self.seed)

    # ----------------------------------------------------------------- decide
    def should_accept(self, new_score: float) -> GateDecision:
        if self.best_score is None:      # first candidate always wins
            self.best_score = new_score
            self.strikes = 0
            d = GateDecision(True, new_score, new_score, 0.0, "first candidate")
            self.history.append(d.as_dict())
            return d

        delta = new_score - self.best_score
        if delta >= self.min_improvement:
            self.best_score = new_score
            self.strikes = 0
            d = GateDecision(True, new_score, new_score - delta, delta, "improved")
        elif self.strikes < self.patience:
            self.strikes += 1
            d = GateDecision(True, new_score, self.best_score, delta, "patience")
        else:
            d = GateDecision(False, new_score, self.best_score, delta,
                             "no improvement; keeping incumbent")
        self.history.append(d.as_dict())
        return d

    def reset(self) -> None:
        self.best_score = None
        self.strikes = 0
        self.history.clear()
