"""The closed loop: scan -> write data -> train -> gate -> promote weights -> rescan.

One round
---------
1. **scan**   the current (EMA) weights act as a masked-diffusion sampler and
              produce ``(x, log_rnd)``; the scan ESS is logged
2. **write**  the shard is persisted to ``round_XX.npz`` next to a manifest, so
              the data a given weight produced is reproducible and auditable
3. **train**  ``steps_per_round`` steps of the weighted denoising objective on
              that shard
4. **gate**   the candidate (EMA) weights are scored on a held-out reference set
5. **promote** only if the gate says the candidate is better; otherwise the
              incumbent checkpoint is restored, so a bad round cannot poison the
              next scan

Why training is strictly on-policy
----------------------------------
The importance weights ``log_rnd`` are only valid for the policy that produced
the samples.  Training on an older shard (experience replay) would silently bias
the objective, so by default each round trains only on its own fresh shard; the
older shards stay on disk for analysis.  Set ``replay=True`` if you want to
measure that bias, but do not treat the resulting model as unbiased.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import device as dev_mod
from .gatekeeper import Gatekeeper
from .losses import EMA, ess, masked_denoise_loss, scan
from .models.registry import build, build_from_config
from .optim import build_optimizer


@dataclass
class LoopConfig:
    arch: str = "res-s"
    arch_overrides: dict = field(default_factory=dict)   # channels/n_blocks/... on top of the preset
    max_hours: float = 0.0        # 0 = unlimited; otherwise stop cleanly after this long
    resume: str | None = None     # path to best.pt to continue from
    # ---- knobs for unbounded runs ----
    anneal_rounds: int = 0        # rounds to reach `beta`; 0 = rounds-1 (finite) or 60 (infinite)
    keep_shards: int = 0          # keep only the newest N npz shards (0 = keep all)
    stop_after_rejects: int = 0   # stop after N consecutive gate rejections (0 = never)
    lr_reduce_on_reject: float = 0.0   # multiply LR on rejection (0 = off)
    lr_min: float = 1e-5
    gate_min_improvement_floor: float = 1e-6  # min_improvement decays with LR, floored here
    reset_optimizer: bool = False   # on --resume, start with a fresh optimizer state
                                    # (otherwise the decayed LR is restored and --lr is ignored)
    log_jsonl: bool = True        # append one JSON line per round
    # torch.compile the *scan-only* inference copy (bf16 weights, autocast off).
    # Measured on a 4080S at L=16: forward 29.5 ms -> 10.2 ms (2.9x), which cuts
    # a round from ~262 s to ~155 s.  Training keeps the fp32 master weights.
    compile_scan: bool = False
    # torch.compile the training forward+loss too.  Requires moving the autocast
    # context out of LatticeNet.logits (dynamo cannot trace it), which is done
    # here by flipping model.autocast_enabled and wrapping the loss call instead.
    compile_train: bool = False
    gate_ref_refresh: int = 0     # re-draw the gate reference every N rounds (0 = never)
    L: int = 8
    q: int = 3
    beta: float = 1.2
    anneal: bool = True          # ramp beta from beta_start to beta across rounds
    beta_start: float = 0.5
    J: float = 1.0
    rounds: int = 6
    steps_per_round: int = 100
    resample_every: int = 10     # re-scan every N optimizer steps (paper default: 10)
    max_samples_per_round: int = 0   # hard cap on samples produced per round (0 = no cap)
    scan_batch: int = 128
    train_batch: int = 128
    num_replicates: int = 8
    optimizer: str = "muon"
    lr: float = 1e-3
    ema_decay: float = 0.999
    gate_mode: str = "holdout_nll"
    gate_min_improvement: float = 1e-4
    gate_patience: int = 0
    gate_ref_size: int = 512
    replay: bool = False
    out_dir: str = "runs/loop"
    seed: int = 0


def _reward_fn(cfg: LoopConfig, beta: float | None = None):
    """r(x) = -beta * H(x).  ``beta`` overrides cfg.beta (used by annealing)."""
    b = cfg.beta if beta is None else beta
    if cfg.q <= 1:
        from .physics import ising2d_ham

        return lambda x: -b * ising2d_ham(2 * x - 1, cfg.J, 0.0)
    from .physics import potts2d_ham

    return lambda x: -b * potts2d_ham(x, cfg.J, cfg.q)


def make_reference(cfg: LoopConfig, device, size: int, seed: int = 99) -> torch.Tensor:
    """Held-out ground truth from the exact-validated C++ sampler (gate only)."""
    from .samplers import cpp_ext

    cpp_ext.set_threads(4)
    B = min(1024, size)
    n_collect = max(1, math.ceil(size / B))
    samples, _ = cpp_ext.collect("potts_sw", L=cfg.L, B=B, beta=cfg.beta, J=cfg.J, q=cfg.q,
                                 burn_in=400, n_collect=n_collect, collect_every=10, seed=seed)
    ref = torch.from_numpy(samples[:size].astype(np.int64)).to(device)
    return ref


def _scan_with_ema(model, ema: EMA, reward, cfg: LoopConfig, device):
    ema.store(list(model.parameters()))
    ema.copy_to(list(model.parameters()))
    x, log_rnd = scan(model, reward, cfg.scan_batch, device)
    ema.restore(list(model.parameters()))
    return x, log_rnd


def _score_with_ema(model, ema: EMA, gate: Gatekeeper, reference, cfg, device,
                    beta: float | None = None) -> float:
    ema.store(list(model.parameters()))
    ema.copy_to(list(model.parameters()))
    s = gate.score(model, reference=reference, reward_fn=_reward_fn(cfg, beta),
                   scan_fn=scan, batch_size=cfg.scan_batch)
    ema.restore(list(model.parameters()))
    return s


def run_loop(cfg: LoopConfig, device=None, verbose: bool = True) -> dict:
    dev = dev_mod.resolve(device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = build(cfg.arch, L=cfg.L, vocab_size=cfg.q + 1, device=dev,
                  **dict(cfg.arch_overrides or {}))
    params = list(model.parameters())
    ema = EMA(params, decay=cfg.ema_decay)
    opt = build_optimizer(model, kind=cfg.optimizer, lr=cfg.lr)

    # Scan-only compiled copy: the master model stays fp32 for training and for
    # the gate (whose float32 score resolution we rely on), while the O(D)
    # sequential forward pass -- 68% of a round -- runs compiled in bf16.
    scan_model = None
    if cfg.compile_scan:
        import copy

        scan_model = copy.deepcopy(model).to(torch.bfloat16)
        scan_model.autocast_enabled = False      # autocast breaks dynamo tracing
        scan_model.eval()
        scan_model = torch.compile(scan_model)
        if verbose:
            print("compile_scan: built a compiled bf16 inference copy "
                  "(first scan will trigger compilation)")
        with torch.no_grad():                    # compile at the real shape, off the clock
            _warm_reward = _reward_fn(cfg, cfg.beta_start)
            scan(scan_model, _warm_reward, cfg.scan_batch, dev)
    reference = make_reference(cfg, dev, cfg.gate_ref_size)
    gate = Gatekeeper(mode=cfg.gate_mode, min_improvement=cfg.gate_min_improvement,
                      patience=cfg.gate_patience)
    best_path = out / "best.pt"
    arch_cfg = model.config()

    start_round = 0
    total_samples = 0          # samples scanned since the very first round
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        if "ema" in ck:
            ema.load_state_dict(ck["ema"])
        if "opt" in ck and not cfg.reset_optimizer:
            try:
                opt.load_state_dict(ck["opt"])
            except Exception as exc:      # arch changed since the checkpoint
                print(f"warning: could not restore optimizer state ({exc}); continuing with fresh state")
        stored = float(ck.get("score", 0.0))
        # The stored score was measured against whatever reference/beta that run
        # used.  If the target or the reference changed (e.g. after fixing a beta
        # schedule), comparing a new score against it is meaningless, so re-score
        # the incumbent against the *current* reference.
        gate.best_score = _score_with_ema(model, ema, gate, reference, cfg, dev, beta=cfg.beta)
        total_samples = int(ck.get("total_samples", 0))
        start_round = int(ck.get("round", -1)) + 1
        print(f"resumed from {cfg.resume}: continuing at round {start_round}; "
              f"incumbent re-scored {stored:+.5f} -> {gate.best_score:+.5f} "
              f"(current reference, beta={cfg.beta})")

    if verbose:
        print(f"device={dev_mod.describe(dev)} arch={cfg.arch} params={model.num_params()/1e6:.4f}M "
              f"L={cfg.L} q={cfg.q} beta={cfg.beta} optimizer={cfg.optimizer} "
              f"gate={cfg.gate_mode} rounds={cfg.rounds}")
        print(f"reference set: {tuple(reference.shape)} (C++ Swendsen-Wang, gate only)")
        print(f"model name: {model.name}-s{total_samples}   "
              f"(arch part is stored inside the checkpoint; the -s<N> counter is the "
              f"total number of samples scanned so far)")

    history: list[dict] = []
    shards: list[tuple[torch.Tensor, torch.Tensor]] = []

    # `rounds <= 0` means run until externally stopped (Ctrl-C / pkill / max_hours)
    limit = cfg.rounds if (cfg.rounds and cfg.rounds > 0) else 10 ** 9
    if cfg.anneal:
        if cfg.anneal_rounds > 0:
            anneal_span = cfg.anneal_rounds
        elif limit < 10 ** 9:
            anneal_span = max(1, limit - 1)
        else:
            anneal_span = 60
            print("unbounded run without --anneal-rounds: annealing over the first 60 rounds")
    else:
        anneal_span = 1
    if cfg.anneal and abs(cfg.beta - cfg.beta_start) < 1e-12:
        print(f"WARNING: annealing is on but --beta == --beta-start == {cfg.beta}. "
              f"The schedule is a no-op: the sampler will stay at that temperature "
              f"for the whole run. Pass --beta <target> --beta-start <warm start> "
              f"for a real ramp.")

    # Training forward: optionally compiled.  The master model keeps fp32 weights;
    # the autocast context is hoisted out of the model (see LatticeNet.autocast_enabled)
    # and around the compiled call, which is what makes it traceable.
    if cfg.compile_train and dev.type != "cpu":
        model.autocast_enabled = False
        _compiled_loss = torch.compile(masked_denoise_loss, dynamic=False)
        if verbose:
            print("compile_train: training forward+loss compiled "
                  "(first training step will trigger compilation)")

        def train_loss(model_, weights, samples, reps):
            with dev_mod.autocast(dev, enabled=True):
                return _compiled_loss(model_, weights, samples, reps)

        # one throw-away step at the real shapes; gradients are discarded
        _xm = torch.randint(0, cfg.q + 1, (cfg.train_batch, cfg.L * cfg.L), device=dev)
        _wm = torch.zeros(cfg.train_batch, device=dev)
        train_loss(model, _wm, _xm, cfg.num_replicates).backward()
        opt.zero_grad(set_to_none=True)
        if verbose:
            print("compile_train: warmed up at the real shapes")
    else:
        train_loss = masked_denoise_loss

    def _push_ema_into_scan_model():
        with torch.no_grad():
            for p, s in zip(scan_model.parameters(), ema.shadow):
                p.copy_(s)                       # fp32 -> bf16 cast happens here

    def scan_and_count(reward_fn):
        """Scan with the EMA weights and account for the samples produced."""
        nonlocal total_samples
        if scan_model is not None:
            _push_ema_into_scan_model()
            res = scan(scan_model, reward_fn, cfg.scan_batch, dev)
        else:
            res = _scan_with_ema(model, ema, reward_fn, cfg, dev)
        total_samples += int(cfg.scan_batch)
        return res

    t_start = time.perf_counter()
    reject_streak = 0
    for r in range(start_round, limit):
        if cfg.max_hours and (time.perf_counter() - t_start) / 3600.0 >= cfg.max_hours:
            print(f"time budget {cfg.max_hours}h reached after {r} rounds; stopping cleanly")
            break
        # A fixed reference set can be memorised over a long run: the gate would
        # then accept rounds that only improve the score on those 2048 configs.
        # Re-draw it periodically and re-score the incumbent on the new set so
        # the comparison stays fair (scores from different sets are not
        # comparable).
        if (cfg.gate_ref_refresh and r > start_round
                and (r - start_round) % cfg.gate_ref_refresh == 0):
            reference = make_reference(cfg, dev, cfg.gate_ref_size, seed=99 + r)
            inc = torch.load(best_path, map_location=dev, weights_only=False)
            model.load_state_dict(inc["model"])
            ema.load_state_dict(inc["ema"])
            gate.best_score = _score_with_ema(model, ema, gate, reference, cfg, dev,
                                              beta=cfg.beta)
            print(f"  [refresh] gate reference re-drawn at round {r}; "
                  f"incumbent re-scored to {gate.best_score:+.5f}")

        t0 = time.perf_counter()
        # beta annealing: at low temperature the self-normalised importance
        # weights degenerate (one sample takes all the mass), so we start hot
        # and cool down -- the same trick as --use_anneal in the original code.
        if cfg.anneal:
            beta_r = cfg.beta_start + (cfg.beta - cfg.beta_start) * min(1.0, r / max(1, anneal_span))
        else:
            beta_r = cfg.beta
        reward = _reward_fn(cfg, beta_r)

        # 1 ---------- scan with the current (EMA) weights -------------------
        x, log_rnd = scan_and_count(reward)
        scan_ess = ess(log_rnd)
        shards.append((x, log_rnd))

        # 2 ---------- persist the shard ------------------------------------
        np.savez_compressed(out / f"round_{r:06d}.npz",
                            x=x.cpu().numpy().astype(np.int16),
                            log_rnd=log_rnd.cpu().numpy().astype(np.float32))
        if cfg.keep_shards > 0:
            for old in sorted(out.glob("round_*.npz"))[:-cfg.keep_shards]:
                old.unlink(missing_ok=True)

        # 3 ---------- train on this round's data ---------------------------
        pool_x, pool_w = x, log_rnd
        if cfg.replay and len(shards) > 1:
            pool_x = torch.cat([s[0] for s in shards], dim=0)
            pool_w = torch.cat([s[1] for s in shards], dim=0)
        n_pool = pool_x.shape[0]
        losses = []
        # How many rescans this round will do.  samples/round = scan_batch * (1 + n_rescans),
        # so this is the single place that decides the per-round sample budget.
        n_rescans = ((cfg.steps_per_round - 1) // cfg.resample_every
                     if cfg.resample_every > 0 else 0)
        if cfg.max_samples_per_round > 0:
            cap = max(0, cfg.max_samples_per_round // cfg.scan_batch - 1)
            n_rescans = min(n_rescans, cap)
        rescans_done = 0
        if verbose and r == start_round:
            print(f"samples/round = {cfg.scan_batch} x (1 + {n_rescans}) = "
                  f"{cfg.scan_batch * (1 + n_rescans)}"
                  + (f"   [capped by --max-samples-per-round "
                     f"{cfg.max_samples_per_round}]" if cfg.max_samples_per_round else ""))
        model.train()
        for step in range(cfg.steps_per_round):
            # Re-scan periodically: reusing one scan for many steps overfits the
            # handful of samples that dominate the (degenerate) importance
            # weights, which the hold-out gate then correctly rejects.
            if (step and cfg.resample_every > 0 and step % cfg.resample_every == 0
                    and rescans_done < n_rescans):
                pool_x, pool_w = scan_and_count(reward)
                n_pool = pool_x.shape[0]
                rescans_done += 1
            idx = torch.randint(0, n_pool, (cfg.train_batch,), device=dev)
            loss = train_loss(model, pool_w[idx], pool_x[idx], cfg.num_replicates)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ema.update(params)
            losses.append(loss.item())

        # name of the weights as they stand now: arch + samples scanned so far
        full_name = f"{model.name}-s{total_samples}"

        # 4 ---------- gate -------------------------------------------------
        score = _score_with_ema(model, ema, gate, reference, cfg, dev, beta=beta_r)
        prev_best = gate.best_score
        decision = gate.should_accept(score)

        # 5 ---------- promote or roll back ---------------------------------
        if decision.accepted:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                        "opt": opt.state_dict(), "arch": arch_cfg,
                        "sakura_name": full_name, "total_samples": total_samples,
                        "round": r, "score": score}, best_path)
            action = "promoted"
            reject_streak = 0
        elif not best_path.exists():
            # can happen when resuming into a fresh --out-dir and the first round
            # is rejected: keep the current weights instead of crashing
            print(f"  warning: no {best_path} to roll back to; keeping current weights")
            action = "kept (no checkpoint to roll back to)"
            reject_streak += 1
        else:
            ckpt = torch.load(best_path, map_location=dev, weights_only=False)
            model.load_state_dict(ckpt["model"])
            ema.load_state_dict(ckpt["ema"])
            action = "rolled back to round %d" % ckpt["round"]
            reject_streak += 1
            if cfg.lr_reduce_on_reject:
                for g in opt.param_groups:
                    g["lr"] = max(cfg.lr_min, g["lr"] * cfg.lr_reduce_on_reject)
                # Scale the acceptance threshold with the step size.  If only
                # the LR decays, the achievable improvement shrinks below a
                # fixed min_improvement and the run can never promote again
                # (observed: 8 rejections at 2e-5 vs a 1e-4 threshold).
                gate.min_improvement = max(cfg.gate_min_improvement_floor,
                                           gate.min_improvement * cfg.lr_reduce_on_reject)

        dev_mod.synchronize(dev)
        dt = time.perf_counter() - t0
        row = {"round": r, "model": full_name, "samples": total_samples,
               "beta": round(beta_r, 4), "scan_ess": round(scan_ess, 6),
               "train_loss_mean": round(float(np.mean(losses)), 4),
               "train_loss_last": round(float(losses[-1]), 4),
               "score_before": None if prev_best is None else round(prev_best, 6),
               "score_after": round(score, 6),
               "accepted": decision.accepted, "action": action,
               "lr": round(float(opt.param_groups[0]["lr"]), 8),
               "min_improvement": gate.min_improvement,
               "reject_streak": reject_streak, "seconds": round(dt, 2)}
        history.append(row)
        if cfg.log_jsonl:
            with open(out / "metrics.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if verbose:
            print(f"  round {r}: {full_name}  beta={beta_r:.3f} ESS={row['scan_ess']:.4f} "
                  f"loss={row['train_loss_mean']:.3f} "
                  f"score={row['score_after']:+.5f} "
                  f"(prev {row['score_before']}) -> {action}  [{dt:.1f}s]")
        if cfg.stop_after_rejects and reject_streak >= cfg.stop_after_rejects:
            print(f"stopping: {reject_streak} consecutive gate rejections "
                  f"(--stop-after-rejects {cfg.stop_after_rejects}); "
                  f"the sampler has stopped improving")
            break

    if verbose and history:
        print(f"final weights: {history[-1]['model']}")
    metrics = {"config": asdict(cfg), "device": dev_mod.describe(dev),
               "sakura_name": history[-1]["model"] if history else model.name,
               "arch": arch_cfg, "params": model.num_params(),
               "history": history, "gate_history": gate.history,
               "best_score": gate.best_score,
               "best_checkpoint": str(best_path)}
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if verbose:
        n_acc = sum(h["accepted"] for h in history)
        print(f"\n{n_acc}/{len(history)} rounds promoted; best score {gate.best_score}")
        print(f"artifacts: {out}")
    return metrics
