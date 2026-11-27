"""Evaluate a trained checkpoint: generate samples, plot them, compare to exact SW.

This is the successor of the original `potts_model_eval.ipynb` for the new
architecture: it loads a closed-loop checkpoint (model + EMA + arch config),
samples with the EMA weights, and produces both figures and numbers against the
C++ Swendsen-Wang reference (which was itself validated against exact
enumeration -- see scripts/validate_samplers.py).

Usage:
    python scripts/eval_checkpoint.py --ckpt runs/potts_L16_b1.2/best.pt \
        --beta 1.2 --n 4096 --out results/eval
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mdns import device as dev_mod
from mdns.losses import EMA, ess, scan
from mdns.models.registry import build_from_config
from mdns.samplers import cpp_ext
from mdns.physics import potts2d_ham

PALETTE = ["#540D6E", "#EE4266", "#FFD23F"]


# ------------------------------------------------------------------ helpers
def load_model(ckpt_path: str, device, use_ema: bool = True):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["arch"]
    L, N = int(cfg["L"]), int(cfg["vocab_size"])
    model = build_from_config(cfg, L=L, vocab_size=N, device=device)
    model.load_state_dict(ckpt["model"])
    if use_ema and "ema" in ckpt:
        ema = EMA(list(model.parameters()), decay=ckpt["ema"]["decay"])
        ema.load_state_dict(ckpt["ema"])
        ema.store(list(model.parameters()))
        ema.copy_to(list(model.parameters()))
    model.eval()
    return model, ckpt, L, N


def sample_model(model, reward, n: int, chunk: int, device):
    xs, ls = [], []
    for _ in range(max(1, math.ceil(n / chunk))):
        x, l = scan(model, reward, chunk, device)
        xs.append(x)
        ls.append(l)
    return torch.cat(xs)[:n], torch.cat(ls)[:n]


def corr_vs_r(S: torch.Tensor, q: int, rmax: int) -> dict:
    """Normalised Potts 2-point correlation along rows (axis 0) and columns (axis 1).

    C(r) = (<delta(s_i, s_{i+r})> - 1/q) / (1 - 1/q);  C(0) = 1.
    """
    out = {}
    for axis, name in ((1, "rows"), (2, "cols")):
        vals = []
        for r in range(rmax + 1):
            eq = (S == torch.roll(S, shifts=-r, dims=axis)).float().mean().item()
            vals.append((eq - 1.0 / q) / (1.0 - 1.0 / q))
        out[name] = vals
    return out


def mag_fraction(S: torch.Tensor, q: int) -> np.ndarray:
    """Fraction of sites in the dominant state."""
    flat = S.reshape(S.shape[0], -1)
    counts = torch.stack([(flat == c).sum(1) for c in range(q)], dim=1)
    return (counts.max(1).values.float() / flat.shape[1]).cpu().numpy()


def plot_samples(mdns: torch.Tensor, ref: torch.Tensor, L: int, q: int, path: Path,
                 k: int = 5):
    fig, axes = plt.subplots(2, k, figsize=(1.9 * k, 4.0), constrained_layout=True)
    cmap = matplotlib.colors.ListedColormap(PALETTE[:q])
    for row, (S, title) in enumerate(((mdns, "MDNS (trained)"), (ref, "Swendsen-Wang (exact)"))):
        for j in range(k):
            axes[row, j].imshow(S[j].reshape(L, L).cpu().numpy(), cmap=cmap, vmin=0, vmax=q - 1,
                                origin="lower", interpolation="nearest")
            axes[row, j].set_xticks([]); axes[row, j].set_yticks([])
            if j == 0:
                axes[row, j].set_ylabel(title, fontsize=11)
    fig.suptitle(f"Potts q={q}  L={L}  samples", fontsize=13)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_observables(e_m: np.ndarray, e_r: np.ndarray, m_m: np.ndarray, m_r: np.ndarray,
                     corr_m: dict, corr_r: dict, path: Path, beta: float, ess_val: float):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    bins = np.linspace(min(e_m.min(), e_r.min()), max(e_m.max(), e_r.max()), 40)
    ax[0].hist(e_r, bins=bins, density=True, alpha=.55, label="SW (exact)", color="#1f77b4")
    ax[0].hist(e_m, bins=bins, density=True, alpha=.55, label="MDNS", color="#d62728")
    ax[0].set_xlabel("energy / site"); ax[0].set_ylabel("density")
    ax[0].set_title(f"Energy distribution (beta={beta})"); ax[0].legend(); ax[0].grid(alpha=.3)

    bins = np.linspace(0.3, 1.0, 40)
    ax[1].hist(m_r, bins=bins, density=True, alpha=.55, label="SW (exact)", color="#1f77b4")
    ax[1].hist(m_m, bins=bins, density=True, alpha=.55, label="MDNS", color="#d62728")
    ax[1].set_xlabel("dominant-state fraction"); ax[1].set_ylabel("density")
    ax[1].set_title("Magnetisation (order parameter)"); ax[1].legend(); ax[1].grid(alpha=.3)

    r = np.arange(len(corr_m["rows"]))
    ax[2].plot(r, corr_r["rows"], "o-", color="#1f77b4", label="SW rows")
    ax[2].plot(r, corr_m["rows"], "s--", color="#d62728", label="MDNS rows")
    ax[2].plot(r, corr_r["cols"], "o-", color="#2ca02c", alpha=.6, label="SW cols")
    ax[2].plot(r, corr_m["cols"], "s--", color="#ff7f0e", alpha=.8, label="MDNS cols")
    ax[2].set_xlabel("distance r"); ax[2].set_ylabel("normalised correlation")
    ax[2].set_title("2-point correlation"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    fig.text(0.5, -0.02, f"scan ESS = {ess_val:.4f}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--beta", type=float, default=1.2)
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-ema", action="store_true")
    args = ap.parse_args()

    dev = dev_mod.resolve(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, ckpt, L, N = load_model(args.ckpt, dev, use_ema=not args.no_ema)
    q = N - 1
    print(f"model={ckpt.get('sakura_name', model.name)}")
    print(f"device={dev_mod.describe(dev)}  ckpt round={ckpt.get('round')}  "
          f"arch={ckpt['arch']['block']} C={ckpt['arch']['channels']} "
          f"blocks={ckpt['arch']['n_blocks']}  params={sum(p.numel() for p in model.parameters()):,}")

    reward = lambda x: -args.beta * potts2d_ham(x, args.J, q)
    x, log_rnd = sample_model(model, reward, args.n, args.chunk, dev)
    ess_val = ess(log_rnd)
    S = x.reshape(-1, L, L)

    cpp_ext.set_threads(4)
    s_ref, _ = cpp_ext.collect("potts_sw", L=L, B=1024, beta=args.beta, J=args.J, q=q,
                               burn_in=400, n_collect=max(1, args.n // 1024),
                               collect_every=10, seed=7)
    S_ref = torch.from_numpy(s_ref[: args.n].astype(np.int64)).reshape(-1, L, L)
    print(f"samples: MDNS {tuple(x.shape)}  SW {tuple(S_ref.shape)}")

    e_m = potts2d_ham(x, args.J, q).detach().cpu().numpy() / (L * L)
    e_r = potts2d_ham(S_ref.reshape(args.n, L * L), args.J, q).detach().cpu().numpy() / (L * L)
    m_m, m_r = mag_fraction(S, q), mag_fraction(S_ref, q)

    bins = np.linspace(min(e_m.min(), e_r.min()), max(e_m.max(), e_r.max()), 40)
    h1, _ = np.histogram(e_r, bins=bins, density=True)
    h2, _ = np.histogram(e_m, bins=bins, density=True)
    tv_e = 0.5 * np.abs(h1 - h2).sum() * (bins[1] - bins[0])
    mbins = np.linspace(0, 1, 40)
    g1, _ = np.histogram(m_r, bins=mbins, density=True)
    g2, _ = np.histogram(m_m, bins=mbins, density=True)
    tv_m = 0.5 * np.abs(g1 - g2).sum() * (mbins[1] - mbins[0])

    corr_m = corr_vs_r(S, q, L // 2)
    corr_r = corr_vs_r(S_ref, q, L // 2)
    corr_max_err = max(abs(a - b) for k in corr_m for a, b in zip(corr_m[k], corr_r[k]))

    report = {
        "checkpoint": args.ckpt, "sakura_name": ckpt.get("sakura_name", model.name),
        "round": ckpt.get("round"), "arch": ckpt["arch"],
        "beta": args.beta, "n_samples": int(x.shape[0]),
        "scan_ess": round(ess_val, 4),
        "energy_per_site": {"mdns": round(float(e_m.mean()), 5),
                            "sw": round(float(e_r.mean()), 5),
                            "delta": round(float(e_m.mean() - e_r.mean()), 5),
                            "delta_pct": round(100 * float(e_m.mean() - e_r.mean()) / abs(float(e_r.mean())), 3)},
        "energy_std": {"mdns": round(float(e_m.std()), 5), "sw": round(float(e_r.std()), 5)},
        "energy_tv": round(float(tv_e), 4),
        "mag_fraction": {"mdns_mean": round(float(m_m.mean()), 4),
                         "sw_mean": round(float(m_r.mean()), 4),
                         "mdns_std": round(float(m_m.std()), 4),
                         "sw_std": round(float(m_r.std()), 4),
                         "tv": round(float(tv_m), 4)},
        "corr_max_abs_err": round(float(corr_max_err), 4),
        "corr_mdns_rows": [round(v, 4) for v in corr_m["rows"]],
        "corr_sw_rows": [round(v, 4) for v in corr_r["rows"]],
    }
    plot_samples(S, S_ref, L, q, out / "samples.png")
    plot_observables(e_m, e_r, m_m, m_r, corr_m, corr_r, out / "observables.png",
                     args.beta, ess_val)
    with open(out / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"figures: {out/'samples.png'}  {out/'observables.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
