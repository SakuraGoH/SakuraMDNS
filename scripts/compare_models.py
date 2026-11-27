"""Side-by-side comparison of several checkpoints against the exact sampler.

    python scripts/compare_models.py --n 4096 --out results/compare

Produces, in `results/compare/`:
  * `samples_compare.png`     one row per model (+ exact Swendsen-Wang), 5 samples each
  * `observables_compare.png` energy / magnetisation / correlation, all models overlaid
  * `compare.json`            the metric table, scored against the same reference

Shape conventions in mdns.physics: `potts2d_ham` takes flat (B, D) spins, while
`potts2d_magnetization` / `potts_2pt_corr_direction` take (n, L, L) lattices -- so
the flat tensor is kept alongside the reshaped one.
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
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from eval_checkpoint import corr_vs_r, load_model, mag_fraction, sample_model  # noqa: E402
from mdns import device as dev_mod  # noqa: E402
from mdns.physics import potts2d_ham  # noqa: E402
from mdns.samplers import cpp_ext  # noqa: E402

PALETTE = ["#540D6E", "#EE4266", "#FFD23F"]

DEFAULT_SPECS = [
    "res-s 167K=runs/potts_L16_b1.2/best.pt",
    "b6c64 96K=runs/potts_L16_nbt2x6_forever/best.pt",
    "b8c64 128K=results/archive_b8c64_round024.pt",
    "b8c128 485K=results/archive_b8c128_round005.pt",
]


def energy_per_site(flat: torch.Tensor, q: int, L: int) -> np.ndarray:
    """flat: (n, D) -> energy/site as numpy."""
    return (potts2d_ham(flat.float(), 1.0, q) / (L * L)).detach().cpu().numpy()


def tv_hist(a: np.ndarray, b: np.ndarray, bins: int = 40) -> float:
    lo = float(min(a.min(), b.min())); hi = float(max(a.max(), b.max()))
    pa, _ = np.histogram(a, bins=bins, range=(lo, hi))
    pb, _ = np.histogram(b, bins=bins, range=(lo, hi))
    return float(0.5 * np.abs(pa / pa.sum() - pb / pb.sum()).sum())


def show(flat_one: torch.Tensor, L: int, q: int, ax):
    img = flat_one.detach().cpu().numpy().reshape(L, L)
    ax.imshow(img, cmap=ListedColormap(PALETTE[:q]), vmin=-0.5, vmax=q - 0.5,
              interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="*", default=DEFAULT_SPECS, help="label=path entries")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--beta", type=float, default=1.2)
    ap.add_argument("--out", default="results/compare")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    dev = dev_mod.resolve(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    specs = [(s.split("=", 1)[0].strip(), s.split("=", 1)[1].strip()) for s in args.specs]

    # ---- exact reference, shared by every model -------------------------
    ck0 = torch.load(specs[0][1], map_location="cpu", weights_only=False)
    L = int(ck0["arch"]["L"]); q = int(ck0["arch"]["vocab_size"]) - 1
    B_ref, n_collect = 1024, max(1, int(np.ceil(args.n / 1024)))
    print(f"exact reference: Potts q={q} L={L} beta={args.beta}  ({args.n} samples)")
    ref_np, _ = cpp_ext.collect("potts_sw", L=L, B=B_ref, beta=args.beta, J=1.0, q=q,
                                burn_in=400, n_collect=n_collect, collect_every=10, seed=7)
    ref_flat = torch.as_tensor(ref_np[: args.n]).to(dev).long()
    ref3 = ref_flat.reshape(-1, L, L)
    e_ref = energy_per_site(ref_flat, q, L)
    m_ref = mag_fraction(ref3, q)
    corr_ref = corr_vs_r(ref3, q, rmax=L // 2)

    table, rows = [], []
    for label, path in specs:
        print(f"--- {label}  ({path})")
        model, ckpt, Lm, Nq = load_model(path, dev)
        qm = Nq - 1
        assert (Lm, qm) == (L, q), f"{label}: lattice/vocab mismatch ({Lm},{qm}) vs ({L},{q})"
        reward = lambda x, _q=qm: -args.beta * potts2d_ham(x, 1.0, _q)
        S, logr = sample_model(model, reward, args.n, args.chunk, dev)
        S = S.reshape(args.n, -1)
        S3 = S.reshape(-1, L, L)
        e = energy_per_site(S, qm, L)
        m = mag_fraction(S3, qm)
        corr = corr_vs_r(S3, qm, rmax=L // 2)
        # ESS/N = 1 / (N * sum(w_hat^2));  earlier this was inverted
        w = torch.softmax(logr.double(), 0)
        ess_val = float(1.0 / (len(w) * w.pow(2).sum()).item())
        corr_err = max(np.abs(np.array(corr["rows"]) - np.array(corr_ref["rows"])).max(),
                       np.abs(np.array(corr["cols"]) - np.array(corr_ref["cols"])).max())
        row = {
            "label": label, "path": path,
            "params": int(sum(p.numel() for p in model.parameters())),
            "round": ckpt.get("round"), "name": ckpt.get("sakura_name"),
            "ess": round(ess_val, 4),
            "energy_mean": round(float(e.mean()), 6),
            "energy_sw": round(float(e_ref.mean()), 6),
            "energy_delta": round(float(e.mean() - e_ref.mean()), 6),
            "energy_tv": round(tv_hist(e, e_ref), 4),
            "mag_tv": round(tv_hist(m, m_ref), 4),
            "corr_max_abs_err": round(float(corr_err), 5),
        }
        print(f"    ESS={row['ess']:.4f}  dE={row['energy_delta']:+.2e}  TV_E={row['energy_tv']:.4f}"
              f"  TV_m={row['mag_tv']:.4f}  corr={row['corr_max_abs_err']:.4f}")
        table.append(row)
        rows.append((label, S.detach().cpu(), e, m, corr))
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # ---- samples figure --------------------------------------------------
    k = 5
    nr = len(rows) + 1
    fig, axes = plt.subplots(nr, k, figsize=(2.15 * k, 2.15 * nr))
    for j in range(k):
        show(ref_flat[j], L, q, axes[nr - 1, j])
    axes[nr - 1, 0].set_ylabel("SW (exact)", fontsize=9)
    for i, (label, S, _, _, _) in enumerate(rows):
        for j in range(k):
            show(S[j], L, q, axes[i, j])
        axes[i, 0].set_ylabel(label, fontsize=9)
    fig.suptitle(f"Potts q={q}  L={L}  beta={args.beta}   top: models, bottom: exact sampler",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "samples_compare.png", dpi=130)
    plt.close(fig)

    # ---- observables figure ---------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.5))
    axs[0].hist(e_ref, bins=45, density=True, histtype="step", lw=2.4, color="k", label="SW (exact)")
    axs[1].hist(m_ref, bins=45, density=True, histtype="step", lw=2.4, color="k", label="SW (exact)")
    for label, S, e, m, corr in rows:
        axs[0].hist(e, bins=45, density=True, histtype="step", lw=1.3, alpha=0.85, label=label)
        axs[1].hist(m, bins=45, density=True, histtype="step", lw=1.3, alpha=0.85, label=label)
        xs = range(len(corr["rows"]))
        axs[2].plot(xs, corr["rows"], "-", lw=1.2, alpha=0.9, label=label)
    axs[0].set_xlabel("energy / site"); axs[0].set_ylabel("density")
    axs[0].set_title(f"Energy distribution (beta={args.beta})")
    axs[1].set_xlabel("dominant-state fraction"); axs[1].set_title("Magnetisation")
    xs_ref = range(len(corr_ref["rows"]))
    axs[2].plot(xs_ref, corr_ref["rows"], "k-", lw=2.4, label="SW rows")
    axs[2].plot(xs_ref, corr_ref["cols"], "k--", lw=2.0, label="SW cols")
    axs[2].set_xlabel("distance r"); axs[2].set_ylabel("normalised correlation")
    axs[2].set_title("2-point correlation")
    for a in axs:
        a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "observables_compare.png", dpi=130)
    plt.close(fig)

    noise = float(e_ref.std()) / np.sqrt(args.n)
    with open(out / "compare.json", "w", encoding="utf-8") as f:
        json.dump({"beta": args.beta, "n": args.n, "L": L, "q": q,
                   "energy_noise_1sigma": round(noise, 6), "models": table}, f, indent=2)

    print("\n=== summary (same exact reference for every model) ===")
    hdr = f"{'model':<16}{'params':>9}{'ESS':>8}{'dE':>11}{'TV_E':>8}{'TV_m':>8}{'corr':>9}"
    print(hdr); print("-" * len(hdr))
    for r in table:
        print(f"{r['label']:<16}{r['params']:>9}{r['ess']:>8.4f}{r['energy_delta']:>+11.2e}"
              f"{r['energy_tv']:>8.4f}{r['mag_tv']:>8.4f}{r['corr_max_abs_err']:>9.4f}")
    print(f"\n1 sigma on energy/site at n={args.n}: {noise:.2e}")
    print(f"wrote {out}/samples_compare.png, {out}/observables_compare.png, {out}/compare.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
