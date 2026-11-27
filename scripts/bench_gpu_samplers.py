"""Benchmark GPU vs CPU Potts samplers in *effective samples per second*.

Raw sweeps/s is misleading here: a cluster sweep and a checkerboard sweep do
different amounts of useful work, and two samplers with equal sweep rates can
differ by 100x in how many independent samples they yield.  So we measure, per
(algorithm, L):

    tau_E, tau_M   integrated autocorrelation time of energy and of the order
                   parameter (magnetization), in sweeps, Sokal window
    tau_eff        max(tau_E, tau_M) -- the slowest mode governs the usable rate
    ESS/s          B * sweeps/s / (2 * tau_eff)

and the practical number: seconds needed to produce 100k samples.

Usage:  python scripts/bench_gpu_samplers.py [--quick]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mdns import device as dev_mod  # noqa: E402
from mdns.samplers import cpp_ext, gpu  # noqa: E402

CPU_THREADS = 4
BETA = 1.2          # ordered phase: the case the user actually cares about
Q = 3
TARGET_SECONDS = 3.0


def tau_int(x: np.ndarray, c: float = 6.0):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    if n < 16 or not np.isfinite(x).all():
        return np.nan, True
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(f * np.conj(f), n=nfft)[:n].real
    if acf[0] <= 0:
        return np.nan, True
    acf /= acf[0]
    tau = 0.5
    for t in range(1, n // 2):
        tau += acf[t]
        if t >= c * tau:
            return max(tau, 0.5), False
    return max(tau, 0.5), True


# ------------------------------------------------------------- time series
def series_cpu(kind: str, L: int, B: int, T: int, beta: float, q: int, burn: int = 400):
    cpp_ext.set_threads(CPU_THREADS)
    S = cpp_ext.init_state(kind, B, L, q, seed=3)
    if burn:
        cpp_ext.run(kind, S, beta=beta, q=q, sweeps=burn, record_energy=False, seed=3)
    Es, Ms = np.empty((T, B)), np.empty((T, B))
    for t in range(T):
        e = cpp_ext.run(kind, S, beta=beta, q=q, sweeps=1, record_energy=True, seed=3 + t)
        Es[t] = e[:, 0]
        St = torch.from_numpy(S)
        counts = torch.stack([(St == c_).sum(1) for c_ in range(q)], dim=1)
        Ms[t] = (counts.max(1).values.float() / (L * L)).numpy()
    return Es, Ms


def series_gpu(kind: str, L: int, B: int, T: int, beta: float, q: int, burn: int = 400):
    dev = dev_mod.resolve("auto")
    S = gpu.init_state(kind, L, B, q, dev, seed=3)
    if burn:
        S = gpu.run(kind, S, beta=beta, q=q, sweeps=burn, record=False, seed=3)["S"]
    r = gpu.run(kind, S, beta=beta, q=q, sweeps=T, record=True, seed=3)
    return r["energy"].cpu().numpy().T, r["mag"].cpu().numpy().T


def measure_tau(kind: str, backend: str, L: int, q: int, T: int, B: int = 128):
    fn = series_cpu if backend == "cpu" else series_gpu
    Es, Ms = fn(kind, L, B, T, BETA, q)
    out = {}
    for name, series in (("E", Es), ("M", Ms)):
        stats = [tau_int(series[:, b]) for b in range(series.shape[1])]
        vals = [t for t, _ in stats if np.isfinite(t)]
        out[f"tau_{name}"] = float(np.median(vals)) if vals else float("nan")
        out[f"tau_{name}_hit_limit"] = any(flag for _, flag in stats)
    out["tau_eff"] = float(max(out["tau_E"], out["tau_M"]))
    out["T"] = T
    return out


# ------------------------------------------------------------------- rates
# The iGPU and the CPU share one power/thermal budget and the host is not
# otherwise idle, so single measurements swing by 2-4x.  Take the best of a
# few trials: the minimum is contaminated by interference, the best is the
# closest to the hardware's real capability.
TRIALS = 2


def _best(fn) -> float:
    return max(fn() for _ in range(TRIALS))


def rate_cpu(kind: str, L: int, B: int, q: int) -> float:
    cpp_ext.set_threads(CPU_THREADS)
    S = cpp_ext.init_state(kind, B, L, q, seed=9)
    cpp_ext.run(kind, S, beta=BETA, q=q, sweeps=20, record_energy=False, seed=9)

    def trial() -> float:
        t0 = time.perf_counter()
        cpp_ext.run(kind, S, beta=BETA, q=q, sweeps=3, record_energy=False, seed=9)
        dt = time.perf_counter() - t0
        n = int(np.clip(TARGET_SECONDS / max(dt / 3, 1e-6), 3, 500))
        t0 = time.perf_counter()
        cpp_ext.run(kind, S, beta=BETA, q=q, sweeps=n, record_energy=False, seed=9)
        return n / (time.perf_counter() - t0)

    return _best(trial)


def rate_gpu(kind: str, L: int, B: int, q: int) -> float:
    dev = dev_mod.resolve("auto")
    S = gpu.init_state(kind, L, B, q, dev, seed=9)
    gpu.run(kind, S, beta=BETA, q=q, sweeps=3, record=False, seed=9)
    dev_mod.synchronize(dev)

    def trial() -> float:
        t0 = time.perf_counter()
        gpu.run(kind, S, beta=BETA, q=q, sweeps=3, record=False, seed=9)
        dev_mod.synchronize(dev)
        dt = time.perf_counter() - t0
        n = int(np.clip(TARGET_SECONDS / max(dt / 3, 1e-6), 3, 300))
        t0 = time.perf_counter()
        gpu.run(kind, S, beta=BETA, q=q, sweeps=n, record=False, seed=9)
        dev_mod.synchronize(dev)
        return n / (time.perf_counter() - t0)

    return _best(trial)


ALGOS = [("potts_sw", "cpu"), ("potts_glauber", "cpu"),
         ("potts_cb", "gpu"), ("potts_wolff", "gpu")]


def main() -> int:
    quick = "--quick" in sys.argv
    dev = dev_mod.resolve("auto")
    print(f"device: {dev_mod.describe(dev)} | beta={BETA} q={Q} cpu_threads={CPU_THREADS}")

    Ls = [16] if quick else [16, 32]
    Bs = [1024] if quick else [1024, 8192]
    T = 600 if quick else 1200

    taus = {}
    for L in Ls:
        for kind, backend in ALGOS:
            m = measure_tau(kind, backend, L, Q, T, B=128)
            taus[(kind, L)] = m
            print(f"  tau {kind:14s} L={L:2d}  E={m['tau_E']:7.2f}  M={m['tau_M']:7.2f} "
                  f"(limit={m['tau_E_hit_limit'] or m['tau_M_hit_limit']})")

    rows = []
    print(f"\n{'algo':16s}{'dev':5s}{'L':4s}{'B':7s}{'sweeps/s':>11s}{'samples/s':>12s}"
          f"{'tau_eff':>9s}{'ESS/s':>11s}")
    for L in Ls:
        for B in Bs:
            for kind, backend in ALGOS:
                try:
                    rate = (rate_cpu if backend == "cpu" else rate_gpu)(kind, L, B, Q)
                except Exception as exc:  # noqa: BLE001
                    print(f"  {kind} L={L} B={B} failed: {exc}")
                    continue
                tau = taus[(kind, L)]["tau_eff"]
                samples_per_s = B * rate
                ess = samples_per_s / (2 * tau)
                rows.append({"algo": kind, "backend": backend, "L": L, "B": B,
                             "beta": BETA, "q": Q,
                             "sweeps_per_s": round(rate, 3),
                             "samples_per_s": round(samples_per_s, 1),
                             "tau_eff": round(tau, 3),
                             "ess_per_s": round(ess, 1),
                             "seconds_per_100k_samples": round(1e5 / samples_per_s, 3),
                             "seconds_per_100k_eff_samples": round(1e5 / ess, 3)})
                print(f"{kind:16s}{backend:5s}{L:<4d}{B:<7d}{rate:11.2f}{samples_per_s:12.0f}"
                      f"{tau:9.2f}{ess:11.0f}")

    Path(ROOT / "results").mkdir(exist_ok=True)
    out = ROOT / "results" / "bench_gpu_samplers.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"device": dev_mod.describe(dev), "beta": BETA, "q": Q,
                   "cpu_threads": CPU_THREADS, "tau": {f"{k[0]}_L{k[1]}": v for k, v in taus.items()},
                   "rows": rows}, f, indent=2)
    print(f"\nwrote {out}")

    # relative ranking per (L, B)
    print("\nrelative to CPU Swendsen-Wang (ESS/s ratio):")
    for L in Ls:
        for B in Bs:
            base = [r for r in rows if r["L"] == L and r["B"] == B and r["algo"] == "potts_sw"]
            if not base:
                continue
            b0 = base[0]["ess_per_s"]
            parts = [f"{r['algo']}={r['ess_per_s'] / b0:.2f}x"
                     for r in rows if r["L"] == L and r["B"] == B]
            print(f"  L={L:2d} B={B:5d}  " + "  ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
