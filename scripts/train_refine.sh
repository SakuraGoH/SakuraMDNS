#!/usr/bin/env bash
# Refinement phase: keep training an existing run at a FIXED beta (no annealing).
#
#   OUT=runs/potts_L16_nbt2x6_forever bash scripts/train_refine.sh
#
# Why a separate launcher
# ----------------------
# The annealing phase ends with the model only ~10 rounds old at the target
# beta, and by then the gate has decayed the learning rate to ~5e-4 while its
# acceptance threshold is still 1e-4 -- larger than the ~2e-5 improvement a
# decayed LR can produce, so no round can ever be promoted again.  This phase:
#
#   * fixes beta at the target (no further annealing),
#   * RESUMES from best.pt but drops the optimizer state (--reset-optimizer), so
#     the LR below actually takes effect instead of restoring the decayed one,
#   * uses a smaller, refinement-appropriate LR,
#   * lowers --gate-min-improvement and scales it with the LR on rejections
#     (floor 1e-6), so real but small improvements can still be promoted,
#   * keeps the crash-recovery restart loop.
set -u

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/root/miniconda3/bin/python}"
OUT="${OUT:-runs/potts_L16_nbt2x6_forever}"
BETA="${BETA:-1.2}"
LR="${LR:-0.002}"
STEPS="${STEPS:-800}"
EVERY="${EVERY:-20}"
MIN_IMP="${MIN_IMP:-2e-5}"
SLEEP="${SLEEP:-30}"

ARCH_ARGS="--arch nbt-2 --n-blocks 6 --channels 64 --compression-ratio 2 --n-inner 2 --n-heads 4"
COMMON="--L 16 --q 3 --rounds 0 --anneal-rounds 1 \
        --beta ${BETA} --beta-start ${BETA} --no-anneal \
        --steps-per-round ${STEPS} --resample-every ${EVERY} --scan-batch 1024 \
        --optimizer muon --lr ${LR} \
        --gate-mode holdout_nll --gate-ref-size 2048 --gate-ref-refresh 20 \
        --gate-min-improvement ${MIN_IMP} --gate-min-improvement-floor 1e-6 \
        --lr-reduce-on-reject 0.85 \
        --keep-shards 40 \
        --compile --compile-train \
        --out-dir ${OUT}"

cd "$PROJ" || exit 1
mkdir -p logs runs
LOG="${LOG:-logs/refine_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG") 2>&1
echo "refinement phase: beta=${BETA} lr=${LR} steps/round=${STEPS} every=${EVERY} min_imp=${MIN_IMP}"
echo "resuming from ${OUT}/best.pt, dropping the optimizer state"
echo "logging to $LOG"

if [ ! -f mdns/samplers/build/libmdns_samplers.so ]; then
  echo "[setup] building native samplers"
  g++ -O3 -march=native -fopenmp -fPIC -shared -std=c++17 \
      mdns/samplers/cpp/samplers.cpp -o mdns/samplers/build/libmdns_samplers.so || exit 1
fi

while true; do
  if [ -f "${OUT}/best.pt" ]; then
    RESUME="--resume ${OUT}/best.pt --reset-optimizer"
  else
    RESUME=""
    echo "WARNING: no ${OUT}/best.pt -- starting this phase from scratch"
  fi
  echo "=============================================================="
  echo "[$(date '+%F %T')] refine attempt   ${RESUME:-fresh start}"
  echo "=============================================================="
  $PY -u scripts/run_closed_loop.py $ARCH_ARGS $COMMON $RESUME
  echo "[$(date '+%F %T')] exited with $?; relaunching in ${SLEEP}s"
  sleep "${SLEEP}"
done
