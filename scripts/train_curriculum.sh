#!/usr/bin/env bash
# Two-stage curriculum: inherit a previous model's DATA, then stand on its own.
#
#   bash scripts/train_curriculum.sh
#
# Stage 1 -- warmup
#   A fresh (larger) model is trained on shards produced by the previous model,
#   until its gate score catches up with that model.  The shards carry
#   self-normalised importance weights for pi, so replaying them is a valid
#   off-policy estimate: the student inherits the teacher's data, not its
#   weights.  That matters because the teacher is stuck at its objective's
#   optimum (a residual ~1.5% temperature bias) -- copying weights would copy
#   the bias, while re-deriving from weighted data gives the student a chance to
#   land somewhere better.
#
# Stage 2 -- self-generated data with a sliding window
#   The student scans for itself; the replay window keeps the newest W rounds and
#   evicts the rest, so the data tracks the current policy instead of drifting
#   stale.
set -u

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/root/miniconda3/bin/python}"
TEACHER="${TEACHER:-runs/potts_L16_nbt2x6_forever/best.pt}"
WARMUP_DIR="${WARMUP_DIR:-data/warmup_old}"
OUT="${OUT:-runs/potts_L16_b${BLOCKS}c${CH}_curriculum}"
BLOCKS="${BLOCKS:-8}"
CH="${CH:-64}"
WINDOW="${WINDOW:-4}"
WARMUP_STEPS="${WARMUP_STEPS:-12000}"
LR="${LR:-0.002}"
SLEEP="${SLEEP:-30}"

COMMON="--arch nbt-2 --n-blocks ${BLOCKS} --channels ${CH} --compression-ratio 2 --n-inner 2 --n-heads 4 \
        --L 16 --q 3 --rounds 0 \
        --beta 1.2 --beta-start 1.2 --no-anneal \
        --steps-per-round 800 --resample-every 20 --scan-batch 1024 \
        --optimizer muon --lr ${LR} \
        --gate-mode holdout_nll --gate-ref-size 2048 --gate-ref-refresh 20 \
        --gate-min-improvement 1e-5 --gate-min-improvement-floor 5e-6 \
        --lr-reduce-on-reject 0.85 \
        --window-rounds ${WINDOW} \
        --warmup-shards ${WARMUP_DIR} --warmup-min-round 0 \
        --warmup-steps ${WARMUP_STEPS} --warmup-eval-every 1000 \
        --warmup-ckpt ${TEACHER} \
        --keep-shards 40 \
        --compile --compile-train \
        --out-dir ${OUT}"

cd "$PROJ" || exit 1
mkdir -p logs runs data
LOG="${LOG:-logs/curriculum_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG") 2>&1
echo "curriculum: arch=b${BLOCKS}c${CH} teacher=${TEACHER} warmup=${WARMUP_DIR} window=${WINDOW} lr=${LR}"
echo "logging to ${LOG}"

if [ ! -f mdns/samplers/build/libmdns_samplers.so ]; then
  echo "[setup] building native samplers"
  g++ -O3 -march=native -fopenmp -fPIC -shared -std=c++17 \
      mdns/samplers/cpp/samplers.cpp -o mdns/samplers/build/libmdns_samplers.so || exit 1
fi

# Stage 1 only runs on a fresh start; afterwards the checkpoint carries
# "warmup": true and the loop skips straight to stage 2.
if [ ! -f "${WARMUP_DIR}/round_000000.npz" ] && [ -f "${TEACHER}" ]; then
  echo "=== generating warmup shards from the teacher ==="
  $PY -u scripts/gen_shards.py --ckpt "${TEACHER}" --n 24 --out "${WARMUP_DIR}"
fi

while true; do
  if [ -f "${OUT}/best.pt" ]; then
    RESUME="--resume ${OUT}/best.pt --reset-optimizer"
  else
    RESUME=""
  fi
  echo "=============================================================="
  echo "[$(date '+%F %T')] curriculum attempt   ${RESUME:-fresh start}"
  echo "=============================================================="
  $PY -u scripts/run_closed_loop.py $COMMON $RESUME
  echo "[$(date '+%F %T')] exited with $?; relaunching in ${SLEEP}s"
  sleep "${SLEEP}"
done
