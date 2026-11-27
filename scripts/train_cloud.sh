#!/usr/bin/env bash
# Two-stage cloud training for Potts L=16, beta=1.2, NBT(bottleneck)+Transformer.
#
# Stage A: beta anneals 0.2 -> 1.2 (the sampler cannot bootstrap at low T, its
#          importance weights degenerate -- measured ESS ~0.01 vs 0.4 at beta=0.2)
# Stage B: resume and hold beta=1.2, which is where the previous pure-residual
#          run ended up "systematically too hot": it only trained ~10 rounds at
#          the target temperature.
#
# Both stages stop cleanly on their own time budget, so the box never bills
# past the plan.  Interrupt anytime: best.pt is written on every promotion and
# stage B resumes from it.
#
# Usage:  nohup bash scripts/train_cloud.sh > logs/train_8h.log 2>&1 &
set -u

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/root/miniconda3/bin/python}"
OUT="${OUT:-runs/potts_L16_nbt2x6}"
STAGE_A_HOURS="${STAGE_A_HOURS:-4.6}"
STAGE_B_HOURS="${STAGE_B_HOURS:-3.4}"

# architecture: nbt-2 preset with 6 blocks  => 96K params, mid width 32,
# 4 attention heads, transformer in every block (measured 7.4 s / scan @ B=1024)
ARCH_ARGS="--arch nbt-2 --n-blocks 6 --channels 64 --compression-ratio 2 --n-inner 2 --n-heads 4"

COMMON="--L 16 --q 3 --optimizer muon --lr 0.01 --compile --compile-train \
        --steps-per-round 600 --resample-every 25 --scan-batch 1024 \
        --gate-mode holdout_nll --gate-ref-size 2048 --gate-min-improvement 1e-4 \
        --out-dir ${OUT}"

cd "$PROJ" || exit 1
mkdir -p logs runs

# Mirror everything to a timestamped log while still printing to the terminal, so
# `bash scripts/train_forever.sh` can simply be run in the foreground: you watch
# the rounds scroll by and the record survives a closed browser tab.
LOG="${LOG:-logs/run_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG") 2>&1
echo "logging to $LOG"

if [ ! -f mdns/samplers/build/libmdns_samplers.so ]; then
  echo "[setup] building native samplers"
  g++ -O3 -march=native -fopenmp -fPIC -shared -std=c++17 \
      mdns/samplers/cpp/samplers.cpp -o mdns/samplers/build/libmdns_samplers.so || exit 1
fi

echo "=============================================================="
echo "Stage A: beta 0.2 -> 1.2, budget ${STAGE_A_HOURS}h"
echo "=============================================================="
$PY -u scripts/run_closed_loop.py $ARCH_ARGS $COMMON \
    --rounds 60 --beta 1.2 --beta-start 0.2 --max-hours "${STAGE_A_HOURS}"

CKPT="${OUT}/best.pt"
if [ ! -f "$CKPT" ]; then
  echo "no checkpoint produced by stage A -- aborting"; exit 1
fi

echo "=============================================================="
echo "Stage B: hold beta=1.2, resume from ${CKPT}, budget ${STAGE_B_HOURS}h"
echo "=============================================================="
$PY -u scripts/run_closed_loop.py $ARCH_ARGS $COMMON \
    --rounds 120 --beta 1.2 --beta-start 1.2 --no-anneal \
    --max-hours "${STAGE_B_HOURS}" --resume "$CKPT"

echo "=============================================================="
echo "done. evaluate with:"
echo "  $PY scripts/eval_checkpoint.py --ckpt ${CKPT} --beta 1.2 --n 4096 --out results/eval_final"
echo "=============================================================="
