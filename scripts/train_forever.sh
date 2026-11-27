#!/usr/bin/env bash
# Unbounded ("run forever") training with crash recovery.
#
#   nohup bash scripts/train_forever.sh > logs/forever.log 2>&1 &
#
# What this does differently from train_cloud.sh:
#   * --rounds 0            never stops on its own (stop with pkill / shutdown)
#   * --anneal-rounds 60    beta reaches 1.2 after 60 rounds and then HOLDS there
#                           (the schedule is a function of the absolute round
#                           number, so it survives restarts unchanged)
#   * --keep-shards 40      only the newest 40 npz shards are kept (~3 MB) so a
#                           months-long run cannot fill the disk
#   * --gate-ref-refresh 20 the gate reference set is re-drawn every 20 rounds,
#                           otherwise the model eventually memorises the 2048
#                           fixed reference configs and the gate lies to you
#   * crash recovery        if the process dies (OOM, network, CUDA error) it is
#                           relaunched with --resume from best.pt, so no progress
#                           is lost: model + EMA + optimizer state are restored
#
# Everything is resumable, so you can also kill it, change hyper-parameters and
# restart with the same --out-dir.
set -u

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/root/miniconda3/bin/python}"
OUT="${OUT:-runs/potts_L16_nbt2x6_forever}"
ANNEAL_ROUNDS="${ANNEAL_ROUNDS:-60}"
CTRL_C_SLEEP="${CTRL_C_SLEEP:-30}"

# nbt-2 preset widened to 6 blocks: 96K params, bottleneck ratio 2 (mid=32),
# 4 attention heads, a transformer block at the bottleneck of every block.
# Measured on a 4080S at L=16: 7.4 s per scan (B=1024), 138 ms per train step.
ARCH_ARGS="--arch nbt-2 --n-blocks 6 --channels 64 --compression-ratio 2 --n-inner 2 --n-heads 4"

COMMON="--L 16 --q 3 --rounds 0 --anneal-rounds ${ANNEAL_ROUNDS} \
        --beta 1.2 --beta-start 0.2 \
        --steps-per-round 600 --resample-every 25 --scan-batch 1024 \
        --optimizer muon --lr 0.01 \
        --gate-mode holdout_nll --gate-ref-size 2048 --gate-min-improvement 1e-4 \
        --gate-ref-refresh 20 \
        --keep-shards 40 \
        --lr-reduce-on-reject 0.7 \
        --compile --compile-train \
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

CKPT="${OUT}/best.pt"
attempt=0
while true; do
  attempt=$((attempt + 1))
  RESUME=""
  if [ -f "$CKPT" ]; then RESUME="--resume ${CKPT}"; fi
  echo "=============================================================="
  echo "[$(date '+%F %T')] attempt ${attempt}  ${RESUME:-(fresh start)}"
  echo "=============================================================="
  $PY -u scripts/run_closed_loop.py $ARCH_ARGS $COMMON $RESUME
  code=$?
  echo "[$(date '+%F %T')] loop exited with code ${code}; relaunching in ${CTRL_C_SLEEP}s"
  sleep "${CTRL_C_SLEEP}"
done
