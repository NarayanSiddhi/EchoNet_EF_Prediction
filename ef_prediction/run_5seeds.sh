#!/usr/bin/env bash
# Multi-seed runs of the two main models (EchoStature fused + video-only).
# EPOCHS defaults to 300 (paper config). NRUNS defaults to 5.
# Run ids start at 11 so checkpoints/fused/run_1_best.pth is not overwritten.
# Fused seed is 42 + run id. Video-only uses that same seed.
#
# 7 runs x 100 epochs:
#   EPOCHS=100 NRUNS=7 bash ef_prediction/run_5seeds.sh
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-300}"
NRUNS="${NRUNS:-5}"
TAG="${TAG:-ep${EPOCHS}_n${NRUNS}}"
LOGDIR="ef_prediction/logs/${TAG}"
mkdir -p "${LOGDIR}"

echo "=== ${NRUNS} runs, ${EPOCHS} epochs, tag ${TAG} ==="

for i in $(seq 0 $((NRUNS - 1))); do
  run=$((11 + i))
  seed=$((42 + run))
  echo "----- fused run ${run} (seed ${seed}) start $(date) -----"
  python -m ef_prediction.train_fused --run "${run}" --epochs "${EPOCHS}" \
    2>&1 | tee "${LOGDIR}/fused_run${run}.log"
  python -m ef_prediction.evaluate_ef_fused --run "${run}" --tag "${TAG}" \
    2>&1 | tee -a "${LOGDIR}/fused_run${run}.log"

  echo "----- real seed ${seed} start $(date) -----"
  python -m ef_prediction.train_real --seed "${seed}" --epochs "${EPOCHS}" \
    2>&1 | tee "${LOGDIR}/real_seed${seed}.log"
  python -m ef_prediction.evaluate_ef_real \
    --checkpoint "ef_prediction/checkpoints/real/seed_${seed}_best.pth" \
    --tag "seed${seed}_${TAG}" \
    2>&1 | tee -a "${LOGDIR}/real_seed${seed}.log"
done

python ef_prediction/summarize_5seeds.py --tag "${TAG}"
echo "=== finished $(date) ==="
