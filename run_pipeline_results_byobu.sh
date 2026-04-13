#!/usr/bin/env bash
# Run pipeline checks + metrics; intended for: byobu new-session -s echonet_results -n "bash run_pipeline_results_byobu.sh"
set -euo pipefail
REPO="/data/home/sai/Documents/EchoNet-Pediatric-BIGAN-AUGMENTATION"
cd "$REPO"
LOG="${REPO}/pipeline_results_latest.log"
exec > >(tee -a "$LOG") 2>&1

echo "=============================================="
echo "Pipeline results run — $(date -Iseconds)"
echo "Host: $(hostname) | CUDA: ${CUDA_VISIBLE_DEVICES:-default}"
echo "Log: $LOG"
echo "=============================================="

echo ""
echo "=== 1) Architecture smoke (DCGAN + U-Net) ==="
python - <<'PY'
import sys
import torch
sys.path.insert(0, "use_case_1_balance_dataset/final_pipeline")
from c3dgan_arch import Generator, Discriminator
from use_case_3_perfect_reconstruction.models import PerfectReconstructionGenerator
B, z, c = 2, torch.randn(2, 128), torch.randn(2, 11)
for s in (64, 128):
    G = Generator(128, 11, s)
    D = Discriminator(11, s)
    f = G(z, c)
    assert f.shape == (2, 1, 32, s, s)
    assert D(f, c).shape == (2, 1)
for cond in ("concat", "film"):
    m = PerfectReconstructionGenerator(64, spatial_size=64, conditioning=cond)
    x = torch.randn(2, 1, 16, 64, 64)
    d = torch.zeros(2, 11)
    d[:, 0] = 1
    d[:, 2] = 1
    d[:, 7] = 1
    y = m(x, d)
    assert y.shape == x.shape
print("OK: DCGAN sizes 64/128; U-Net concat+film")
PY

echo ""
echo "=== 2) FID sanity (same reals vs same reals → ~0) + paired real vs synthetic ==="
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/processed_full/manifest_full.csv", nrows=12)
df[["processed_path"]].to_csv("/tmp/uc1_real_mini.csv", index=False)
df[["processed_path"]].to_csv("/tmp/uc1_syn_mini.csv", index=False)
PY
python use_case_1_balance_dataset/evaluate_quality_metrics.py \
  --real_manifest /tmp/uc1_real_mini.csv \
  --synthetic_manifest /tmp/uc1_syn_mini.csv \
  --video_dir . \
  --real_video_col processed_path \
  --synthetic_video_col processed_path \
  --num_samples 12 \
  --batch_size 1 \
  --frames_per_chunk 8 \
  --output_file "${REPO}/fid_sanity_latest.json" \
  --device cpu

python use_case_1_balance_dataset/evaluate_quality_metrics.py \
  --paired_manifest use_case_1_balance_dataset/c3dgan/mixed_dataset/comparison/ratio_30pct/manifest.csv \
  --video_dir use_case_1_balance_dataset \
  --num_samples 50 \
  --batch_size 1 \
  --frames_per_chunk 8 \
  --device cuda \
  --output_file "${REPO}/uc1_fid_paired_latest.json" || true

echo ""
echo "=== 3) UC1 quality JSON (merged) ==="
if [[ -f uc1_quality_metrics.json ]]; then cat uc1_quality_metrics.json; else echo "(missing)"; fi

echo ""
echo "=== 4) ALL_EVALUATION_RESULTS_VERIFIED.json (if present) ==="
if [[ -f ALL_EVALUATION_RESULTS_VERIFIED.json ]]; then head -c 4000 ALL_EVALUATION_RESULTS_VERIFIED.json; echo ""; else echo "(missing)"; fi

echo ""
echo "=== Done $(date -Iseconds) ==="
echo "Attach to this session:  byobu attach -t echonet_results"
echo "Or read log:             cat $LOG"
