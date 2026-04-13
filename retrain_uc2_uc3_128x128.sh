#!/bin/bash
# Retrain UC2 & UC3 at 128×128 resolution
# Generated: 2026-04-09

set -e  # Exit on error

echo "=========================================="
echo "UC2/UC3 Retraining at 128×128 Resolution"
echo "=========================================="
echo ""

# Step 1: Train UC3 (Perfect Reconstruction)
echo "STEP 1: Training UC3 Perfect Reconstruction Model..."
cd use_case_3_perfect_reconstruction

python train_reconstruction.py \
  --manifest ../data/processed_full/train_manifest_filtered_clean.csv \
  --checkpoint_dir ./ckpt_uc3_128x128_T32 \
  --conditioning film \
  --epochs 50 \
  --video_length 32 \
  --video_size 128 \
  --batch_size 4 \
  --base_channels 64 \
  --lr 1e-4 \
  --lambda_temp 0.1 \
  --device cuda

cd ..
echo "✓ UC3 training complete!"
echo ""

# Step 2: Generate UC2 Demographic Variations
echo "STEP 2: Generating UC2 Demographic Variations..."
cd use_case_2_demographic_variations

python generate_demographic_variations_fixed.py \
  --manifest ../data/processed_full/train_manifest_filtered_clean.csv \
  --checkpoint ../use_case_3_perfect_reconstruction/ckpt_uc3_128x128_T32/recon_best.pt \
  --output_dir ../demographic_variations_128x128_T32 \
  --video_length 32 \
  --video_size 128 \
  --diversity_weight 0.3 \
  --use_reference \
  --max_videos 100

cd ..
echo "✓ UC2 generation complete!"
echo ""

echo "=========================================="
echo "Retraining Complete!"
echo "=========================================="
echo "Results:"
echo "  - UC3 checkpoints: use_case_3_perfect_reconstruction/ckpt_uc3_128x128_T32/"
echo "  - UC2 variations: demographic_variations_128x128_T32/"
echo ""
echo "Next steps:"
echo "  1. Run cardiac cycle coverage analysis: python analyze_cardiac_cycle_coverage.py"
echo "  2. Evaluate quality metrics (SSIM, PSNR, FID)"
echo "  3. Update paper with new resolution and metrics"
