#!/bin/bash
# Script to run fixed demographic variation generation

MANIFEST="data/processed/manifest.csv"
CHECKPOINT="use_case_3_perfect_reconstruction/perfect_reconstruction_c3dgan/c3dgan_best.pt"
OUTPUT_DIR="demographic_variations_fixed"
MAX_VIDEOS=${1:-10}  # Default to 10, or pass as argument

echo "=========================================="
echo "Running Fixed Demographic Variation Generation"
echo "=========================================="
echo "Manifest: $MANIFEST"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "Max videos: $MAX_VIDEOS"
echo "=========================================="

python use_case_2_demographic_variations/generate_demographic_variations_fixed.py \
    --manifest "$MANIFEST" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --max_videos "$MAX_VIDEOS" \
    --use_reference \
    --diversity_weight 0.15 \
    --device cuda

echo ""
echo "=========================================="
echo "Generation complete!"
echo "Check results in: $OUTPUT_DIR"
echo "=========================================="
