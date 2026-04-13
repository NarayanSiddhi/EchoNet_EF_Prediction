# Data Augmentation for Dataset Balancing

This module uses **PerfectReconstructionGenerator** to balance the dataset across demographic groups by generating high-quality synthetic videos that preserve cardiac motion patterns and work well with Grad-CAM analysis.

## Why PerfectReconstructionGenerator?

- ✅ **Proven with Grad-CAM**: Already validated in your codebase
- ✅ **Preserves cardiac motion**: U-Net encoder-decoder architecture maintains temporal patterns
- ✅ **Works with minimal samples**: Can generate multiple variations from even 1-2 real videos
- ✅ **High quality**: SSIM > 0.99, suitable for diagnostic tasks

## Quick Start

### Option 1: Run in Byobu Session (Recommended)

```bash
./run_data_augmentation_byobu.sh
```

This will:
1. Create a persistent byobu session
2. Analyze dataset imbalance
3. Generate synthetic videos to balance underrepresented groups
4. Create a monitoring window to track progress

**To detach**: Press `Ctrl+A` then `D`, or type `byobu detach`  
**To reattach**: `byobu attach -t data_augmentation`

### Option 2: Run Directly

**Step 1: Analyze dataset imbalance**
```bash
python Data_Augmentation.py --mode analyze \
    --manifest data/processed_full/manifest_full.csv \
    --output_dir data_augmentation_output \
    --target_samples 500
```

**Step 2: Generate synthetic videos**
```bash
python Data_Augmentation.py --mode generate \
    --checkpoint use_case_3_perfect_reconstruction/perfect_reconstruction_c3dgan/c3dgan_best.pt \
    --manifest data/processed_full/manifest_full.csv \
    --output_dir data_augmentation_output \
    --target_samples 500 \
    --device cuda \
    --variations_per_video 4
```

## How It Works

1. **Analyzes imbalance**: Identifies groups with <500 samples
2. **Generates variations**: For each real video in underrepresented groups:
   - Perfect copy (same demographics)
   - Age variation (change age, keep sex/BMI)
   - Sex variation (change sex, keep age/BMI)
   - BMI variation (change BMI, keep age/sex)
3. **Balances dataset**: Generates enough variations to reach target samples per group

## Output

- **Generated videos**: `data_augmentation_output/*.mp4`
- **Manifest**: `data_augmentation_output/generated_manifest.csv`
- **Analysis**: `data_augmentation_output/imbalance_analysis.json`

## Parameters

- `--target_samples`: Target number of samples per group (default: 500)
- `--variations_per_video`: Max variations per real video (default: 4)
- `--video_length`: Number of frames (default: 16)
- `--video_size`: Spatial resolution (default: 64)
- `--device`: Device to use (cuda/cpu)

## Next Steps

After generation:
1. Combine `generated_manifest.csv` with original manifest
2. Train EF prediction model on balanced dataset
3. Validate with Grad-CAM analysis

## Notes

- Uses **PerfectReconstructionGenerator** (not C3DResUNetGenerator)
- Requires pretrained checkpoint from `use_case_3_perfect_reconstruction`
- Generates videos that preserve cardiac motion patterns
- Suitable for Grad-CAM visualization
