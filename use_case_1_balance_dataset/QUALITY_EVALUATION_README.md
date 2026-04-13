# Use Case 1 Quality Evaluation: FID/FVD Metrics

## Overview

Use Case 1 generates videos from **random noise** (not reconstruction), so pixel-level metrics like SSIM/PSNR are not applicable. Instead, we use distribution-based metrics to evaluate quality:

- **FID (Fréchet Inception Distance)**: Measures feature distribution similarity between real and synthetic videos
- **FVD (Fréchet Video Distance)**: Video-specific metric for temporal consistency and realism

## Why These Metrics?

**Reviewer Concern**: "How do you know UC1 videos are realistic and not just noise?"

**Answer**: FID/FVD are the standard metrics for evaluating GAN-generated content from noise. They measure:
- **Distribution similarity**: How well synthetic videos match the statistical distribution of real videos
- **Feature-level realism**: Whether synthetic videos capture the same high-level features as real videos
- **Temporal consistency** (FVD): Whether video motion patterns are realistic

## Installation

```bash
# Install required packages
pip install torch torchvision imageio scipy tqdm pandas numpy

# For FVD (optional, requires I3D model)
pip install pytorch-i3d
# Download I3D weights: https://github.com/piergiaj/pytorch-i3d
```

## Usage

### Basic Usage (FID only)

```bash
python evaluate_quality_metrics.py \
    --real_manifest data/processed_full/manifest_full.csv \
    --synthetic_manifest use_case_1_balance_dataset/c3dgan/generated_videos/generated_manifest.csv \
    --video_dir data/processed/videos \
    --num_samples 1000 \
    --use_fid \
    --output_file uc1_quality_metrics.json
```

### Full Usage (FID + FVD)

```bash
python evaluate_quality_metrics.py \
    --real_manifest data/processed_full/manifest_full.csv \
    --synthetic_manifest use_case_1_balance_dataset/c3dgan/generated_videos/generated_manifest.csv \
    --video_dir data/processed/videos \
    --num_samples 1000 \
    --use_fid \
    --use_fvd \
    --output_file uc1_quality_metrics.json
```

### Arguments

- `--real_manifest`: Path to CSV with real video paths
- `--synthetic_manifest`: Path to CSV with synthetic video paths
- `--video_dir`: Base directory containing video files
- `--real_video_col`: Column name for real video paths (default: `video_path`)
- `--synthetic_video_col`: Column name for synthetic video paths (default: `video_path`)
- `--num_samples`: Number of videos to sample for evaluation (default: 1000)
- `--batch_size`: Batch size for feature extraction (default: 8)
- `--output_file`: Output JSON file for metrics (default: `uc1_quality_metrics.json`)
- `--device`: Device to use: `cuda` or `cpu` (default: `cuda`)
- `--use_fid`: Calculate FID (default: True)
- `--use_fvd`: Calculate FVD (requires I3D model)

## Interpreting Results

### FID Score

- **< 50**: Excellent quality (synthetic videos are very similar to real videos)
- **50-100**: Good quality (synthetic videos are realistic)
- **100-200**: Acceptable quality (synthetic videos are usable but may have artifacts)
- **> 200**: Poor quality (synthetic videos are clearly distinguishable from real videos)

### FVD Score

- **< 100**: Excellent temporal consistency
- **100-200**: Good temporal consistency
- **200-500**: Acceptable temporal consistency
- **> 500**: Poor temporal consistency

## Output Format

The script generates a JSON file with:

```json
{
  "num_real_videos": 1000,
  "num_synthetic_videos": 1000,
  "fid": 45.23,
  "fvd": 89.67
}
```

## Notes

1. **FID is sufficient for most cases**: FID is the standard metric for GAN evaluation and is easier to compute than FVD.

2. **Sampling**: The script samples videos randomly for evaluation. Use `--num_samples` to control the evaluation set size (larger = more accurate but slower).

3. **Memory**: Feature extraction requires GPU memory. Reduce `--batch_size` if you encounter OOM errors.

4. **Video format**: The script handles grayscale videos and automatically converts them to RGB for feature extraction.

## Integration with Paper

After running the evaluation, add results to the paper:

```latex
\subsection{Quality Evaluation for Use Case 1}

Use Case 1 generates videos from random noise, so reconstruction metrics (SSIM/PSNR) are not applicable. Instead, we evaluate quality using Fréchet Inception Distance (FID) and Fréchet Video Distance (FVD), standard metrics for GAN-generated content.

\textbf{Results:}
- FID Score: XX.XX (excellent/good/acceptable)
- FVD Score: XX.XX (excellent/good/acceptable)

These metrics confirm that synthetic videos capture realistic feature distributions and temporal patterns, validating the generator's ability to produce plausible echocardiogram videos from noise.
```

## Troubleshooting

**Error: "Inception model not loaded"**
- Install torchvision: `pip install torchvision`

**Error: "I3D model not loaded"**
- Install pytorch-i3d and download I3D weights
- Or skip FVD: remove `--use_fvd` flag

**Error: "CUDA out of memory"**
- Reduce `--batch_size` (e.g., `--batch_size 4`)
- Use CPU: `--device cpu` (slower but works)

**Error: "No videos found"**
- Check that `--video_dir` is correct
- Verify video paths in manifests are relative to `--video_dir`
