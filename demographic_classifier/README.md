# Demographic Classifier Evaluation

This module implements **Option A: Demographic Classification Accuracy** and **Option B: Distribution Divergence Metrics** to strengthen evaluation beyond reconstruction quality (SSIM, PSNR, Grad-CAM).

## Overview

### Option A: Demographic Classification Accuracy
- Trains a 3D CNN classifier to predict sex, age bin, and BMI category from videos
- Evaluates classifier on both real and synthetic videos
- **If classifier accuracy is similar → proves demographic conditioning worked**

### Option B: Distribution Divergence Metrics
- Computes KL divergence, Jensen-Shannon divergence, and entropy
- Shows distribution entropy increases and skew reduces quantitatively
- **Makes redistribution mathematically grounded**

## Files

- `train_demographic_classifier.py` - Train classifier on real videos
- `evaluate_real_vs_synthetic.py` - Compare classifier performance on real vs synthetic videos
- `calculate_distribution_metrics.py` - Calculate distribution divergence metrics

## Usage

### Step 1: Train the Demographic Classifier

```bash
cd demographic_classifier
python train_demographic_classifier.py
```

This will:
- Train a 3D CNN to predict sex, age bin, and BMI from videos
- Use real videos from the training set
- Save the best model to `checkpoints/best.pth`
- Save training history to `checkpoints/training_history.json`
- Save baseline metrics to `results/real_videos_metrics.json`

**Expected Training Time**: ~30-60 minutes depending on GPU

### Step 2: Evaluate Real vs Synthetic Videos

```bash
python evaluate_real_vs_synthetic.py
```

This will:
- Load the trained classifier
- Evaluate on real videos (validation set)
- Evaluate on synthetic videos (Use Case 2 variations)
- Compare accuracies
- Save comparison results to `results/real_vs_synthetic_comparison.json`

**Interpretation**:
- If accuracy difference < 0.05: ✓ Excellent - demographic conditioning works perfectly
- If accuracy difference < 0.10: ✓ Good - demographic conditioning is effective
- If accuracy difference < 0.15: ⚠️ Moderate - partial effectiveness
- If accuracy difference ≥ 0.15: ❌ Poor - conditioning may not be working

### Step 3: Calculate Distribution Metrics

```bash
python calculate_distribution_metrics.py
```

This will:
- Load original dataset distribution
- Load augmented dataset (original + synthetic from Use Case 2)
- Calculate KL divergence, JS divergence, and entropy for each demographic dimension
- Show entropy increase (higher = more balanced)
- Save results to `results/distribution_metrics.json`

**Interpretation**:
- Higher entropy increase = more effective redistribution
- Lower JS divergence = augmented distribution similar to original (expected, as synthetic videos are derived from real)

## Results Location

All results are saved in `demographic_classifier/results/`:
- `real_videos_metrics.json` - Baseline classifier performance on real videos
- `real_vs_synthetic_comparison.json` - Comparison between real and synthetic videos
- `distribution_metrics.json` - Distribution divergence metrics

## Model Architecture

The demographic classifier uses:
- **3D CNN Backbone**: 4 blocks of 3D convolutions (64→128→256→512 channels)
- **Classification Heads**: Separate heads for sex (3 classes), age (8 bins), and BMI (4 categories)
- **Input**: Grayscale videos, 32 frames × 128×128 pixels
- **Output**: Multi-task classification (sex, age bin, BMI category)

## Requirements

- PyTorch
- scikit-learn
- scipy
- pandas
- numpy
- opencv-python
- tqdm
- pyyaml

## Notes

- The classifier is trained only on real videos to establish a baseline
- Synthetic videos are evaluated using the same classifier
- Similar accuracy on both indicates synthetic videos encode demographic signals correctly
- This validation does NOT require EF labels, making it independent of downstream tasks
