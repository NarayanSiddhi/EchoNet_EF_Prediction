# C3DGAN: Conditional 3D Convolutional GAN for Echocardiogram Video Generation

## Overview

This module implements a **Conditional C3DGAN** (3D Convolutional Generative Adversarial Network) to generate synthetic echocardiogram videos specifically for **underrepresented groups** in the EchoNet Pediatric dataset. The goal is to balance the dataset by generating synthetic videos for groups with fewer than 500 samples.

## Problem Statement

The EchoNet Pediatric dataset has imbalanced class distributions across:
- **View**: A4C vs PSAX
- **Sex**: Female (F), Male (M), Other (O)
- **Age bins**: 0-1, 2-5, 6-10, 11-15, 16-18 years

**Current Status:**
- 19 underrepresented groups identified (<500 samples each)
- ~5,000 synthetic videos needed to balance the dataset
- Groups like `A4C_O_11-15` have only 1 sample (need 499 more)

## Architecture

### Conditional Generator (`ConditionalC3DGenerator`)
- **Input**: Random noise (100-dim) + Class label embedding
- **Output**: Synthetic video (1 channel, 96 frames, 128×128 pixels)
- **Architecture**: 3D transposed convolutions with batch normalization
- **Conditioning**: Class labels embedded and concatenated with noise

### Conditional Discriminator (`ConditionalC3DDiscriminator`)
- **Input**: Video (1 channel, 96 frames, 128×128) + Class label
- **Output**: Real/Fake probability
- **Architecture**: 3D convolutions with label conditioning
- **Loss**: Binary Cross-Entropy (BCE)

## Files

- **`models.py`**: Generator and Discriminator architectures
- **`dataloader.py`**: Data loading with class labels
- **`train.py`**: Training script with GPU acceleration
- **`generate.py`**: Script to generate synthetic videos for underrepresented groups
- **`config.yaml`**: Configuration file

## Usage

### 1. Training

Train the conditional C3DGAN on the full dataset:

```bash
python3 c3dgan/train.py --config c3dgan/config.yaml
```

**Training Parameters:**
- Epochs: 200
- Batch size: 8 (adjust based on GPU memory)
- Learning rate: 0.0002
- Device: CUDA (GPU)

**Outputs:**
- Checkpoints saved every 10 epochs: `c3dgan/checkpoints/checkpoint_epoch_*.pth`
- Sample videos: `c3dgan/generated_videos/epoch_*_*.mp4`
- Tensorboard logs: `c3dgan/outputs/logs/`

### 2. Generation

After training, generate synthetic videos for underrepresented groups:

```bash
python3 c3dgan/generate.py \
    --checkpoint c3dgan/checkpoints/checkpoint_epoch_200.pth \
    --config c3dgan/config.yaml \
    --output_dir c3dgan/generated_videos
```

This will:
- Load the trained generator
- Identify underrepresented groups
- Generate the exact number of videos needed per group
- Save videos and create a manifest CSV

### 3. Monitor Training

```bash
# View training logs
tail -f c3dgan/training.log

# Tensorboard
tensorboard --logdir c3dgan/outputs/logs
```

## Configuration

Edit `c3dgan/config.yaml` to adjust:

- **Model parameters**: `nz`, `ngf`, `ndf`, `video_length`, `video_size`
- **Training**: `n_epochs`, `batch_size`, `lr_g`, `lr_d`
- **Augmentation target**: `target_samples_per_group` (default: 500)
- **Paths**: Manifest, video directories, output directories

## Underrepresented Groups

The system automatically identifies groups with <500 samples:

1. **A4C_O_11-15**: 1 sample (need 499)
2. **PSAX_O_11-15**: 1 sample (need 499)
3. **PSAX_O_2-5**: 1 sample (need 499)
4. **A4C_O_2-5**: 2 samples (need 498)
5. **A4C_M_0-1**: 145 samples (need 355)
6. **A4C_F_0-1**: 161 samples (need 339)
7. **PSAX_M_0-1**: 171 samples (need 329)
8. **PSAX_F_0-1**: 197 samples (need 303)
9. ... and 11 more groups

**Total**: ~5,000 synthetic videos needed

## Expected Results

After training and generation:
- Balanced dataset with ≥500 samples per group
- High-quality synthetic videos suitable for classification
- Improved model performance on underrepresented groups
- Ready for downstream tasks (e.g., BiGAN augmentation)

## Notes

- **GPU Required**: Training uses CUDA acceleration (NVIDIA RTX A5000)
- **Memory**: Batch size 8 recommended for 24GB GPU
- **Time**: ~2-4 hours per epoch (depends on dataset size)
- **Quality**: Monitor generated samples during training to assess quality

## Integration with Preprocessing

The C3DGAN uses videos from the preprocessing pipeline:
- Input: Preprocessed videos (96 frames, 128×128, grayscale)
- Output: Synthetic videos in same format
- Manifest: Generated videos added to dataset manifest

## Next Steps

1. Train C3DGAN for 200 epochs
2. Generate synthetic videos for underrepresented groups
3. Combine with original dataset
4. Use augmented dataset for downstream tasks (BiGAN, classification, etc.)

