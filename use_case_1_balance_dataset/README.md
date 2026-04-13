# Use Case 1: Balance Dataset - C3DGAN for Underrepresented Groups

## Overview

This use case addresses dataset imbalance by generating synthetic echocardiogram videos specifically for **underrepresented demographic groups** using a Conditional 3D Generative Adversarial Network (C3DGAN). The model generates videos from **random noise** conditioned on demographic class labels, targeting groups with fewer than 500 samples.

## Problem Statement

The EchoNet Pediatric dataset exhibits severe demographic imbalances across:
- **View**: A4C (Apical 4-Chamber) vs PSAX (Parasternal Short-Axis)
- **Sex**: Female (F), Male (M), Other (O)
- **Age bins**: 0-1, 2-5, 6-10, 11-15, 16-18 years
- **BMI categories**: Underweight, Normal, Overweight, Obese

**Imbalance Statistics:**
- 19 underrepresented groups identified (<500 samples each)
- ~5,000 synthetic videos needed to balance the dataset
- Extreme cases: `A4C_O_11-15` has only 1 sample (needs 499 more)
- Groups like `A4C_M_0-1` have 145 samples (needs 355 more)

## What is Generated

**Synthetic echocardiogram videos** generated from random noise, conditioned on demographic class labels:
- **Format**: MP4 videos, 96 frames, 128×128 pixels, grayscale
- **Target**: Generate exact number needed per group to reach 500 samples
- **Total**: ~5,000 synthetic videos across 19 underrepresented groups
- **Quality**: High-quality videos suitable for downstream tasks (classification, EF prediction)

## Input

### Training Input
- **Real Videos**: Preprocessed echocardiogram videos from EchoNet Pediatric dataset
  - Format: 96 frames × 128×128 pixels, grayscale
  - Source: `data/processed/videos/`
- **Manifest CSV**: Contains video paths and demographic metadata
  - Columns: `view`, `sex`, `age_bin`, `bmi_category`, `file_path`, etc.
  - Source: `data/processed/manifest.csv` or `preprocessing/` output

### Generation Input
- **Random Noise**: 100-dimensional random vectors sampled from normal distribution
- **Class Labels**: Demographic class labels (e.g., `A4C_F_0-1_normal`, `PSAX_M_2-5_overweight`)
- **Trained Model Checkpoint**: Pre-trained generator weights

## Output

### Training Output
- **Model Checkpoints**: Saved every 10 epochs
  - Location: `c3dgan/checkpoints/checkpoint_epoch_*.pth`
  - Format: PyTorch state dict with generator and discriminator weights
- **Sample Videos**: Generated during training for quality monitoring
  - Location: `c3dgan/generated_videos/epoch_*_*.mp4`
- **Training Logs**: Tensorboard logs and training metrics
  - Location: `c3dgan/outputs/logs/`

### Generation Output
- **Synthetic Videos**: Generated videos for underrepresented groups
  - Location: `c3dgan/generated_videos/` or `synthetic_paired_dataset/`
  - Naming: `synth_XXXX_classlabel.mp4`
  - Format: MP4, 96 frames, 128×128, grayscale
- **Generated Manifest**: CSV file with metadata for all generated videos
  - Location: `c3dgan/generated_videos/generated_manifest.csv`
  - Columns: `view`, `sex`, `age_bin`, `bmi_bin`, `class_label`, `file_name`, `file_path`, `is_synthetic`, `source`

## Architecture

### Generator: ConditionalC3DGeneratorImproved

**Input**: Random noise (100-dim) + Class label embedding

**Architecture**:
```
Input: Noise [B, 100] + Label [B]
  ↓
Label Embedding: Embedding(n_classes, 100) → [B, 100]
  ↓
Concatenate: [B, 200]
  ↓
FC Layer: Linear(200 → ngf*8*6*6*6)
  ↓
Reshape: [B, ngf*8, 6, 6, 6]
  ↓
ConvTranspose3d Layers (Upsampling):
  - 6×6×6 → 12×12×12 (stride=2)
  - 12×12×12 → 24×24×24 (stride=2)
  - 24×24×24 → 48×48×48 (stride=2)
  - 48×48×48 → 96×96×96 (stride=2)
  - 96×96×96 → 96×128×128 (spatial only, stride=(1,2,2))
  ↓
Output: [B, 1, 96, 128, 128]
```

**Key Components**:
- **Label Embedding**: Maps class labels to 100-dim vectors
- **3D Transposed Convolutions**: Upsample from 6×6×6 to 96×128×128
- **Batch Normalization**: Applied after each conv layer
- **ReLU Activations**: Non-linearity between layers
- **Tanh Output**: Final activation to [-1, 1] range

**Parameters**:
- `nz`: Noise dimension (100)
- `ngf`: Generator filters (128)
- `nc`: Output channels (1, grayscale)
- `n_classes`: Number of demographic class combinations (~20)
- `video_length`: 96 frames
- `video_size`: 128×128 pixels

### Discriminator: ConditionalC3DDiscriminator

**Input**: Video [B, 1, 96, 128, 128] + Class label

**Architecture**:
```
Input: Video [B, 1, 96, 128, 128] + Label [B]
  ↓
3D Convolutions (Downsampling):
  - 128×128×96 → 64×64×48 (stride=2)
  - 64×64×48 → 32×32×24 (stride=2)
  - 32×32×24 → 16×16×12 (stride=2)
  - 16×16×12 → 8×8×6 (stride=2)
  ↓
Label Conditioning: Concatenate label embedding
  ↓
Real/Fake Classifier: Conv3d → [B, 1]
```

**Loss Function**: Binary Cross-Entropy (BCE)

## Training Process

### Configuration
- **Epochs**: 200
- **Batch Size**: 8 (adjustable based on GPU memory)
- **Learning Rate**: 
  - Generator: 0.0002
  - Discriminator: 0.0002
- **Optimizer**: Adam (betas: 0.5, 0.999)
- **Device**: CUDA (GPU required)

### Training Script
```bash
python3 c3dgan/train.py --config c3dgan/config.yaml
```

### Training Steps
1. **Data Loading**: Load real videos with class labels from manifest
2. **Noise Generation**: Sample random noise vectors for each batch
3. **Label Conditioning**: Embed class labels and concatenate with noise
4. **Generator Forward**: Generate synthetic videos
5. **Discriminator Training**: Classify real vs. fake videos
6. **Generator Training**: Minimize discriminator loss (adversarial training)
7. **Checkpointing**: Save model every 10 epochs

### Loss Functions
- **Generator Loss**: Adversarial loss (fool discriminator)
- **Discriminator Loss**: Binary cross-entropy (distinguish real/fake)

## Generation Process

### Generation Script
```bash
python3 c3dgan/generate.py \
    --checkpoint c3dgan/checkpoints/checkpoint_epoch_200.pth \
    --config c3dgan/config.yaml \
    --output_dir c3dgan/generated_videos
```

### Generation Steps
1. **Load Trained Model**: Load generator from checkpoint
2. **Identify Underrepresented Groups**: Analyze manifest to find groups <500 samples
3. **Calculate Needs**: Determine exact number of videos needed per group
4. **Generate Videos**: For each group:
   - Sample random noise vectors
   - Embed class label
   - Generate video through generator
   - Save as MP4
5. **Create Manifest**: Record all generated videos with metadata

### Underrepresented Groups Example
1. **A4C_O_11-15**: 1 sample → Generate 499
2. **PSAX_O_11-15**: 1 sample → Generate 499
3. **PSAX_O_2-5**: 1 sample → Generate 499
4. **A4C_O_2-5**: 2 samples → Generate 498
5. **A4C_M_0-1**: 145 samples → Generate 355
6. **A4C_F_0-1**: 161 samples → Generate 339
7. **PSAX_M_0-1**: 171 samples → Generate 329
8. **PSAX_F_0-1**: 197 samples → Generate 303
9. ... and 11 more groups

**Total**: ~5,000 synthetic videos

## Files and Directories

### Core Files
- **`c3dgan/models_improved.py`**: Generator and Discriminator architectures
- **`c3dgan/dataloader.py`**: Data loading with class labels
- **`c3dgan/train.py`**: Training script (if exists)
- **`c3dgan/generate.py`**: Generation script for underrepresented groups
- **`c3dgan/config.yaml`**: Configuration file
- **`c3dgan/README.md`**: Detailed C3DGAN documentation

### Pipeline Files
- **`preprocessing/preprocess.py`**: Data preprocessing pipeline
- **`preprocessing/config.yaml`**: Preprocessing configuration
- **`final_pipeline/train_c3dgan.py`**: Alternative training pipeline
- **`final_pipeline/generate_videos.py`**: Alternative generation pipeline
- **`Data_Augmentation.py`**: Data augmentation utilities

### Output Directories
- **`c3dgan/checkpoints/`**: Model checkpoints
- **`c3dgan/generated_videos/`**: Generated synthetic videos
- **`synthetic_paired_dataset/`**: Alternative output location
- **`c3dgan/outputs/logs/`**: Training logs (Tensorboard)

## Configuration

### Key Parameters (`c3dgan/config.yaml`)
```yaml
model:
  nz: 100                    # Noise dimension
  ngf: 128                   # Generator filters
  ndf: 64                    # Discriminator filters
  nc: 1                      # Output channels (grayscale)
  n_classes: 20              # Number of demographic classes
  video_length: 96           # Temporal frames
  video_size: 128            # Spatial resolution

training:
  n_epochs: 200
  batch_size: 8
  lr_g: 0.0002               # Generator learning rate
  lr_d: 0.0002               # Discriminator learning rate

augmentation:
  target_samples_per_group: 500  # Target samples per demographic group

paths:
  manifest: "data/processed/manifest.csv"
  video_dir: "data/processed/videos"
  checkpoint_dir: "c3dgan/checkpoints"
  output_dir: "c3dgan/generated_videos"
```

## Expected Results

### After Training
- **Model Quality**: Generator learns to produce realistic echocardiogram videos
- **Checkpoints**: Saved every 10 epochs for evaluation
- **Training Metrics**: Monitor discriminator accuracy and generator loss

### After Generation
- **Balanced Dataset**: All demographic groups have ≥500 samples
- **High-Quality Videos**: Synthetic videos suitable for classification
- **Manifest**: Complete manifest with original + synthetic videos
- **Ready for Downstream Tasks**: Augmented dataset ready for:
  - EF prediction model training
  - Classification tasks
  - Further augmentation (e.g., BiGAN)

## Usage Example

### Step 1: Preprocess Dataset
```bash
python preprocessing/preprocess.py --config preprocessing/config.yaml
```

### Step 2: Train C3DGAN
```bash
python c3dgan/train.py --config c3dgan/config.yaml
```

### Step 3: Generate Synthetic Videos
```bash
python c3dgan/generate.py \
    --checkpoint c3dgan/checkpoints/checkpoint_epoch_200.pth \
    --config c3dgan/config.yaml \
    --output_dir c3dgan/generated_videos
```

### Step 4: Combine with Original Dataset
```bash
# Merge manifests
cat data/processed/manifest.csv c3dgan/generated_videos/generated_manifest.csv > augmented_manifest.csv
```

## Technical Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended: RTX A5000 or similar)
- **Memory**: 24GB GPU memory (batch size 8)
- **Storage**: ~50GB for generated videos
- **Time**: 
  - Training: ~2-4 hours per epoch (depends on dataset size)
  - Generation: ~1-2 seconds per video

## Key Differences from Other Use Cases

- **Input**: Random noise (not real videos)
- **Purpose**: Balance underrepresented groups (not preserve cardiac motion)
- **Output**: Videos for specific demographic classes
- **Architecture**: Traditional GAN (not encoder-decoder)
- **Conditioning**: Class labels (not continuous demographics)

## Notes

- **Quality Monitoring**: Check generated samples during training
- **Hyperparameter Tuning**: Adjust learning rates and batch size based on GPU
- **Class Imbalance**: Model handles imbalanced training data
- **Reproducibility**: Set random seeds for consistent results
