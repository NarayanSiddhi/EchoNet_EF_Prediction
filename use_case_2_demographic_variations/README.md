# Use Case 2: Demographic Variations - Preserving Cardiac Motion

## Overview

This use case generates **3 demographic variations** for each real echocardiogram video while **preserving cardiac motion patterns**. Using a Perfect Reconstruction C3D-GAN architecture, it systematically alters one demographic attribute (age, sex, or BMI) per variation while maintaining the original cardiac function and ejection fraction (EF).

## Problem Statement

The EchoNet Pediatric dataset has demographic imbalances that can lead to:
- **Biased EF prediction models**: Models favor overrepresented demographic groups
- **Poor generalization**: Reduced performance on underrepresented groups
- **Clinical inequity**: Unequal diagnostic accuracy across patient populations
- **Limited training data**: Insufficient samples for robust model training in minority groups

**Solution**: Generate synthetic variations that preserve cardiac motion while altering demographics, enabling:
- Dataset balancing (4× expansion: 7,791 → 31,164 videos)
- Controlled bias testing
- Robust model training across all demographic groups

## What is Generated

For each real video $V_i$ with demographics $(S_i, A_i, B_i)$ (sex, age, BMI), we generate **3 synthetic variations**:

1. **Age Variation**: $\hat{V}_{i,age} = G(V_i, D_{age})$ where $D_{age}$ encodes $(S_i, A_{new}, B_i)$
   - Preserves: Sex and BMI
   - Alters: Age bin (cycled to next available bin)

2. **Sex Variation**: $\hat{V}_{i,sex} = G(V_i, D_{sex})$ where $D_{sex}$ encodes $(S_{new}, A_i, B_i)$
   - Preserves: Age and BMI
   - Alters: Sex (F↔M, O→M)

3. **BMI Variation**: $\hat{V}_{i,bmi} = G(V_i, D_{bmi})$ where $D_{bmi}$ encodes $(S_i, A_i, B_{new})$
   - Preserves: Sex and Age
   - Alters: BMI category (cycled to next available category)

**Generation Statistics**:
- **Original videos**: 7,791
- **Variations per video**: 3 (age, sex, BMI)
- **Total synthetic videos**: **23,373**
- **Augmented dataset size**: 7,791 original + 23,373 synthetic = **31,164 total videos**
- **Expansion ratio**: 4× (4 videos per original)

## Input

### Model Input
- **Real Video**: $V_i$ - Original echocardiogram video
  - Format: 16 frames × 64×64 pixels, grayscale
  - Normalized to $[-1, 1]$ range
  - Source: `data/processed/videos/` or manifest CSV

- **Demographics**: $D$ - One-hot encoded demographic vector [11 dimensions]
  - Sex: 2 dimensions (F, M, O)
  - Age bins: 5 dimensions (0-1, 2-5, 6-10, 11-15, 16-18)
  - BMI categories: 4 dimensions (underweight, normal, overweight, obese)
  - Format: `[sex_0, sex_1, age_0-1, age_2-5, age_6-10, age_11-15, age_16-18, bmi_underweight, bmi_normal, bmi_overweight, bmi_obese]`

### Generation Script Input
- **Manifest CSV**: Path to manifest with video paths and demographics
  - Required columns: `processed_path`, `sex`, `age`, `age_bin`, `weight`, `height`, `EF`
  - Optional: `bmi_category` (computed if missing)
- **Checkpoint Path**: Trained Perfect Reconstruction C3D-GAN model
  - Location: `perfect_reconstruction_c3dgan/c3dgan_best.pt`
- **Output Directory**: Where to save generated variations

## Output

### Generated Videos
- **Location**: `demographic_variations/`
- **Naming Pattern**: `video_XXXX_varY_[age|sex|bmi]_variation.mp4`
  - `XXXX`: Original video index (zero-padded)
  - `varY`: Variation number (1=age, 2=sex, 3=BMI)
- **Format**: MP4, 16 frames, 64×64, grayscale
- **Total**: 23,373 variation videos

### Manifest File
- **Location**: `demographic_variations/variations_manifest.csv`
- **Columns**:
  - `original_id`: Index of original video
  - `original_path`: Path to original video
  - `synthetic_path`: Path to generated variation
  - `variation_type`: `age_variation`, `sex_variation`, or `bmi_variation`
  - `EF`: Original ejection fraction (preserved)
  - `original_sex`, `original_age`, `original_bmi`: Original demographics
  - `synthetic_sex`, `synthetic_age`, `synthetic_bmi`: Altered demographics
  - `variation_description`: Human-readable description of change

### Validation Outputs
- **GradCAM Results**: `best_gradcam_results/variations/`
- **Category Visualizations**: `best_gradcam_results/demographic_categories/`
- **Validation Metrics**: Attention similarity scores (cosine similarity, spatial correlation)

## Architecture

### Generator: PerfectReconstructionGenerator

**Architecture Type**: U-Net style encoder-decoder with 3D convolutions

**Input**: Video [B, 1, 16, 64, 64] + Demographics [B, 11]

**Architecture Flow**:
```
Input Video [B,1,16,64,64]
  ↓
ENCODER (4 levels, spatial downsampling)
  Level 1: 64×64 → Conv3d(1→64) + ResidualBlocks
  Level 2: 32×32 → Conv3d(64→128) + ResidualBlocks + Demo Fusion
  Level 3: 16×16 → Conv3d(128→256) + ResidualBlocks + Demo Fusion
  Level 4: 8×8   → Conv3d(256→512) + ResidualBlocks
  ↓
BOTTLENECK
  ResidualBlocks(512) × 4 + Demo Fusion
  ↓
DECODER (4 levels, spatial upsampling with skip connections)
  Level 4→3: 8×8 → 16×16 + Skip from Encoder L3
  Level 3→2: 16×16 → 32×32 + Skip from Encoder L2
  Level 2→1: 32×32 → 64×64 + Skip from Encoder L1
  Level 1: Final reconstruction
  ↓
Output: [B, 1, 16, 64, 64]
```

**Key Components**:

1. **DemographicEmbedding**:
   - Input: 11-dim one-hot vector
   - Architecture: Linear(11→64) + LayerNorm + ReLU + Dropout(0.1) + Linear(64→128)
   - Output: 128-dim demographic embedding

2. **ResidualBlock3D**:
   - Conv3d(3×3×3) + BatchNorm + ReLU
   - Conv3d(3×3×3) + BatchNorm
   - SE Attention: AdaptiveAvgPool3d → Conv(1×1×1) → ReLU → Conv(1×1×1) → Sigmoid
   - Element-wise multiplication with SE weights
   - Residual connection

3. **SpatialDemographicFusion**:
   - Projects demographic embedding to feature channels
   - Expands to spatial dimensions (B, C, T, H, W)
   - Concatenates with video features
   - Conv3d(1×1×1) fusion + BatchNorm + ReLU
   - Applied at Encoder L2, L3, and Bottleneck

4. **U-Net Skip Connections**:
   - Preserve fine-grained details across encoder-decoder levels
   - Concatenate encoder features with decoder features

**Parameters**:
- `base_channels`: 64 (scales to 128, 256, 512)
- Total parameters: ~15M

### Discriminator: PatchDiscriminator3D

**Architecture**: PatchGAN 3D with spectral normalization

**Input**: Video [B, 1, 16, 64, 64] + Demographics [B, 11]

**Architecture**:
```
Conv3d(1→64, kernel=4, stride=2) + LeakyReLU(0.2)
Conv3d(64→128, kernel=4, stride=2) + BatchNorm + LeakyReLU(0.2)
Conv3d(128→256, kernel=4, stride=2) + BatchNorm + LeakyReLU(0.2)
Conv3d(256→512, kernel=4, stride=2) + BatchNorm + LeakyReLU(0.2)
Dropout3d(0.3)
  ↓
Real/Fake Classifier: Conv3d(512→1, kernel=4, padding=1)  # Patch-level
Demographic Classifier: AdaptiveAvgPool3d(1) → Linear(512→256) → Linear(256→11)
```

**Features**:
- Spectral Normalization: Applied to all convolutions
- Patch-level discrimination: Operates on video patches
- Auxiliary demographic classification: Ensures demographic preservation

## Loss Functions

### Generator Loss

**Total Generator Loss**:
$$\mathcal{L}_G = \lambda_{pixel} \mathcal{L}_{pixel} + \lambda_{SSIM} \mathcal{L}_{SSIM} + \lambda_{temporal} \mathcal{L}_{temporal} + \lambda_{GAN} \mathcal{L}_{GAN} + \lambda_{demo} \mathcal{L}_{demo}$$

1. **Pixel Reconstruction Loss** ($\lambda_{pixel} = 100.0$):
   $$\mathcal{L}_{pixel} = \|G(V, D) - V\|_1 + 0.5 \|G(V, D) - V\|_2^2$$
   - Combines L1 and L2 losses for exact pixel matching
   - Critical for perfect reconstruction

2. **SSIM Loss** ($\lambda_{SSIM} = 5.0$):
   $$\mathcal{L}_{SSIM} = 1 - SSIM(G(V, D), V)$$
   - Preserves structural similarity
   - Ensures visual quality

3. **Temporal Consistency Loss** ($\lambda_{temporal} = 10.0$):
   $$\mathcal{L}_{temporal} = \sum_{t=1}^{T-1} \|(G(V, D)_{:,t+1} - G(V, D)_{:,t}) - (V_{:,t+1} - V_{:,t})\|_2^2$$
   - Preserves frame-to-frame differences
   - Ensures smooth cardiac motion

4. **Adversarial Loss (LSGAN)** ($\lambda_{GAN} = 1.0$):
   $$\mathcal{L}_{GAN} = \mathbb{E}[(D(G(V, D)) - 1)^2]$$
   - Ensures realistic video generation
   - Uses Mean Squared Error (MSE) for stability

5. **Demographic Preservation Loss** ($\lambda_{demo} = 5.0$):
   $$\mathcal{L}_{demo} = \text{BCE}(D_{demo}(G(V, D)), D)$$
   - Binary Cross-Entropy loss
   - Ensures demographic features are correctly encoded

### Discriminator Loss

$$\mathcal{L}_D = \frac{1}{2}[\mathbb{E}[(D(V) - 1)^2] + \mathbb{E}[(D(G(V, D)) - 0)^2]] + \lambda_{demo} \mathcal{L}_{demo}$$

## Training Process

### Training Configuration
- **Epochs**: 200
- **Batch Size**: 4
- **Optimizer**: Adam
  - Generator learning rate: $10^{-4}$
  - Discriminator learning rate: $10^{-4}$
  - Betas: (0.5, 0.999)
- **Gradient Clipping**: 1.0 (both generator and discriminator)
- **Device**: CUDA (GPU)
- **Mixed Precision**: Enabled for faster training

### Training Data
- **Dataset**: EchoNet-Pediatric, 7,791 processed videos
- **Video Format**: 16 frames × 64×64 pixels, grayscale
- **Normalization**: $[-1, 1]$ range

### Checkpointing
- Best model saved based on generator loss
- Resume capability from checkpoints
- Checkpoint saved every 10 epochs
- Location: `perfect_reconstruction_c3dgan/c3dgan_best.pt`

## Generation Process

### Generation Script
```bash
python generate_demographic_variations.py \
    --manifest data/processed/manifest.csv \
    --checkpoint perfect_reconstruction_c3dgan/c3dgan_best.pt \
    --output_dir demographic_variations \
    --video_length 16 \
    --video_size 64 \
    --device cuda
```

### Generation Workflow

For each real video $V_i$:

1. **Load and Preprocess**:
   - Load video from manifest
   - Resize to 64×64 spatial resolution
   - Sample/extend to 16 frames
   - Normalize to $[-1, 1]$

2. **Extract Original Demographics**:
   - Sex: One-hot encode (F=0, M=1, O=0)
   - Age: One-hot encode into 5 bins (0-1, 2-5, 6-10, 11-15, 16-18)
   - BMI: One-hot encode into 4 categories (underweight, normal, overweight, obese)
   - Combined: 11-dimensional vector

3. **Generate Three Variations**:

   **Age Variation**:
   ```python
   # Get current age bin index
   orig_age_idx = argmax(demographics[2:7])
   # Cycle to next age bin
   next_age_idx = (orig_age_idx + 1) % 5
   # Create new demographics
   age_varied_demo = demographics.clone()
   age_varied_demo[2:7] = 0.0
   age_varied_demo[2 + next_age_idx] = 1.0
   # Generate
   synthetic_age = G(real_video, age_varied_demo)
   ```

   **Sex Variation**:
   ```python
   # Flip sex (F↔M, O→M)
   orig_sex_idx = argmax(demographics[:2])
   next_sex_idx = 1 - orig_sex_idx if orig_sex_idx in [0,1] else 1
   sex_varied_demo = demographics.clone()
   sex_varied_demo[:2] = 0.0
   sex_varied_demo[next_sex_idx] = 1.0
   synthetic_sex = G(real_video, sex_varied_demo)
   ```

   **BMI Variation**:
   ```python
   # Cycle to next BMI category
   orig_bmi_idx = argmax(demographics[7:])
   next_bmi_idx = (orig_bmi_idx + 1) % 4
   bmi_varied_demo = demographics.clone()
   bmi_varied_demo[7:] = 0.0
   bmi_varied_demo[7 + next_bmi_idx] = 1.0
   synthetic_bmi = G(real_video, bmi_varied_demo)
   ```

4. **Post-process and Save**:
   - Denormalize from $[-1, 1]$ to $[0, 255]$
   - Convert to uint8
   - Save as MP4 video
   - Record metadata in manifest CSV

## Validation: GradCAM Analysis

### Validation Methodology
- **Sample size**: 50 randomly selected video pairs (original + 3 variations)
- **Total comparisons**: 150 (50 × 3 variations)
- **Metrics**:
  - **Cosine Similarity**: Overall attention pattern similarity (target: > 0.75)
  - **Spatial Correlation**: Spatial distribution similarity (target: > 0.70)

### Validation Results

**Overall Performance**:
- **Cosine Similarity**: **0.8781 ± 0.1139** (range: 0.4886 - 0.9918)
- **Spatial Correlation**: **0.9204 ± 0.0933** (range: 0.5371 - 0.9964)
- **Samples above thresholds**:
  - Cosine similarity > 0.75: **130/150 (86.7%)**
  - Spatial correlation > 0.70: **143/150 (95.3%)**

**By Variation Type**:
- **Age Variations** (50 samples):
  - Cosine: **0.8786 ± 0.1100**
  - Spatial: **0.9218 ± 0.0842**
- **Sex Variations** (50 samples):
  - Cosine: **0.8783 ± 0.1195**
  - Spatial: **0.9198 ± 0.0998**
- **BMI Variations** (50 samples):
  - Cosine: **0.8773 ± 0.1145**
  - Spatial: **0.9196 ± 0.0972**

### Validation Assessment

**✅ Attention Pattern Preservation: PASS**
- Target: Cosine similarity > 0.75
- Actual: 0.8781 (exceeds threshold by 17%)
- **Interpretation**: Synthetic variations follow the same attention patterns as originals, indicating preserved cardiac motion

**✅ Spatial Attention Consistency: PASS**
- Target: Spatial correlation > 0.70
- Actual: 0.9204 (exceeds threshold by 31%)
- **Interpretation**: Model focuses on the same cardiac regions (ventricles, atria, valves) in both original and synthetic videos

## Quality Metrics

**Inherited from Perfect Reconstruction Training**:
- **SSIM**: > 0.99 (near-perfect structural similarity)
- **PSNR**: > 48 dB (high signal-to-noise ratio)
- **MSE**: < 1.0 (low pixel-level error)

## Files and Scripts

### Generation Scripts
- **`generate_demographic_variations.py`**: Main generation script
- **`validate_demographic_variations.py`**: Validation script
- **`run_demographic_variations_byobu.sh`**: Batch generation script

### GradCAM Scripts
- **`gradcam_demographic_variations.py`**: GradCAM analysis for variations
- **`generate_demographic_category_gradcam.py`**: Category-specific GradCAM
- **`run_gradcam_variations_byobu.sh`**: Batch GradCAM script

### Documentation
- **`MD1_DATA_IMBALANCE.md`**: Comprehensive documentation
- **`MICCAI_DEMOGRAPHIC_VARIATIONS_SECTION.md`**: Research paper section
- **`MICCAI_DEMOGRAPHIC_VARIATIONS_CONCISE.md`**: Concise version
- **`CORRECTED_DEMOGRAPHIC_TABLE.md`**: Demographic statistics

### Output Directories
- **`demographic_variations/`**: Generated variation videos
- **`best_gradcam_results/variations/`**: GradCAM results
- **`best_gradcam_results/demographic_categories/`**: Category visualizations

## Key Findings

1. **Cardiac Motion Preservation**: High attention similarity (0.88) confirms variations preserve cardiac motion patterns
2. **Spatial Consistency**: High spatial correlation (0.92) indicates model focuses on same cardiac regions
3. **Uniform Performance**: All variation types show similar attention patterns, confirming consistent quality
4. **Clinical Relevance**: Attention maps highlight cardiac structures (ventricles, valves) in both originals and variations
5. **Dataset Balancing**: 4× dataset expansion enables balanced training across demographic groups

## Application to EF Prediction

The augmented dataset (original + synthetic variations) is used to:
1. **Train EF prediction models** on a more balanced demographic distribution
2. **Evaluate model bias** across demographic groups using controlled variations
3. **Test robustness** by assessing whether models rely on demographics vs. cardiac function

## Technical Specifications Summary

| Component | Specification |
|-----------|--------------|
| **Architecture** | U-Net style 3D GAN with encoder-decoder |
| **Input/Output** | 16 frames × 64×64 grayscale videos |
| **Demographics** | 11-dimensional one-hot encoding (sex: 2, age: 5, BMI: 4) |
| **Generator Parameters** | ~15M parameters |
| **Discriminator Parameters** | ~5M parameters |
| **Training Epochs** | 200 |
| **Batch Size** | 4 |
| **Learning Rates** | Generator: 1e-4, Discriminator: 1e-4 |
| **Loss Weights** | Pixel: 100, SSIM: 5, Temporal: 10, GAN: 1, Demo: 5 |
| **Original Dataset Size** | 7,791 videos |
| **Synthetic Videos Generated** | 23,373 (3 per original) |
| **Augmented Dataset Size** | 31,164 videos |
| **GradCAM Validation** | 150 comparisons (50 samples × 3 variations) |
| **Attention Similarity** | 0.8781 (cosine), 0.9204 (spatial) |
| **Validation Status** | ✅ PASS (both metrics exceed thresholds) |
