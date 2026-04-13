# C3D-GAN Architecture: Complete Flow Description

## Overview

The C3D-GAN (Conditional 3D Generative Adversarial Network) is a **U-Net style encoder-decoder architecture** with 3D convolutions designed for generating synthetic echocardiogram videos **conditioned on demographic attributes**. 

**Note on Naming**: Despite the name "C3DGAN", this architecture uses a **U-Net encoder-decoder** structure, not the traditional C3D architecture (Tran et al. 2015). The "C3D" here refers to "3D Convolutional" operations, not the specific C3D network architecture. A more accurate name would be "Conditional 3D U-Net GAN" or "3D U-Net GAN". 

### Key Point: Demographics are Required Inputs

**YES, demographics are given as input to the model to generate synthetic videos.** The generator function is:
$$\hat{V} = G(V, D)$$

Where:
- $V$: Real echocardiogram video (required input)
- $D$: Demographic attributes (required input) - one-hot encoded vector of 11 dimensions
- $\hat{V}$: Generated synthetic video (output)

The model is a **conditional GAN**, meaning it generates videos conditioned on both the input video AND the demographic attributes. Demographics are fused into the generator at multiple levels to control the generation process.

### Two Use Cases

The same architecture is used for two distinct purposes:

1. **Use Case 1: Perfect Reconstruction (1 Real → 1 Synthetic)** - For EF prediction and bias testing
   - Demographics input: **Same as original video** ($D_i = D_{original}$)
   
2. **Use Case 2: Demographic Variations (1 Real → 3 Synthetic)** - For addressing data imbalance
   - Demographics input: **Altered from original** ($D_{variation} \neq D_{original}$)
   - One demographic attribute changed per variation

The architecture processes **both video and demographic inputs** through embedding layers, a generator core with multi-level fusion, and produces synthetic video outputs. The key difference between the two use cases lies in **what demographic values are provided as input**.

---

## System Flow Diagram (Both Use Cases)

```
┌─────────────────────────────────────────────┐
│              INPUT                           │
├─────────────────────────────────────────────┤
│  • Real Video [B,1,16,64,64]               │
│    (Grayscale, normalized)                  │
│  • Demographics [B,11]                      │
│    (Sex=2, Age=5, BMI=4)                    │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│         VIDEO EMBEDDING                      │
├─────────────────────────────────────────────┤
│  Features:                                  │
│  • Conv3d(1→64)                             │
│  • Output: H_V^0 [B,64,16,64,64]           │
└──────────────┬──────────────────────────────┘
               │
               ├──────────────────────────────┐
               │                              │
               ▼                              ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│  USE CASE 1              │  │  USE CASE 2              │
│  Demographics:          │  │  Demographics:          │
│  • Same as original     │  │  • Age Variation        │
│                         │  │  • Sex Variation        │
│                         │  │  • BMI Variation        │
└──────────────┬───────────┘  └──────────────┬───────────┘
               │                             │
               └──────────────┬──────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────┐
│         DEMOGRAPHIC EMBEDDING              │
├─────────────────────────────────────────────┤
│  Features:                                  │
│  • Linear(11→64→128)                       │
│  • Output: H_D^0 [B,128]                   │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│              FUSION                           │
├─────────────────────────────────────────────┤
│  Features:                                  │
│  • Combines Video + Demographics            │
│  • Applied at Encoder L1, L2, L3, Bottleneck│
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│         GENERATOR (U-Net)                    │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │         ENCODER                     │   │
│  │  Features:                          │   │
│  │  • Conv3d layers (downsample)        │   │
│  │  • ResidualBlocks with SE Attention │   │
│  │  • Levels: L1→L2→L3→L4              │   │
│  │  • Demographic Fusion at each level │   │
│  └──────────────┬──────────────────────┘   │
│                 │                          │
│                 │ Skip Connections         │
│                 │ (Preserve details)       │
│                 │                          │
│                 ▼                          │
│  ┌─────────────────────────────────────┐   │
│  │         DECODER                     │   │
│  │  Features:                          │   │
│  │  • ConvTranspose3d (upsample)       │   │
│  │  • ResidualBlocks                   │   │
│  │  • Levels: L4→L3→L2→L1              │   │
│  │  • Skip connections from encoder     │   │
│  └──────────────┬──────────────────────┘   │
│                 │                          │
└─────────────────┼──────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│         DISCRIMINATOR                       │
├─────────────────────────────────────────────┤
│  Features:                                  │
│  • Video Feature Extraction:                │
│    Conv3d×4 (1→64→128→256→512)             │
│  • Label Embedding:                         │
│    Embedding(n_classes→512)                 │
│  • Classification:                          │
│    Real/Fake logits [B,1]                   │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│              OUTPUT                         │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │  USE CASE 1 OUTPUT                 │   │
│  │  • 1 Synthetic Video               │   │
│  │  • Same demographics                │   │
│  │  • SSIM > 0.99, PSNR > 48 dB       │   │
│  │  • Preserves all cardiac motion     │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │  USE CASE 2 OUTPUT                 │   │
│  │  • 3 Synthetic Videos               │   │
│  │  • Age/Sex/BMI variations           │   │
│  │  • Preserves cardiac motion         │   │
│  │  • Alters 1 demographic per video   │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  Features:                                  │
│  • GAP → H_V^L [B,1024]                    │
│  • H_D^L [B,128]                           │
│  • Concat → FC Layers → Final Output       │
│                                             │
│  GradCAM:                                   │
│  • Attention maps showing focus regions     │
│  • Validates cardiac motion preservation    │
│                                             │
└─────────────────────────────────────────────┘
```

### Important Notes:

**Both use cases use the SAME architecture and flow** - only the demographic input values differ:
- Same generator (U-Net encoder-decoder)
- Same discriminator
- Same training process
- Only difference: What demographics are fed as input

**Use Case 1 (Perfect Reconstruction)**:
- Demographics: Same as original video
- Output: 1 synthetic video (near-perfect copy)
- Purpose: EF prediction, bias testing

**Use Case 2 (Demographic Variations)**:
- Demographics: 3 different variations (Age/Sex/BMI changed)
- Output: 3 synthetic videos (one per variation)
- Purpose: Dataset balancing, address imbalance

### Key Explanations:

**Skip Connections**: Direct paths from encoder to decoder that preserve fine details (like edges, textures). Without them, details get lost during compression.

**Discriminator**: Checks if generated video looks real. Uses video features + class labels to classify real vs fake.

**Losses**: Error signals that tell the model how wrong it is. Multiple losses ensure the video is accurate, realistic, and preserves motion.

**GradCAM**: Visualizes which parts of the input video the generator focuses on when creating synthetic video (shows attention maps).

---

## 1. Use Cases Overview

### 1.1 Use Case 1: Perfect Reconstruction (1 Real → 1 Synthetic)

**Purpose**: Generate near-perfect synthetic copies for EF prediction and bias testing.

**Input**: Real video $V_i$ + Demographics $D_i$ (same as original)

**Process**: Single forward pass: $\hat{V}_i = G(V_i, D_i)$

**Output**: Synthetic video $\hat{V}_i$ (SSIM > 0.99, PSNR > 48 dB)

**Result**: 7,791 real → 7,791 synthetic copies

### 1.2 Use Case 2: Demographic Variations (1 Real → 3 Synthetic)

**Purpose**: Generate synthetic videos with altered demographics to address dataset imbalance.

**Input**: Real video $V_i$ + Original demographics $(S_i, A_i, B_i)$

**Process**: Three forward passes with different demographic conditions:
1. **Age Variation**: $G(V_i, D_{age})$ where $D_{age} = (S_i, A_{new}, B_i)$
2. **Sex Variation**: $G(V_i, D_{sex})$ where $D_{sex} = (S_{new}, A_i, B_i)$
3. **BMI Variation**: $G(V_i, D_{bmi})$ where $D_{bmi} = (S_i, A_i, B_{new})$

**Output**: 3 synthetic videos preserving cardiac motion, altering one demographic each

**Result**: 7,791 real → 23,373 synthetic (4× dataset expansion)

---

## 2. Input Specifications

### 2.1 Echocardiogram Video Input

**Format**: Grayscale video `[B, 1, 16, 64, 64]`
- 16 temporal frames, 64×64 spatial resolution
- Normalized to `[-1, 1]`

### 2.2 Demographic Input (REQUIRED)

**Format**: One-hot encoded vector `[B, 11]`
- Sex: 2 dims (F/M/O)
- Age bins: 5 dims (0-1, 2-5, 6-10, 11-15, 16-18)
- BMI: 4 dims (underweight, normal, overweight, obese)

**Usage**: Embedded to `[B, 128]` and fused at Encoder L1, L2, L3, and Bottleneck

**Examples**:
- **Use Case 1**: Same demographics as original video
- **Use Case 2**: One attribute changed per variation (age/sex/BMI)

---

## 3. Architecture Flow

The same C3D-GAN architecture is used for both use cases. The generator $G(V, D)$ takes a video $V$ and demographics $D$ as input and produces a synthetic video $\hat{V}$. The difference between use cases is only in the demographic conditioning:
- **Use Case 1**: $D_{input} = D_{original}$ (same demographics)
- **Use Case 2**: $D_{input} \neq D_{original}$ (altered demographics)

### 3.1 Embedding Layers

#### 3.1.1 Video Embedding Path

**Video Input**:
- Input: `[B, 1, 16, 64, 64]` (grayscale, normalized to [-1, 1])
- **Purpose**: Raw echocardiogram video frames

**3D Encoder**:
- `Conv3d(1 → 64, kernel=1×1×1)`: Channel expansion
- Output: `[B, 64, 16, 64, 64]`
- **Purpose**: Initial feature extraction, preserves temporal dimension

**Video Features ($H_V^0$)**:
- Output: `[B, 64, 16, 64, 64]`
- **Purpose**: Initial video embedding for generator core

#### 3.1.2 Demographic Embedding Path

**Demographics Input**:
- Input: `[B, 11]` (one-hot: Sex=2, Age=5, BMI=4)
- **Purpose**: Demographic conditioning attributes

**Demographic Embedding**:
- `Linear(11 → 64)` + LayerNorm + ReLU + Dropout(0.1)
- `Linear(64 → 128)` + LayerNorm + ReLU
- Output: `[B, 128]`
- **Purpose**: Project demographics to feature space for fusion

**Demographic Features ($H_D^0$)**:
- Output: `[B, 128]`
- **Purpose**: Fused at multiple generator levels (Encoder L1, L2, L3, Bottleneck)

---

### 3.2 Generator Core (Encoder-Decoder)

The generator core follows a U-Net architecture with 4 encoder levels, a bottleneck, and 4 decoder levels. Demographic features are fused at multiple levels through Spatial Demographic Fusion.

#### 3.2.1 Encoder Pathway

**Encoder Level 1** (64×64 spatial):
- `Conv3d(64→64)` + ResidualBlock3D × 2 + Demographic Fusion
- Output: `[B, 64, 16, 64, 64]`
- **Purpose**: Extract features, fuse demographics, save skip connection

**Encoder Level 2** (32×32 spatial):
- `Conv3d(64→128, stride=(1,2,2))` + ResidualBlock3D × 2 + Demographic Fusion
- Output: `[B, 128, 16, 32, 32]`
- **Purpose**: Spatial downsampling, feature refinement, save skip connection

**Encoder Level 3** (16×16 spatial):
- `Conv3d(128→256, stride=(1,2,2))` + ResidualBlock3D × 2 + Demographic Fusion
- Output: `[B, 256, 16, 16, 16]`
- **Purpose**: Further downsampling, save skip connection

**Encoder Level 4** (8×8 spatial):
- `Conv3d(256→512, stride=(1,2,2))` + ResidualBlock3D × 2
- Output: `[B, 512, 16, 8, 8]`
- **Purpose**: Final encoding level before bottleneck

#### 3.2.2 Bottleneck

**Bottleneck**:
- ResidualBlock3D(512) × 4 + Demographic Fusion
- Output: `[B, 512, 16, 8, 8]`
- **Purpose**: Deep feature refinement with final demographic conditioning

#### 3.2.3 Decoder Pathway

**Decoder Level 4** (8×8 → 16×16):
- `ConvTranspose3d(512→256, stride=(1,2,2))` + ResidualBlock3D + Skip Connection
- Output: `[B, 512, 16, 16, 16]`
- **Purpose**: Upsample and fuse with encoder L3 features

**Decoder Level 3** (16×16 → 32×32):
- `ConvTranspose3d(512→128, stride=(1,2,2))` + ResidualBlock3D + Skip Connection
- Output: `[B, 256, 16, 32, 32]`
- **Purpose**: Upsample and fuse with encoder L2 features

**Decoder Level 2** (32×32 → 64×64):
- `ConvTranspose3d(256→64, stride=(1,2,2))` + ResidualBlock3D + Skip Connection
- Output: `[B, 128, 16, 64, 64]`
- **Purpose**: Upsample and fuse with encoder L1 features

**Decoder Level 1** (Final Reconstruction):
- `Conv3d(128→64)` + ResidualBlock3D × 2 + `Conv3d(64→32)` + `Conv3d(32→1)` + Tanh
- Output: `[B, 1, 16, 64, 64]`
- **Purpose**: Generate final synthetic video

#### 3.2.4 Key Components

**ResidualBlock3D**:
- Two `Conv3d(3×3×3)` layers with BatchNorm
- SE Attention: Channel-wise feature recalibration
- Residual connection: `output = SE_attention(conv_out) + input`
- **Purpose**: Feature refinement with attention

**Spatial Demographic Fusion**:
- Project demographics `[B,128]` → expand to `[B,C,T,H,W]`
- Concatenate with video features → `Conv3d(2C→C, 1×1×1)` fusion
- Applied at: Encoder L1, L2, L3, and Bottleneck
- **Purpose**: Condition generation on demographics at multiple scales

**Skip Connections**:
- U-Net style: Encoder L1→Decoder L1, L2→L2, L3→L3
- Method: Feature concatenation
- **Purpose**: Preserve fine-grained details for reconstruction

---

### 3.3 Output Layer

**Global Average Pooling**:
- `AdaptiveAvgPool3d(1)` on synthetic video `[B,1,16,64,64]`
- Output: $H_V^L$ = `[B, 1024]`
- **Purpose**: Aggregate video features

**Feature Concatenation**:
- Concatenate: $H_V^L$ `[B,1024]` + $H_D^L$ `[B,128]`
- Output: `[B, 1152]`
- **Purpose**: Combine video and demographic features

**Fully Connected Layers**:
- `Linear(1152→512)` + ReLU + Dropout(0.3)
- `Linear(512→256)` + ReLU + Dropout(0.3)
- `Linear(256→output_dim)`
- **Purpose**: Final prediction/regression head

---

## 4. Discriminator Architecture (For Training)

### 4.1 Conditional 3D Discriminator Overview

The discriminator is a **conditional PatchGAN** that distinguishes between real and synthetic videos while also classifying demographic attributes. It operates on video patches to assess realism at multiple spatial locations.

**Key Features**:
- **Conditional Architecture**: Takes both video and class labels as input
- **PatchGAN Design**: Operates on video patches for better training stability
- **Dual Output**: Real/Fake classification + Demographic classification
- **Spectral Normalization**: Applied to all convolutions for training stability

### 4.2 Input Processing

**Inputs**:
- **Video**: `[B, 1, 16, 64, 64]` (real or synthetic, grayscale)
- **Class Labels**: `[B]` (integer class indices for demographic groups)

**Label Embedding**:
- `Embedding(n_classes → ndf*8)`: Projects class labels to feature space
- Output: `[B, ndf*8]` (e.g., `[B, 512]` for ndf=64)
- Purpose: Condition discriminator on demographic attributes

### 4.3 Video Feature Extraction Pathway

**Stage 1: Initial Convolution**
- `Conv3d(1 → ndf, kernel=4×4×4, stride=2, padding=1)`
- LeakyReLU(0.2)
- Output: `[B, 64, 16, 32, 32]` (spatial downsampling: 64×64 → 32×32)
- **Purpose**: Extract low-level spatiotemporal features

**Stage 2: Progressive Downsampling**
- `Conv3d(64 → 128, kernel=4×4×4, stride=2)` + BatchNorm3d + LeakyReLU(0.2)
  - Output: `[B, 128, 16, 16, 16]` (32×32 → 16×16)
- `Conv3d(128 → 256, kernel=4×4×4, stride=2)` + BatchNorm3d + LeakyReLU(0.2)
  - Output: `[B, 256, 16, 8, 8]` (16×16 → 8×8)
- `Conv3d(256 → 512, kernel=4×4×4, stride=2)` + BatchNorm3d + LeakyReLU(0.2)
  - Output: `[B, 512, 16, 4, 4]` (8×8 → 4×4)

**Key Design Choices**:
- **Spatial-only Downsampling**: Temporal dimension (16 frames) preserved throughout
- **Progressive Channel Expansion**: 1 → 64 → 128 → 256 → 512 channels
- **Batch Normalization**: Applied after each convolution (except first) for stability

### 4.4 Feature Aggregation

**Global Average Pooling**:
- `AdaptiveAvgPool3d(1)`: Reduces spatial and temporal dimensions to 1×1×1
- Flatten: `[B, 512, 16, 4, 4]` → `[B, 512]`
- **Purpose**: Aggregate spatiotemporal features into a single feature vector

### 4.5 Conditional Fusion

**Label-Video Feature Fusion**:
- Video features: `[B, 512]` (from GAP)
- Label embedding: `[B, 512]` (from embedding layer)
- Concatenation: `[B, 1024]` (video + label features)
- **Purpose**: Combine video realism features with demographic conditioning

### 4.6 Real/Fake Classification Head

**Final Classification Layer**:
- `Linear(1024 → 1)`: Binary classification (real vs fake)
- Output: Logits `[B, 1]` (no sigmoid - uses BCEWithLogitsLoss)
- **Loss Function**: Binary Cross-Entropy with Logits (numerically stable)

**Training Behavior**:
- **Real videos**: Target output = 1 (high logit)
- **Synthetic videos**: Target output = 0 (low logit)
- **Gradient Penalty**: Applied for WGAN-GP training (λ=10.0)

### 4.7 Discriminator Loss Components

**Adversarial Loss** (LSGAN):
$$\mathcal{L}_{adv} = \frac{1}{2}[\mathbb{E}[(D(V_{real}) - 1)^2] + \mathbb{E}[(D(G(V, D)) - 0)^2]]$$

**Gradient Penalty** (WGAN-GP):
$$\mathcal{L}_{GP} = \lambda_{GP} \mathbb{E}[(\|\nabla_{\hat{V}} D(\hat{V})\|_2 - 1)^2]$$
- Applied to interpolated samples between real and fake videos
- Prevents discriminator from becoming too confident

**Total Discriminator Loss**:
$$\mathcal{L}_D = \mathcal{L}_{adv} + \mathcal{L}_{GP}$$

### 4.8 Training Strategy

**Update Frequency**:
- **n_critic = 5**: Discriminator updates 5 times per generator update
- **Purpose**: Keep discriminator ahead of generator for stable training

**Training Steps**:
1. Forward pass with real videos → compute real loss
2. Forward pass with synthetic videos → compute fake loss
3. Compute gradient penalty on interpolated samples
4. Backward pass and update discriminator weights
5. Repeat 5 times before updating generator

**Stability Techniques**:
- Spectral Normalization: Constrains discriminator capacity
- Gradient Penalty: Prevents mode collapse
- Mixed Precision (FP16): Faster training with lower memory

### 4.9 Discriminator Architecture Summary

| Component | Specification |
|-----------|--------------|
| **Type** | Conditional 3D PatchGAN |
| **Input** | Video `[B,1,16,64,64]` + Labels `[B]` |
| **Feature Extraction** | 4× 3D Convolutions (1→64→128→256→512) |
| **Downsampling** | Spatial only (64×64 → 4×4), temporal preserved |
| **Conditioning** | Label embedding (n_classes → 512) |
| **Fusion** | Concatenation of video + label features |
| **Output** | Real/Fake logits `[B,1]` |
| **Parameters** | ~5M |
| **Normalization** | Spectral Normalization on all convolutions |
| **Loss** | LSGAN + Gradient Penalty (WGAN-GP) |

---

## 5. Loss Functions

### 5.1 Generator Loss

$$\mathcal{L}_G = \lambda_{pixel} \mathcal{L}_{pixel} + \lambda_{SSIM} \mathcal{L}_{SSIM} + \lambda_{temporal} \mathcal{L}_{temporal} + \lambda_{GAN} \mathcal{L}_{GAN} + \lambda_{demo} \mathcal{L}_{demo}$$

**Components**:
- **Pixel Loss** (λ=100): L1 + L2 for exact pixel matching
- **SSIM Loss** (λ=5): Structural similarity preservation
- **Temporal Loss** (λ=10): Frame-to-frame consistency for smooth motion
- **Adversarial Loss** (λ=1): LSGAN - encourages realistic generation
- **Demographic Loss** (λ=5): Ensures correct demographic encoding

### 5.2 Discriminator Loss

$$\mathcal{L}_D = \frac{1}{2}[\mathbb{E}[(D(V_{real}) - 1)^2] + \mathbb{E}[(D(G(V, D)) - 0)^2]] + \lambda_{GP} \mathcal{L}_{GP}$$

- **Adversarial Loss**: Distinguish real vs fake videos
- **Gradient Penalty** (λ=10): WGAN-GP for training stability

---

## 6. Complete Flow

### 6.1 Use Case 1: Perfect Reconstruction

```
V_i [B,1,16,64,64] + D_i [B,11] (same as original)
    ↓
Video Embedding → H_V^0 [B,64,16,64,64]
Demographic Embedding → H_D^0 [B,128]
    ↓
Generator Core (U-Net Encoder-Decoder with Fusion)
    ↓
Synthetic Video: V̂_i [B,1,16,64,64]
```

**Result**: 7,791 real → 7,791 synthetic copies

### 6.2 Use Case 2: Demographic Variations

**Three forward passes with different demographics**:
1. Age Variation: $G(V_i, D_{age})$ where $D_{age} = (S_i, A_{new}, B_i)$
2. Sex Variation: $G(V_i, D_{sex})$ where $D_{sex} = (S_{new}, A_i, B_i)$
3. BMI Variation: $G(V_i, D_{bmi})$ where $D_{bmi} = (S_i, A_i, B_{new})$

**Result**: 7,791 real → 23,373 synthetic videos (3 per original)

---


---

## 7. Data Flow Summary

**Generator Forward Pass**:
```
Video [B,1,16,64,64] → Encoder (L1→L4) + Demographics Fusion
    ↓
Bottleneck + Fusion → Decoder (L4→L1) + Skip Connections
    ↓
Synthetic Video [B,1,16,64,64] → GAP → FC Layers → Output
```

**Demographic Flow**:
```
Demographics [B,11] → Embedding [B,128] → Fusion at Encoder L1, L2, L3, Bottleneck
```

---

## 8. Model Specifications

| Component | Specification |
|-----------|--------------|
| **Architecture** | U-Net 3D GAN (Encoder-Decoder) |
| **Input Video** | 16 frames × 64×64, grayscale |
| **Input Demographics** | 11-dim one-hot (Sex=2, Age=5, BMI=4) |
| **Generator** | ~15M parameters |
| **Discriminator** | ~5M parameters |
| **Training** | 200 epochs, batch size 4, LR 1e-4 |
| **Loss Weights** | Pixel:100, SSIM:5, Temporal:10, GAN:1, Demo:5 |
| **Device** | CUDA (GPU), Mixed Precision (FP16) |

---

## 9. GradCAM Analysis

**Purpose**: Validate that synthetic videos preserve cardiac motion patterns and focus on clinically relevant regions.

**Methodology**:
- Applied to trained generator model
- Computed gradients with respect to input video
- Generated attention maps showing regions of focus

**Results for Demographic Variations** (150 samples):
- **Cosine Similarity**: 0.8781 ± 0.1139 (target: >0.75) ✅
- **Spatial Correlation**: 0.9204 ± 0.0933 (target: >0.70) ✅
- **Samples above thresholds**: 86.7% (cosine), 95.3% (spatial)

**Key Findings**:
- High attention similarity (0.88) confirms preserved cardiac motion
- Model focuses on same cardiac regions (ventricles, valves) in originals and synthetics
- Attention patterns preserved regardless of demographic changes

---

## 10. Key Design Choices

1. **Spatial-only Downsampling**: Preserves temporal dimension (16 frames) throughout
2. **U-Net Skip Connections**: Preserve fine-grained details for reconstruction
3. **Multi-level Demographic Fusion**: Conditions generation at Encoder L1, L2, L3, and Bottleneck
4. **SE Attention**: Channel-wise feature recalibration in residual blocks
5. **PatchGAN Discriminator**: Operates on video patches for training stability
6. **Spectral Normalization**: Applied to discriminator for stable training
