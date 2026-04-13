# Complete GAN Architecture Overview

**Project**: EchoNet-Pediatric-BIGAN-AUGMENTATION  
**Purpose**: Detailed textual architecture overview for all three use cases

---

## Table of Contents

1. [Use Case 1: Conditional 3D DCGAN](#use-case-1-conditional-3d-dcgan)
2. [Use Case 2: Conditional 3D U-Net GAN (Demographic Variations)](#use-case-2-conditional-3d-u-net-gan-demographic-variations)
3. [Use Case 3: Conditional 3D U-Net GAN (Perfect Reconstruction)](#use-case-3-conditional-3d-u-net-gan-perfect-reconstruction)

---

## Use Case 1: Conditional 3D DCGAN

### Architecture Type
**Conditional 3D DCGAN** (Conditional Deep Convolutional 3D Generative Adversarial Network)

### Complete System Flow

```
═══════════════════════════════════════════════════════════════════════════
INPUT
═══════════════════════════════════════════════════════════════════════════

Random Noise Vector: [B, 100]
Class Label: [B] (demographic class index, ~20 classes)

═══════════════════════════════════════════════════════════════════════════
GENERATOR: ConditionalC3DGeneratorImproved
═══════════════════════════════════════════════════════════════════════════

INPUT PROCESSING:
├─ Random Noise: [B, 100]
├─ Class Label: [B]
│
├─ Label Embedding Layer:
│   └─ Embedding(n_classes=20, embedding_dim=100)
│      └─ Output: [B, 100]
│
└─ Concatenation:
   ├─ Noise: [B, 100]
   ├─ Label Embedding: [B, 100]
   └─ Concatenated: [B, 200]

FULLY CONNECTED LAYER:
└─ Linear(200 → ngf*8*6*6*6)
   └─ ngf = 128
   └─ Output: [B, 98304] = [B, 128*8*6*6*6]

RESHAPE:
└─ Reshape to: [B, 1024, 6, 6, 6]
   └─ (1024 = ngf*8 = 128*8)

PROGRESSIVE 3D TRANSPOSED CONVOLUTIONS (Upsampling):

Level 1: 6×6×6 → 12×12×12
├─ ConvTranspose3d(1024 → 512, kernel=4, stride=2, padding=1)
├─ BatchNorm3d(512)
└─ ReLU
   └─ Output: [B, 512, 12, 12, 12]

Level 2: 12×12×12 → 24×24×24
├─ ConvTranspose3d(512 → 256, kernel=4, stride=2, padding=1)
├─ BatchNorm3d(256)
└─ ReLU
   └─ Output: [B, 256, 24, 24, 24]

Level 3: 24×24×24 → 48×48×48
├─ ConvTranspose3d(256 → 128, kernel=4, stride=2, padding=1)
├─ BatchNorm3d(128)
└─ ReLU
   └─ Output: [B, 128, 48, 48, 48]

Level 4: 48×48×48 → 96×96×96
├─ ConvTranspose3d(128 → 64, kernel=4, stride=2, padding=1)
├─ BatchNorm3d(64)
└─ ReLU
   └─ Output: [B, 64, 96, 96, 96]

Level 5: 96×96×96 → 96×128×128 (Spatial Upsampling Only)
├─ ConvTranspose3d(64 → 32, kernel=(1,4,4), stride=(1,2,2), padding=(0,1,1))
│  └─ Note: Temporal dimension unchanged (96), spatial dimensions upsampled
├─ BatchNorm3d(32)
└─ ReLU
   └─ Output: [B, 32, 96, 128, 128]

FINAL OUTPUT LAYER:
├─ Conv3d(32 → 1, kernel=3, stride=1, padding=1)
└─ Tanh activation
   └─ Output: [B, 1, 96, 128, 128]

POST-PROCESSING:
└─ Scale from [-1, 1] to [0, 1]
   └─ (x + 1) / 2.0

═══════════════════════════════════════════════════════════════════════════
DISCRIMINATOR: ConditionalC3DDiscriminatorImproved
═══════════════════════════════════════════════════════════════════════════

INPUT:
├─ Video: [B, 1, 96, 128, 128]
└─ Class Label: [B]

LABEL EMBEDDING:
└─ Embedding(n_classes=20, embedding_dim=1024)
   └─ Output: [B, 1024] (ndf*8 = 128*8)

3D CONVOLUTIONS (Progressive Downsampling):

Level 1: 96×128×128 → 48×64×64
├─ Conv3d(1 → 128, kernel=4, stride=2, padding=1)
└─ LeakyReLU(0.2)
   └─ Output: [B, 128, 48, 64, 64]

Level 2: 48×64×64 → 24×32×32
├─ Conv3d(128 → 256, kernel=4, stride=2, padding=1)
├─ BatchNorm3d(256)
└─ LeakyReLU(0.2)
   └─ Output: [B, 256, 24, 32, 32]

Level 3: 24×32×32 → 12×16×16
├─ Conv3d(256 → 512, kernel=4, stride=2, padding=1)
├─ BatchNorm3d(512)
└─ LeakyReLU(0.2)
   └─ Output: [B, 512, 12, 16, 16]

Level 4: 12×16×16 → 6×8×8
├─ Conv3d(512 → 1024, kernel=4, stride=2, padding=1)
├─ BatchNorm3d(1024)
└─ LeakyReLU(0.2)
   └─ Output: [B, 1024, 6, 8, 8]

GLOBAL AVERAGE POOLING:
└─ AdaptiveAvgPool3d(1)
   └─ Output: [B, 1024]

CONDITIONAL FUSION:
├─ Video Features: [B, 1024]
├─ Label Embedding: [B, 1024]
└─ Concatenate: [B, 2048]

FULLY CONNECTED LAYER:
└─ Linear(2048 → 1)
   └─ Output: Real/Fake Logits [B, 1]
   └─ Note: No sigmoid (use BCEWithLogitsLoss)

═══════════════════════════════════════════════════════════════════════════
LOSS FUNCTION
═══════════════════════════════════════════════════════════════════════════

Generator Loss:
└─ Binary Cross-Entropy (BCE)
   └─ L_G = -log(D(G(z, c)))
   └─ Where: z = random noise, c = class label

Discriminator Loss:
└─ Binary Cross-Entropy (BCE)
   ├─ L_D_real = -log(D(x, c))
   └─ L_D_fake = -log(1 - D(G(z, c)))
   └─ L_D = (L_D_real + L_D_fake) / 2

═══════════════════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════════════════

Synthetic Video: [B, 1, 96, 128, 128]
- 96 frames (temporal)
- 128×128 pixels (spatial)
- Grayscale (1 channel)
- Normalized to [0, 1] range

═══════════════════════════════════════════════════════════════════════════
KEY PARAMETERS
═══════════════════════════════════════════════════════════════════════════

- Noise dimension (nz): 100
- Generator filters (ngf): 128
- Discriminator filters (ndf): 128
- Number of classes (n_classes): ~20
- Output channels (nc): 1 (grayscale)
- Video length: 96 frames
- Video size: 128×128 pixels
- Total Generator Parameters: ~20M
- Total Discriminator Parameters: ~5M

═══════════════════════════════════════════════════════════════════════════
```

---

## Use Case 2: Conditional 3D U-Net GAN (Demographic Variations)

### Architecture Type
**Conditional 3D U-Net GAN** (pix2pix-style, 3D version)

### Complete System Flow

```
═══════════════════════════════════════════════════════════════════════════
INPUT
═══════════════════════════════════════════════════════════════════════════

Real Video: [B, 1, 16, 64, 64]
Demographics: [B, 11]
  └─ Format: [sex (2-dim), age_bins (5-dim), bmi_categories (4-dim)]
  └─ One-hot encoded vector

═══════════════════════════════════════════════════════════════════════════
GENERATOR: PerfectReconstructionGenerator
═══════════════════════════════════════════════════════════════════════════

INPUT PROCESSING:

Demographic Embedding:
├─ Input: Demographics [B, 11]
│
├─ Layer 1:
│   ├─ Linear(11 → 64)
│   ├─ LayerNorm(64)
│   ├─ ReLU
│   └─ Dropout(0.1)
│
├─ Layer 2:
│   ├─ Linear(64 → 128)
│   ├─ LayerNorm(128)
│   └─ ReLU
│
└─ Output: Demographic Embedding [B, 128]

═══════════════════════════════════════════════════════════════════════════
ENCODER (4 Levels - Spatial Downsampling)
═══════════════════════════════════════════════════════════════════════════

ENCODER LEVEL 1: 64×64 (Spatial Resolution)
├─ Input: Video [B, 1, 16, 64, 64]
│
├─ Initial Convolution:
│   ├─ Conv3d(1 → 64, kernel=7, padding=3)
│   ├─ BatchNorm3d(64)
│   └─ ReLU
│
├─ Residual Block 1:
│   ├─ Conv3d(64 → 64, kernel=3, padding=1) + BatchNorm + ReLU
│   ├─ Conv3d(64 → 64, kernel=3, padding=1) + BatchNorm
│   ├─ SE Attention:
│   │   ├─ AdaptiveAvgPool3d(1) → [B, 64, 1, 1, 1]
│   │   ├─ Conv3d(64 → 4, kernel=1) + ReLU
│   │   ├─ Conv3d(4 → 64, kernel=1) + Sigmoid
│   │   └─ Element-wise multiply
│   └─ Residual: output = SE_attention(conv_out) + input
│
├─ Residual Block 2:
│   └─ (Same as Residual Block 1)
│
└─ Spatial Demographic Fusion:
    ├─ Input: Features [B, 64, 16, 64, 64]
    ├─ Input: Demo Embedding [B, 128]
    │
    ├─ Projection:
    │   └─ Linear(128 → 64)
    │      └─ Output: [B, 64]
    │
    ├─ Spatial Expansion:
    │   └─ View & Expand: [B, 64, 1, 1, 1] → [B, 64, 16, 64, 64]
    │
    ├─ Concatenation:
    │   ├─ Features: [B, 64, 16, 64, 64]
    │   ├─ Demo Spatial: [B, 64, 16, 64, 64]
    │   └─ Concatenated: [B, 128, 16, 64, 64]
    │
    └─ Fusion:
        ├─ Conv3d(128 → 64, kernel=1)
        ├─ BatchNorm3d(64)
        └─ ReLU
           └─ Output: [B, 64, 16, 64, 64]

ENCODER LEVEL 2: 32×32 (Spatial Resolution)
├─ Input: [B, 64, 16, 64, 64]
│
├─ Downsampling:
│   ├─ Conv3d(64 → 128, kernel=(1,4,4), stride=(1,2,2), padding=(0,1,1))
│   │  └─ Note: Spatial downsampling only (64→32), temporal unchanged (16)
│   ├─ BatchNorm3d(128)
│   └─ ReLU
│
├─ Residual Block 1: (128 channels)
├─ Residual Block 2: (128 channels)
│
└─ Spatial Demographic Fusion:
    └─ (Same structure as Level 1, but with 128 channels)
       └─ Output: [B, 128, 16, 32, 32]

ENCODER LEVEL 3: 16×16 (Spatial Resolution)
├─ Input: [B, 128, 16, 32, 32]
│
├─ Downsampling:
│   ├─ Conv3d(128 → 256, kernel=(1,4,4), stride=(1,2,2), padding=(0,1,1))
│   ├─ BatchNorm3d(256)
│   └─ ReLU
│
├─ Residual Block 1: (256 channels)
├─ Residual Block 2: (256 channels)
│
└─ Spatial Demographic Fusion:
    └─ (Same structure, 256 channels)
       └─ Output: [B, 256, 16, 16, 16]

ENCODER LEVEL 4: 8×8 (Spatial Resolution)
├─ Input: [B, 256, 16, 16, 16]
│
├─ Downsampling:
│   ├─ Conv3d(256 → 512, kernel=(1,4,4), stride=(1,2,2), padding=(0,1,1))
│   ├─ BatchNorm3d(512)
│   └─ ReLU
│
├─ Residual Block 1: (512 channels)
└─ Residual Block 2: (512 channels)
   └─ Output: [B, 512, 16, 8, 8]
   └─ Note: No demographic fusion at Level 4

═══════════════════════════════════════════════════════════════════════════
BOTTLENECK
═══════════════════════════════════════════════════════════════════════════

├─ Input: [B, 512, 16, 8, 8]
│
├─ Residual Block 1: (512 channels)
├─ Residual Block 2: (512 channels)
├─ Residual Block 3: (512 channels)
├─ Residual Block 4: (512 channels)
│
└─ Spatial Demographic Fusion:
    └─ (512 channels)
       └─ Output: [B, 512, 16, 8, 8]

═══════════════════════════════════════════════════════════════════════════
DECODER (4 Levels - Spatial Upsampling with Skip Connections)
═══════════════════════════════════════════════════════════════════════════

DECODER LEVEL 4→3: 8×8 → 16×16
├─ Input: Bottleneck [B, 512, 16, 8, 8]
│
├─ Upsampling:
│   ├─ ConvTranspose3d(512 → 256, kernel=(1,4,4), stride=(1,2,2), padding=(0,1,1))
│   ├─ BatchNorm3d(256)
│   └─ ReLU
│      └─ Output: [B, 256, 16, 16, 16]
│
├─ Residual Block: (256 channels)
│
└─ Skip Connection:
    ├─ Decoder Features: [B, 256, 16, 16, 16]
    ├─ Encoder Level 3 Features: [B, 256, 16, 16, 16]
    └─ Concatenate: [B, 512, 16, 16, 16]

DECODER LEVEL 3→2: 16×16 → 32×32
├─ Input: [B, 512, 16, 16, 16]
│
├─ Upsampling:
│   ├─ ConvTranspose3d(512 → 128, kernel=(1,4,4), stride=(1,2,2), padding=(0,1,1))
│   ├─ BatchNorm3d(128)
│   └─ ReLU
│      └─ Output: [B, 128, 16, 32, 32]
│
├─ Residual Block: (128 channels)
│
└─ Skip Connection:
    ├─ Decoder Features: [B, 128, 16, 32, 32]
    ├─ Encoder Level 2 Features: [B, 128, 16, 32, 32]
    └─ Concatenate: [B, 256, 16, 32, 32]

DECODER LEVEL 2→1: 32×32 → 64×64
├─ Input: [B, 256, 16, 32, 32]
│
├─ Upsampling:
│   ├─ ConvTranspose3d(256 → 64, kernel=(1,4,4), stride=(1,2,2), padding=(0,1,1))
│   ├─ BatchNorm3d(64)
│   └─ ReLU
│      └─ Output: [B, 64, 16, 64, 64]
│
├─ Residual Block: (64 channels)
│
└─ Skip Connection:
    ├─ Decoder Features: [B, 64, 16, 64, 64]
    ├─ Encoder Level 1 Features: [B, 64, 16, 64, 64]
    └─ Concatenate: [B, 128, 16, 64, 64]

DECODER LEVEL 1 (Final Processing):
├─ Input: [B, 128, 16, 64, 64]
│
├─ Convolution:
│   ├─ Conv3d(128 → 64, kernel=3, padding=1)
│   ├─ BatchNorm3d(64)
│   └─ ReLU
│
├─ Residual Block 1: (64 channels)
├─ Residual Block 2: (64 channels)
│
└─ Output: [B, 64, 16, 64, 64]

═══════════════════════════════════════════════════════════════════════════
OUTPUT LAYER
═══════════════════════════════════════════════════════════════════════════

├─ Input: [B, 64, 16, 64, 64]
│
├─ Layer 1:
│   ├─ Conv3d(64 → 32, kernel=3, padding=1)
│   ├─ BatchNorm3d(32)
│   └─ ReLU
│      └─ Output: [B, 32, 16, 64, 64]
│
└─ Layer 2 (Final):
    ├─ Conv3d(32 → 1, kernel=7, padding=3)
    └─ Tanh activation
       └─ Output: [B, 1, 16, 64, 64]

═══════════════════════════════════════════════════════════════════════════
DISCRIMINATOR: PatchDiscriminator3D
═══════════════════════════════════════════════════════════════════════════

INPUT:
├─ Video: [B, 1, 16, 64, 64]
└─ Demographics: [B, 11] (used for auxiliary classification)

3D CONVOLUTIONS (Patch-level Discrimination):

Level 1: 64×64 → 32×32
├─ Conv3d(1 → 64, kernel=4, stride=2, padding=1)
└─ LeakyReLU(0.2)
   └─ Output: [B, 64, 16, 32, 32]

Level 2: 32×32 → 16×16
├─ Conv3d(64 → 128, kernel=4, stride=2, padding=1)
├─ BatchNorm3d(128)
└─ LeakyReLU(0.2)
   └─ Output: [B, 128, 16, 16, 16]

Level 3: 16×16 → 8×8
├─ Conv3d(128 → 256, kernel=4, stride=2, padding=1)
├─ BatchNorm3d(256)
└─ LeakyReLU(0.2)
   └─ Output: [B, 256, 16, 8, 8]

Level 4: 8×8 → 4×4
├─ Conv3d(256 → 512, kernel=4, stride=2, padding=1)
├─ BatchNorm3d(512)
└─ LeakyReLU(0.2)
   └─ Output: [B, 512, 16, 4, 4]

DROPOUT:
└─ Dropout3d(0.3)
   └─ Output: [B, 512, 16, 4, 4]

REAL/FAKE CLASSIFIER (Patch-level):
└─ Conv3d(512 → 1, kernel=4, padding=1)
   └─ Output: Real/Fake Patches [B, 1, 16, 4, 4]
   └─ Note: Patch-level output (not single value)

DEMOGRAPHIC CLASSIFIER (Auxiliary):
├─ Input: [B, 512, 16, 4, 4]
│
├─ Global Average Pooling:
│   └─ AdaptiveAvgPool3d(1)
│      └─ Output: [B, 512]
│
├─ Fully Connected Layer 1:
│   └─ Linear(512 → 256)
│      └─ Output: [B, 256]
│
└─ Fully Connected Layer 2:
    └─ Linear(256 → 11)
       └─ Output: Demographic Prediction [B, 11]

═══════════════════════════════════════════════════════════════════════════
LOSS FUNCTION (5-Term Loss)
═══════════════════════════════════════════════════════════════════════════

Generator Loss:
L_G = λ_pixel·L_pixel + λ_SSIM·L_SSIM + λ_temporal·L_temporal + λ_GAN·L_GAN + λ_demo·L_demo

1. Pixel Reconstruction Loss (λ_pixel = 100.0):
   L_pixel = ||G(V, D) - V||_1 + 0.5·||G(V, D) - V||_2^2
   └─ Combines L1 and L2 losses

2. SSIM Loss (λ_SSIM = 5.0):
   L_SSIM = 1 - SSIM(G(V, D), V)
   └─ Preserves structural similarity

3. Temporal Consistency Loss (λ_temporal = 10.0):
   L_temporal = Σ_{t=1}^{T-1} ||(G(V,D)_{t+1} - G(V,D)_t) - (V_{t+1} - V_t)||_2^2
   └─ Preserves frame-to-frame differences

4. Adversarial Loss - LSGAN (λ_GAN = 1.0):
   L_GAN = E[(D(G(V, D)) - 1)^2]
   └─ Uses Mean Squared Error (MSE) for stability

5. Demographic Preservation Loss (λ_demo = 5.0):
   L_demo = BCE(D_demo(G(V, D)), D)
   └─ Binary Cross-Entropy loss
   └─ Ensures demographic features are correctly encoded

Discriminator Loss:
L_D = 0.5·[E[(D(V) - 1)^2] + E[(D(G(V, D)) - 0)^2]] + λ_demo·L_demo

═══════════════════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════════════════

Synthetic Video: [B, 1, 16, 64, 64]
- 16 frames (temporal)
- 64×64 pixels (spatial)
- Grayscale (1 channel)
- Normalized to [-1, 1] range (Tanh output)

USE CASE 2 SPECIFIC:
- Demographics are CHANGED (age/sex/BMI variations)
- Output: Video with modified demographics but preserved cardiac motion

═══════════════════════════════════════════════════════════════════════════
KEY PARAMETERS
═══════════════════════════════════════════════════════════════════════════

- Base channels: 64 (scales to 128, 256, 512)
- Demographic embedding dimension: 128
- Video length: 16 frames
- Video size: 64×64 pixels
- Total Generator Parameters: ~15M
- Total Discriminator Parameters: ~5M

═══════════════════════════════════════════════════════════════════════════
```

---

## Use Case 3: Conditional 3D U-Net GAN (Perfect Reconstruction)

### Architecture Type
**Conditional 3D U-Net GAN** (pix2pix-style, 3D version)

### Complete System Flow

```
═══════════════════════════════════════════════════════════════════════════
INPUT
═══════════════════════════════════════════════════════════════════════════

Real Video: [B, 1, 16, 64, 64]
Demographics: [B, 11]
  └─ Format: [sex (2-dim), age_bins (5-dim), bmi_categories (4-dim)]
  └─ One-hot encoded vector
  └─ ⚠️ SAME demographics as original video (not changed)

═══════════════════════════════════════════════════════════════════════════
GENERATOR: PerfectReconstructionGenerator
═══════════════════════════════════════════════════════════════════════════

[IDENTICAL TO USE CASE 2 - Same Architecture]

See Use Case 2 for complete generator architecture details.

Key Difference:
- Demographics input is SAME as original video (not altered)
- All other components identical to Use Case 2

═══════════════════════════════════════════════════════════════════════════
DISCRIMINATOR: PatchDiscriminator3D
═══════════════════════════════════════════════════════════════════════════

[IDENTICAL TO USE CASE 2 - Same Architecture]

See Use Case 2 for complete discriminator architecture details.

═══════════════════════════════════════════════════════════════════════════
LOSS FUNCTION (5-Term Loss)
═══════════════════════════════════════════════════════════════════════════

[IDENTICAL TO USE CASE 2 - Same Loss Function]

See Use Case 2 for complete loss function details.

═══════════════════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════════════════

Synthetic Video: [B, 1, 16, 64, 64]
- 16 frames (temporal)
- 64×64 pixels (spatial)
- Grayscale (1 channel)
- Normalized to [-1, 1] range (Tanh output)

USE CASE 3 SPECIFIC:
- Demographics are SAME as original video
- Output: Near-perfect copy of original video
- Metrics: SSIM > 0.99, PSNR > 48 dB, MSE < 1.0

═══════════════════════════════════════════════════════════════════════════
KEY PARAMETERS
═══════════════════════════════════════════════════════════════════════════

[IDENTICAL TO USE CASE 2]

- Base channels: 64 (scales to 128, 256, 512)
- Demographic embedding dimension: 128
- Video length: 16 frames
- Video size: 64×64 pixels
- Total Generator Parameters: ~15M
- Total Discriminator Parameters: ~5M

═══════════════════════════════════════════════════════════════════════════
```

---

## Key Component Details

### ResidualBlock3D (Used in Use Cases 2 & 3)

```
INPUT: [B, C, T, H, W]

├─ Conv3d(C → C, kernel=3, padding=1)
├─ BatchNorm3d(C)
└─ ReLU
   └─ Intermediate: [B, C, T, H, W]

├─ Conv3d(C → C, kernel=3, padding=1)
└─ BatchNorm3d(C)
   └─ Conv Output: [B, C, T, H, W]

SE ATTENTION:
├─ AdaptiveAvgPool3d(1)
│  └─ Output: [B, C, 1, 1, 1]
│
├─ Conv3d(C → C//16, kernel=1)
├─ ReLU
├─ Conv3d(C//16 → C, kernel=1)
└─ Sigmoid
   └─ SE Weights: [B, C, 1, 1, 1]

ELEMENT-WISE MULTIPLICATION:
├─ Conv Output: [B, C, T, H, W]
├─ SE Weights: [B, C, 1, 1, 1] (broadcasted)
└─ Weighted: [B, C, T, H, W]

RESIDUAL CONNECTION:
├─ Weighted Output: [B, C, T, H, W]
├─ Input: [B, C, T, H, W]
└─ Add: [B, C, T, H, W]

└─ ReLU
   └─ OUTPUT: [B, C, T, H, W]
```

### SpatialDemographicFusion (Used in Use Cases 2 & 3)

```
INPUT:
├─ Features: [B, C, T, H, W]
└─ Demo Embedding: [B, 128]

├─ Linear Projection:
│   └─ Linear(128 → C)
│      └─ Output: [B, C]
│
├─ Reshape & Expand:
│   ├─ View: [B, C] → [B, C, 1, 1, 1]
│   └─ Expand: [B, C, 1, 1, 1] → [B, C, T, H, W]
│      └─ Demo Spatial: [B, C, T, H, W]
│
├─ Concatenation:
│   ├─ Features: [B, C, T, H, W]
│   ├─ Demo Spatial: [B, C, T, H, W]
│   └─ Concatenated: [B, 2C, T, H, W]
│
└─ Fusion:
    ├─ Conv3d(2C → C, kernel=1)
    ├─ BatchNorm3d(C)
    └─ ReLU
       └─ OUTPUT: [B, C, T, H, W]
```

---

## Architecture Comparison Summary

| Component | Use Case 1 | Use Cases 2 & 3 |
|-----------|------------|-----------------|
| **Architecture Type** | Conditional 3D DCGAN | Conditional 3D U-Net GAN |
| **Generator Style** | Progressive Transposed Convolutions | U-Net Encoder-Decoder |
| **Input** | Random Noise + Class Label | Real Video + Demographics |
| **Output Size** | 96×128×128 | 16×64×64 |
| **Skip Connections** | No | Yes (U-Net style) |
| **Residual Blocks** | No | Yes (with SE Attention) |
| **Demographic Fusion** | No | Yes (Spatial Fusion) |
| **Discriminator** | Global Average Pooling | PatchGAN |
| **Loss Function** | BCE | 5-Term Loss |
| **Conditioning** | Class Label | Demographics Vector |

---

**Document Generated**: February 2026  
**Project**: EchoNet-Pediatric-BIGAN-AUGMENTATION  
**Purpose**: Complete textual architecture overview for diagram creation
