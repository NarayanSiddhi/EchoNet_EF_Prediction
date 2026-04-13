# Actual GAN Architectures Implemented

## Summary

This document clarifies the **actual GAN architectures** implemented in this project, distinguishing them from code naming conventions.

---

## Important Note on Naming

**"C3DGAN" is NOT an architecture type** - it is only a code/file naming convention used in:
- Directory names (`c3dgan/`, `perfect_reconstruction_c3dgan/`)
- Class names (`ConditionalC3DGeneratorImproved`)
- Checkpoint files (`c3dgan_best.pt`)

**The actual architectures implemented are different.**

---

## Architecture 1: Conditional 3D DCGAN (Use Case 1)

### What It Actually Is
**Conditional 3D DCGAN** (Conditional Deep Convolutional 3D Generative Adversarial Network)

### Base Architecture
- **Type**: DCGAN (Deep Convolutional GAN) extended to 3D
- **Generator Style**: Progressive 3D transposed convolutions (DCGAN-style, but 3D)
- **Discriminator Style**: 3D convolutions with global average pooling
- **Conditioning**: Class label conditioning (conditional GAN)
- **Loss Function**: Binary Cross-Entropy (BCE)

### Implementation Details
- **Generator**: `ConditionalC3DGeneratorImproved`
  - Input: Random noise [B, 100] + Class label [B]
  - Architecture: Label embedding → Concatenate → FC → Reshape → 5× ConvTranspose3d layers
  - Output: [B, 1, 96, 128, 128]
  - Activation: Tanh

- **Discriminator**: `ConditionalC3DDiscriminatorImproved`
  - Input: Video [B, 1, 96, 128, 128] + Class label [B]
  - Architecture: 4× Conv3d layers → Global Average Pooling → FC
  - Output: Real/Fake logits [B, 1]

### Purpose
Generate synthetic videos from **random noise** to balance underrepresented demographic groups.

---

## Architecture 2: Conditional 3D U-Net GAN (Use Cases 2 & 3)

### What It Actually Is
**Conditional 3D U-Net GAN** (Conditional 3D U-Net Generative Adversarial Network)

### Base Architecture
- **Type**: U-Net GAN (similar to pix2pix but 3D)
- **Generator Style**: U-Net encoder-decoder with skip connections
- **Discriminator Style**: PatchGAN (patch-level discrimination)
- **Conditioning**: Demographic conditioning (conditional GAN)
- **Loss Function**: Multi-component loss (5 terms: Pixel + SSIM + Temporal + LSGAN + Demographic)

### Implementation Details
- **Generator**: `PerfectReconstructionGenerator`
  - Input: Real video [B, 1, 16, 64, 64] + Demographics [B, 11]
  - Architecture: U-Net encoder-decoder
    - Encoder: 4 levels (64×64 → 32×32 → 16×16 → 8×8)
    - Bottleneck: 4× Residual blocks + Demographic fusion
    - Decoder: 4 levels (8×8 → 16×16 → 32×32 → 64×64) with skip connections
  - Output: [B, 1, 16, 64, 64]

- **Discriminator**: `PatchDiscriminator3D`
  - Input: Video [B, 1, 16, 64, 64] + Demographics [B, 11]
  - Architecture: 4× Conv3d layers → Patch-level real/fake classification
  - Output: Real/Fake patches [B, 1, ...] + Demographic prediction [B, 11]

### Purpose
- **Use Case 2**: Generate demographic variations (change age/sex/BMI while preserving cardiac motion)
- **Use Case 3**: Generate perfect reconstructions (same demographics, near-identical copies)

---

## Key Differences

| Aspect | Use Case 1 (Conditional 3D DCGAN) | Use Cases 2 & 3 (Conditional 3D U-Net GAN) |
|--------|-----------------------------------|---------------------------------------------|
| **Base Architecture** | DCGAN (3D) | U-Net GAN (3D, pix2pix-style) |
| **Generator Type** | Progressive transposed convolutions | Encoder-decoder with skip connections |
| **Discriminator Type** | Global pooling (single score) | PatchGAN (patch-level scores) |
| **Input** | Random noise + class label | Real video + demographics |
| **Output Size** | 96 frames × 128×128 | 16 frames × 64×64 |
| **Loss Function** | BCE (Binary Cross-Entropy) | 5-term loss (Pixel + SSIM + Temporal + LSGAN + Demographic) |
| **Purpose** | Generate from scratch | Reconstruct/edit existing videos |

---

## Why the Confusion?

The codebase uses "C3DGAN" as a naming convention, but:
1. **Use Case 1** uses a **DCGAN-style** architecture (not C3D)
2. **Use Cases 2 & 3** use a **U-Net-style** architecture (not C3D)

The term "C3D" in the code refers to "3D Convolutional operations" (not the C3D architecture from Tran et al. 2015).

---

## Correct Terminology for Papers

**Use these names:**
- ✅ **Conditional 3D DCGAN** (Use Case 1)
- ✅ **Conditional 3D U-Net GAN** (Use Cases 2 & 3)

**Do NOT use:**
- ❌ "C3DGAN" (this is just a code naming convention)
- ❌ "C3D-GAN" (not the actual architecture)
- ❌ "Perfect Reconstruction C3DGAN" (use "Conditional 3D U-Net GAN" instead)

---

## Code Class Names vs. Architecture Types

| Code Class Name | Actual Architecture Type |
|----------------|--------------------------|
| `ConditionalC3DGeneratorImproved` | Conditional 3D DCGAN Generator |
| `ConditionalC3DDiscriminatorImproved` | Conditional 3D DCGAN Discriminator |
| `PerfectReconstructionGenerator` | Conditional 3D U-Net GAN Generator |
| `PatchDiscriminator3D` | Conditional 3D U-Net GAN Discriminator (PatchGAN) |

---

## References

- **DCGAN**: Radford et al. (2015) - "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
- **U-Net**: Ronneberger et al. (2015) - "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- **pix2pix**: Isola et al. (2017) - "Image-to-Image Translation with Conditional Adversarial Networks"
- **PatchGAN**: Used in pix2pix for patch-level discrimination
