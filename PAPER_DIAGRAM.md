# Paper Diagram: Unified GAN Architecture Overview

## Single Unified Diagram (All Use Cases)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ INPUT                                                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│ Use Case 1:                                                              │
│   • Random Noise Vector: [B, 100]                                       │
│   • Class Label: [B] (demographic class index, ~20 classes)             │
│                                                                          │
│ Use Cases 2 & 3:                                                         │
│   • Real Video: [B, 1, 16, 64, 64]                                      │
│   • Demographics: [B, 11] (sex: 2, age: 5, BMI: 4)                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ GAN ARCHITECTURE                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│ ┌──────────────────────────────────────────────────────────────────┐   │
│ │ Use Case 1: Conditional 3D DCGAN                                  │   │
│ ├──────────────────────────────────────────────────────────────────┤   │
│ │ GENERATOR:                                                        │   │
│ │   Label Embedding → Concatenate → FC → Reshape [1024,6,6,6]     │   │
│ │   → 5× ConvTranspose3d: 6×6×6 → 12×12×12 → 24×24×24 →            │   │
│ │     48×48×48 → 96×96×96 → 96×128×128 → Tanh                      │   │
│ │                                                                   │   │
│ │ DISCRIMINATOR:                                                    │   │
│ │   4× Conv3d: 96×128×128 → 48×64×64 → 24×32×32 →                  │   │
│ │     12×16×16 → 6×8×8 → Global Pooling → FC                       │   │
│ │                                                                   │   │
│ │ Loss: BCE (Binary Cross-Entropy)                                 │   │
│ └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│ ┌──────────────────────────────────────────────────────────────────┐   │
│ │ Use Cases 2 & 3: Conditional 3D U-Net GAN                          │   │
│ ├──────────────────────────────────────────────────────────────────┤   │
│ │ GENERATOR (U-Net Encoder-Decoder):                                │   │
│ │   ENCODER: 64×64 → 32×32 → 16×16 → 8×8                            │   │
│ │     Channels: 64 → 128 → 256 → 512                                │   │
│ │     + Residual Blocks + Demographic Fusion                       │   │
│ │                                                                   │   │
│ │   BOTTLENECK: 4× Residual Blocks (512) + Demographic Fusion       │   │
│ │                                                                   │   │
│ │   DECODER: 8×8 → 16×16 → 32×32 → 64×64                            │   │
│ │     Channels: 512 → 256 → 128 → 64                                │   │
│ │     + Skip Connections + Residual Blocks                         │   │
│ │                                                                   │   │
│ │ DISCRIMINATOR (PatchGAN):                                        │   │
│ │   4× Conv3d: 64×64 → 32×32 → 16×16 → 8×8 → 4×4                   │   │
│ │   → Real/Fake Patches + Demographic Classifier                   │   │
│ │                                                                   │   │
│ │ Loss: 5-Term (Pixel + SSIM + Temporal + LSGAN + Demographic)     │   │
│ └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ OUTPUT                                                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│ Use Case 1:                                                              │
│   • Synthetic Video: [B, 1, 96, 128, 128]                               │
│   • Generated from random noise, conditioned on class label              │
│   • ~5,000 videos for dataset balancing                                │
│                                                                          │
│ Use Case 2:                                                              │
│   • Synthetic Video: [B, 1, 16, 64, 64]                                 │
│   • Demographic variations (age/sex/BMI changed)                       │
│   • 23,373 variations (3 per original video)                           │
│                                                                          │
│ Use Case 3:                                                              │
│   • Synthetic Video: [B, 1, 16, 64, 64]                                 │
│   • Perfect reconstruction (same demographics)                          │
│   • 7,791 perfect copies                                                │
│   • Quality: SSIM > 0.99, PSNR > 48 dB, MSE < 1.0                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ GRAD-CAM VALIDATION                                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│ Attention Pattern Analysis (Use Case 2):                                │
│   • Cosine Similarity: 0.8781 ± 0.1139                                  │
│     └─ 86.7% of samples > 0.75 threshold                                │
│   • Spatial Correlation: 0.9204 ± 0.0933                                │
│     └─ 95.3% of samples > 0.70 threshold                                │
│                                                                          │
│ Validation Results:                                                      │
│   • Preserves cardiac motion patterns                                   │
│   • Maintains diagnostic information                                    │
│   • Confirms attention on ventricles, atria, valves                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

**Note**: This is a single unified diagram showing all three use cases in one flow. The GAN Architecture box contains both architectures side-by-side, and the Output box shows results for all three use cases.
