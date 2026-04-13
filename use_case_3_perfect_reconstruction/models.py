"""
Perfect Reconstruction C3D-GAN Generator Architecture

Used for Use Case 3 (reconstruction) and Use Case 2 (demographic translation).

spatial_size:
  - 64: four spatial downsamples (legacy checkpoints)
  - 128: five downsamples for higher-resolution clips (requires matching video_size at train/infer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    """3D Residual Block for better gradient flow"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, max(channels // 16, 1), 1),
            nn.ReLU(),
            nn.Conv3d(max(channels // 16, 1), channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        se_weight = self.se(out)
        out = out * se_weight
        out = out + residual
        return F.relu(out)


class DemographicEmbedding(nn.Module):
    """Embed demographics into latent space"""

    def __init__(self, demo_dim=11, embed_dim=128):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(demo_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

    def forward(self, demographics):
        return self.embedding(demographics)


class SpatialDemographicFusion(nn.Module):
    """Concatenate projected demographics with features and mix with 1x1x1 conv (legacy)."""

    def __init__(self, feature_channels, demo_embed_dim=128):
        super().__init__()
        self.demo_proj = nn.Linear(demo_embed_dim, feature_channels)
        self.fusion = nn.Sequential(
            nn.Conv3d(feature_channels * 2, feature_channels, kernel_size=1),
            nn.BatchNorm3d(feature_channels),
            nn.ReLU(),
        )

    def forward(self, features, demo_embed):
        B, C, T, H, W = features.shape
        demo_features = self.demo_proj(demo_embed)
        demo_spatial = demo_features.view(B, C, 1, 1, 1).expand(B, C, T, H, W)
        combined = torch.cat([features, demo_spatial], dim=1)
        return self.fusion(combined)


class FiLMDemographicConditioning(nn.Module):
    """Feature-wise Linear Modulation (FiLM): per-channel scale and shift from demographics."""

    def __init__(self, feature_channels, demo_embed_dim=128):
        super().__init__()
        self.to_gamma = nn.Linear(demo_embed_dim, feature_channels)
        self.to_beta = nn.Linear(demo_embed_dim, feature_channels)

    def forward(self, features, demo_embed):
        B, C, T, H, W = features.shape
        gamma = self.to_gamma(demo_embed).view(B, C, 1, 1, 1)
        beta = self.to_beta(demo_embed).view(B, C, 1, 1, 1)
        return (1.0 + torch.tanh(gamma)) * features + beta


def _demo_cond(
    conditioning: str,
    feature_channels: int,
    demo_embed_dim: int = 128,
) -> nn.Module:
    if conditioning == "concat":
        return SpatialDemographicFusion(feature_channels, demo_embed_dim)
    if conditioning == "film":
        return FiLMDemographicConditioning(feature_channels, demo_embed_dim)
    raise ValueError(f"conditioning must be 'concat' or 'film', got {conditioning!r}")


class PerfectReconstructionGenerator(nn.Module):
    """U-Net style 3D generator with demographic conditioning.

    Args:
        base_channels: width of first conv features
        spatial_size: 64 (default, original) or 128 (extra encoder/decoder level)
        conditioning: ``concat`` (concat + 1x1 conv, matches older checkpoints) or
            ``film`` (FiLM scale/shift from demographics — preferred for new training).
    """

    def __init__(self, base_channels=64, spatial_size=64, conditioning: str = "concat"):
        super().__init__()
        if spatial_size not in (64, 128):
            raise ValueError("spatial_size must be 64 or 128")
        if conditioning not in ("concat", "film"):
            raise ValueError("conditioning must be 'concat' or 'film'")
        self.spatial_size = spatial_size
        self.conditioning = conditioning
        bc = base_channels

        self.demo_embedding = DemographicEmbedding(demo_dim=11, embed_dim=128)

        self.enc1 = nn.Sequential(
            nn.Conv3d(1, bc, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm3d(bc),
            nn.ReLU(),
            ResidualBlock3D(bc),
            ResidualBlock3D(bc),
        )
        self.demo_fusion1 = _demo_cond(conditioning, bc, 128)

        self.enc2 = nn.Sequential(
            nn.Conv3d(bc, bc * 2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(bc * 2),
            nn.ReLU(),
            ResidualBlock3D(bc * 2),
            ResidualBlock3D(bc * 2),
        )
        self.demo_fusion2 = _demo_cond(conditioning, bc * 2, 128)

        self.enc3 = nn.Sequential(
            nn.Conv3d(bc * 2, bc * 4, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(bc * 4),
            nn.ReLU(),
            ResidualBlock3D(bc * 4),
            ResidualBlock3D(bc * 4),
        )
        self.demo_fusion3 = _demo_cond(conditioning, bc * 4, 128)

        self.enc4 = nn.Sequential(
            nn.Conv3d(bc * 4, bc * 8, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(bc * 8),
            nn.ReLU(),
            ResidualBlock3D(bc * 8),
            ResidualBlock3D(bc * 8),
        )

        if spatial_size == 128:
            self.enc5 = nn.Sequential(
                nn.Conv3d(bc * 8, bc * 8, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
                nn.BatchNorm3d(bc * 8),
                nn.ReLU(),
                ResidualBlock3D(bc * 8),
                ResidualBlock3D(bc * 8),
            )

        self.bottleneck = nn.Sequential(
            ResidualBlock3D(bc * 8),
            ResidualBlock3D(bc * 8),
            ResidualBlock3D(bc * 8),
            ResidualBlock3D(bc * 8),
        )
        self.demo_fusion_bottleneck = _demo_cond(conditioning, bc * 8, 128)

        if spatial_size == 64:
            self.dec4 = nn.Sequential(
                nn.ConvTranspose3d(bc * 8, bc * 4, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
                nn.BatchNorm3d(bc * 4),
                nn.ReLU(),
                ResidualBlock3D(bc * 4),
            )
            self.dec3 = nn.Sequential(
                nn.ConvTranspose3d(bc * 8, bc * 2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
                nn.BatchNorm3d(bc * 2),
                nn.ReLU(),
                ResidualBlock3D(bc * 2),
            )
            self.dec2 = nn.Sequential(
                nn.ConvTranspose3d(bc * 4, bc, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
                nn.BatchNorm3d(bc),
                nn.ReLU(),
                ResidualBlock3D(bc),
            )
            self.dec1 = nn.Sequential(
                nn.Conv3d(bc * 2, bc, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(bc),
                nn.ReLU(),
                ResidualBlock3D(bc),
                ResidualBlock3D(bc),
            )
        else:
            # 128-path decoder: channel counts after each skip concat match enc3/enc2/enc1
            self.dec5 = nn.Sequential(
                nn.ConvTranspose3d(bc * 8, bc * 4, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
                nn.BatchNorm3d(bc * 4),
                nn.ReLU(),
                ResidualBlock3D(bc * 4),
            )
            # dec5 out (bc*4) + e4 (bc*8) -> bc*12
            self.dec4 = nn.Sequential(
                nn.ConvTranspose3d(bc * 12, bc * 4, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
                nn.BatchNorm3d(bc * 4),
                nn.ReLU(),
                ResidualBlock3D(bc * 4),
            )
            # dec4 out (bc*4) + e3 (bc*4) -> bc*8
            self.dec3 = nn.Sequential(
                nn.ConvTranspose3d(bc * 8, bc * 2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
                nn.BatchNorm3d(bc * 2),
                nn.ReLU(),
                ResidualBlock3D(bc * 2),
            )
            # dec3 out (bc*2) + e2 (bc*2) -> bc*4
            self.dec2 = nn.Sequential(
                nn.ConvTranspose3d(bc * 4, bc, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
                nn.BatchNorm3d(bc),
                nn.ReLU(),
                ResidualBlock3D(bc),
            )
            # dec2 out (bc) + e1 (bc) -> bc*2
            self.dec1 = nn.Sequential(
                nn.Conv3d(bc * 2, bc, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(bc),
                nn.ReLU(),
                ResidualBlock3D(bc),
                ResidualBlock3D(bc),
            )

        self.output = nn.Sequential(
            nn.Conv3d(bc, bc // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(bc // 2),
            nn.ReLU(),
            nn.Conv3d(bc // 2, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, x, demographics):
        demo_embed = self.demo_embedding(demographics)

        e1 = self.demo_fusion1(self.enc1(x), demo_embed)
        e2 = self.demo_fusion2(self.enc2(e1), demo_embed)
        e3 = self.demo_fusion3(self.enc3(e2), demo_embed)
        e4 = self.enc4(e3)

        if self.spatial_size == 64:
            b_in = e4
        else:
            b_in = self.enc5(e4)

        b = self.demo_fusion_bottleneck(self.bottleneck(b_in), demo_embed)

        if self.spatial_size == 64:
            d4 = self.dec4(b)
            d4 = torch.cat([d4, e3], dim=1)
            d3 = self.dec3(d4)
            d3 = torch.cat([d3, e2], dim=1)
            d2 = self.dec2(d3)
            d2 = torch.cat([d2, e1], dim=1)
            d1 = self.dec1(d2)
            return self.output(d1)

        u5 = self.dec5(b)
        u5 = torch.cat([u5, e4], dim=1)
        u4 = self.dec4(u5)
        u4 = torch.cat([u4, e3], dim=1)
        u3 = self.dec3(u4)
        u3 = torch.cat([u3, e2], dim=1)
        u2 = self.dec2(u3)
        u2 = torch.cat([u2, e1], dim=1)
        u1 = self.dec1(u2)
        return self.output(u1)
