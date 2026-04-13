"""
Shared C3D DCGAN Generator and Discriminator (conditional on demographics).

Used by train_c3dgan.py and generate_videos.py so checkpoint shapes always match.
Output temporal length is fixed at T=32 (four stride-2 temporal upsamples from 4).
Spatial size must be one of: 32, 64, 128, 256.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_VALID_SIZES = (32, 64, 128, 256)


def _check_size(size: int) -> None:
    if size not in _VALID_SIZES:
        raise ValueError(f"size must be one of {_VALID_SIZES}, got {size}")


class Generator(nn.Module):
    """3D generator: [B, z_dim] + [B, cond_dim] -> [B, 1, 32, H, W]."""

    def __init__(self, z_dim: int = 128, cond_dim: int = 11, size: int = 64):
        super().__init__()
        _check_size(size)
        self.size = size

        self.fc = nn.Linear(z_dim + cond_dim, 512 * 4 * 4 * 4)

        layers: list[nn.Module] = []

        layers.extend(
            [
                nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(256),
                nn.ReLU(True),
            ]
        )
        layers.extend(
            [
                nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(128),
                nn.ReLU(True),
            ]
        )
        layers.extend(
            [
                nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(True),
            ]
        )

        if size == 64:
            layers.extend(
                [
                    nn.ConvTranspose3d(64, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                    nn.BatchNorm3d(32),
                    nn.ReLU(True),
                    nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1),
                    nn.Tanh(),
                ]
            )
        elif size == 128:
            layers.extend(
                [
                    nn.ConvTranspose3d(64, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                    nn.BatchNorm3d(32),
                    nn.ReLU(True),
                ]
            )
            layers.extend(
                [
                    nn.ConvTranspose3d(32, 16, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                    nn.BatchNorm3d(16),
                    nn.ReLU(True),
                    nn.Conv3d(16, 1, kernel_size=3, stride=1, padding=1),
                    nn.Tanh(),
                ]
            )
        elif size == 256:
            layers.extend(
                [
                    nn.ConvTranspose3d(64, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                    nn.BatchNorm3d(32),
                    nn.ReLU(True),
                ]
            )
            layers.extend(
                [
                    nn.ConvTranspose3d(32, 16, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                    nn.BatchNorm3d(16),
                    nn.ReLU(True),
                ]
            )
            layers.extend(
                [
                    nn.ConvTranspose3d(16, 8, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                    nn.BatchNorm3d(8),
                    nn.ReLU(True),
                    nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1),
                    nn.Tanh(),
                ]
            )
        else:  # size == 32
            layers.extend(
                [
                    nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1),
                    nn.Tanh(),
                ]
            )

        self.main = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, cond], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4, 4)
        return self.main(x)


class Discriminator(nn.Module):
    """3D discriminator: [B, 1, 32, H, W] + cond -> logits [B, 1]."""

    def __init__(self, cond_dim: int = 11, size: int = 64):
        super().__init__()
        _check_size(size)
        self.size = size

        layers: list[nn.Module] = []

        if size == 256:
            layers.extend(
                [
                    nn.Conv3d(1, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            layers.extend(
                [
                    nn.Conv3d(32, 64, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                    nn.BatchNorm3d(64),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            layers.extend(
                [
                    nn.Conv3d(64, 64, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                    nn.BatchNorm3d(64),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        elif size == 128:
            layers.extend(
                [
                    nn.Conv3d(1, 32, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            layers.extend(
                [
                    nn.Conv3d(32, 64, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                    nn.BatchNorm3d(64),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        elif size == 64:
            layers.extend(
                [
                    nn.Conv3d(1, 64, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        else:  # size == 32
            layers.extend(
                [
                    nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )

        layers.extend(
            [
                nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(128),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )
        layers.extend(
            [
                nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(256),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )
        layers.extend(
            [
                nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(512),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4 * 4 + cond_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = features.view(features.size(0), -1)
        x = torch.cat([features, cond], dim=1)
        return self.classifier(x)


def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
