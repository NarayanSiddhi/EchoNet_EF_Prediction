"""
Improved C3DGAN Models - Removed AdaptiveAvgPool3d for Better Quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalC3DGeneratorImproved(nn.Module):
    """
    Improved Conditional C3D Generator WITHOUT AdaptiveAvgPool3d.
    Uses fixed-size transposed convolutions to get exact output dimensions.
    This significantly improves sharpness by avoiding blurring from adaptive pooling.
    """
    
    def __init__(self, nz=100, ngf=128, nc=1, n_classes=20, video_length=96, video_size=128):
        """
        Args:
            nz: Noise vector size
            ngf: Generator filters (increased capacity)
            nc: Output channels
            n_classes: Number of class combinations
            video_length: Temporal frames
            video_size: Spatial size
        """
        super(ConditionalC3DGeneratorImproved, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.n_classes = n_classes
        self.video_length = video_length
        self.video_size = video_size
        
        # Embedding for class labels
        self.label_emb = nn.Embedding(n_classes, nz)
        
        # Base generator with conditional input
        self.fc = nn.Linear(nz * 2, ngf * 8 * 6 * 6 * 6)  # *2: noise + label
        
        # 3D Transposed Convolutions for upsampling
        # 6x6x6 -> 12x12x12
        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(ngf * 8, ngf * 4, kernel_size=(4, 4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm3d(ngf * 4),
            nn.ReLU(True)
        )
        
        # 12x12x12 -> 24x24x24
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(ngf * 4, ngf * 2, kernel_size=(4, 4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True)
        )
        
        # 24x24x24 -> 48x48x48
        self.conv3 = nn.Sequential(
            nn.ConvTranspose3d(ngf * 2, ngf, kernel_size=(4, 4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm3d(ngf),
            nn.ReLU(True)
        )
        
        # 48x48x48 -> 96x96x96 (temporal matches video_length=96)
        self.conv4 = nn.Sequential(
            nn.ConvTranspose3d(ngf, ngf // 2, kernel_size=(4, 4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm3d(ngf // 2),
            nn.ReLU(True)
        )
        
        # Final layer: 96x96x96 -> 96x128x128 (spatial upsampling only)
        # Use asymmetric kernel to upsample spatial dimensions while keeping temporal
        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(ngf // 2, ngf // 4, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ngf // 4),
            nn.ReLU(True)
        )
        
        # Output layer: 96x128x128 -> 96x128x128 (no size change, just channel reduction)
        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(ngf // 4, nc, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        """
        Args:
            noise: Random noise (batch_size, nz)
            labels: Class labels (batch_size,)
        Returns:
            Generated videos (batch_size, nc, video_length, video_size, video_size)
        """
        # Embed labels
        label_emb = self.label_emb(labels)  # (batch_size, nz)
        
        # Concatenate noise and label embedding
        input_vec = torch.cat([noise, label_emb], dim=1)  # (batch_size, nz * 2)
        
        # Generate video
        x = self.fc(input_vec)
        x = x.view(-1, self.ngf * 8, 6, 6, 6)
        
        x = self.conv1(x)  # 6->12
        x = self.conv2(x)   # 12->24
        x = self.conv3(x)   # 24->48
        x = self.conv4(x)   # 48->96 (temporal dimension matches)
        x = self.conv5(x)   # 96x96x96 -> 96x128x128 (spatial upsampling)
        x = self.conv6(x)   # Final output layer
        
        # Scale from [-1, 1] to [0, 1]
        x = (x + 1) / 2.0
        
        return x


class ConditionalC3DDiscriminatorImproved(nn.Module):
    """
    Improved Conditional Discriminator with increased capacity.
    """
    
    def __init__(self, nc=1, ndf=128, n_classes=20, video_length=96, video_size=128):
        super(ConditionalC3DDiscriminatorImproved, self).__init__()
        self.n_classes = n_classes
        self.ndf = ndf
        
        # Label embedding - project to feature space
        self.label_emb = nn.Embedding(n_classes, ndf * 8)
        
        # Video processing
        self.conv1 = nn.Sequential(
            nn.Conv3d(nc, ndf, kernel_size=(4, 4, 4), stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(ndf, ndf * 2, kernel_size=(4, 4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(ndf * 2, ndf * 4, kernel_size=(4, 4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv3d(ndf * 4, ndf * 8, kernel_size=(4, 4, 4), stride=2, padding=1, bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final layer - combine video features with label embedding
        self.fc = nn.Linear(ndf * 8 + ndf * 8, 1)  # video features + label embedding
    
    def forward(self, video, labels):
        """
        Args:
            video: Video tensor (batch_size, nc, T, H, W)
            labels: Class labels (batch_size,)
        Returns:
            Logits tensor (batch_size, 1) - no sigmoid (use BCEWithLogitsLoss)
        """
        # Process video
        x = self.conv1(video)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Global average pooling to get feature vector
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view(x.size(0), -1)  # (batch_size, ndf * 8)
        
        # Get label embedding
        label_emb = self.label_emb(labels)  # (batch_size, ndf * 8)
        
        # Concatenate video features and label embedding
        combined = torch.cat([x, label_emb], dim=1)  # (batch_size, ndf * 8 * 2)
        
        # Final classification (logits, no sigmoid)
        out = self.fc(combined)
        
        return out


def weights_init(m):
    """Initialize weights for GAN training stability."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('Embedding') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
