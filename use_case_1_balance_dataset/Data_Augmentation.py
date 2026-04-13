"""
C3D ResU-Net GAN for High-Quality Synthetic Video Generation
Designed for dataset balancing and Grad-CAM compatible video generation

This script implements a Residual U-Net GAN with 3D convolutions to generate
high-quality synthetic echocardiogram videos that:
1. Balance dataset across demographic groups
2. Generate videos with proper patterns for Grad-CAM visualization
3. Support both training and inference with pretrained models

USAGE EXAMPLES:

1. Analyze dataset imbalance:
   python Data_Augmentation.py --mode analyze --manifest data/processed_full/manifest_full.csv

2. Generate synthetic videos using pretrained model:
   python Data_Augmentation.py --mode generate \
       --checkpoint c3dgan/checkpoints/checkpoint_epoch_200.pth \
       --manifest data/processed_full/manifest_full.csv \
       --output_dir data_augmentation_output \
       --target_samples 500

3. Generate with custom config:
   python Data_Augmentation.py --mode generate \
       --checkpoint path/to/checkpoint.pth \
       --config c3dgan/config.yaml \
       --manifest data/processed_full/manifest_full.csv \
       --output_dir data_augmentation_output

ARCHITECTURE:
- Generator: C3D ResU-Net with residual blocks, skip connections, and conditional fusion
- Discriminator: C3D discriminator with conditional input
- Features: Squeeze-and-Excitation attention, U-Net skip connections for detail preservation
- Output: High-quality videos (96 frames, 128x128) suitable for Grad-CAM analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
import yaml
import json
from typing import Dict, List, Tuple, Optional
import imageio
from torch.utils.data import Dataset, DataLoader
import random


# ============================================================================
# C3D RESU-NET GAN ARCHITECTURE
# ============================================================================

class ResidualBlock3D(nn.Module):
    """3D Residual Block with Squeeze-and-Excitation for better feature learning"""
    def __init__(self, channels, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.use_se = use_se
        
        if use_se:
            # Squeeze-and-Excitation for channel attention
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(channels, max(channels // 16, 4), 1),
                nn.ReLU(),
                nn.Conv3d(max(channels // 16, 4), channels, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.use_se:
            se_weight = self.se(out)
            out = out * se_weight
        
        out = out + residual
        return F.relu(out)


class ConditionalEmbedding(nn.Module):
    """Embed class labels into latent space for conditional generation"""
    def __init__(self, n_classes, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, labels):
        emb = self.embedding(labels)
        return self.projection(emb)


class SpatialConditionalFusion(nn.Module):
    """Fuse conditional information spatially into feature maps"""
    def __init__(self, feature_channels, embed_dim=128):
        super().__init__()
        self.embed_proj = nn.Linear(embed_dim, feature_channels)
        self.fusion = nn.Sequential(
            nn.Conv3d(feature_channels * 2, feature_channels, kernel_size=1),
            nn.BatchNorm3d(feature_channels),
            nn.ReLU()
        )

    def forward(self, features, condition_embed):
        B, C, T, H, W = features.shape
        cond_features = self.embed_proj(condition_embed)
        cond_spatial = cond_features.view(B, C, 1, 1, 1).expand(B, C, T, H, W)
        combined = torch.cat([features, cond_spatial], dim=1)
        fused = self.fusion(combined)
        return fused


class C3DResUNetGenerator(nn.Module):
    """
    C3D Residual U-Net Generator for high-quality video generation
    Architecture: Encoder-Decoder with skip connections and residual blocks
    """
    def __init__(self, nz=100, ngf=64, nc=1, n_classes=20, video_length=96, video_size=128):
        super().__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.n_classes = n_classes
        self.video_length = video_length
        self.video_size = video_size
        
        # Conditional embedding
        self.condition_embedding = ConditionalEmbedding(n_classes, embed_dim=128)
        
        # Project noise + condition to initial feature map
        # Start with 6x6x6 to allow proper upsampling to 96x128x128
        self.fc = nn.Linear(nz + 128, ngf * 8 * 6 * 6 * 6)
        
        # ENCODER (Downsampling) - but we're actually building up from small
        # So this is more like initial feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv3d(ngf * 8, ngf * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(ngf * 8),
            nn.ReLU(True),
            ResidualBlock3D(ngf * 8),
            ResidualBlock3D(ngf * 8)
        )
        self.cond_fusion_init = SpatialConditionalFusion(ngf * 8, 128)
        
        # Upsampling layers to reach target dimensions
        # 6x6x6 -> 12x12x12
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose3d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(ngf * 4),
            nn.ReLU(True),
            ResidualBlock3D(ngf * 4),
            ResidualBlock3D(ngf * 4)
        )
        self.cond_fusion1 = SpatialConditionalFusion(ngf * 4, 128)
        
        # 12x12x12 -> 24x24x24
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True),
            ResidualBlock3D(ngf * 2),
            ResidualBlock3D(ngf * 2)
        )
        self.cond_fusion2 = SpatialConditionalFusion(ngf * 2, 128)
        
        # 24x24x24 -> 48x48x48
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose3d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(ngf),
            nn.ReLU(True),
            ResidualBlock3D(ngf),
            ResidualBlock3D(ngf)
        )
        self.cond_fusion3 = SpatialConditionalFusion(ngf, 128)
        
        # 48x48x48 -> 96x96x96 (temporal dimension matches)
        self.upsample4 = nn.Sequential(
            nn.ConvTranspose3d(ngf, ngf // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(ngf // 2),
            nn.ReLU(True),
            ResidualBlock3D(ngf // 2)
        )
        
        # 96x96x96 -> 96x128x128 (spatial upsampling only)
        self.upsample_spatial = nn.Sequential(
            nn.ConvTranspose3d(ngf // 2, ngf // 4, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ngf // 4),
            nn.ReLU(True),
            ResidualBlock3D(ngf // 4)
        )
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv3d(ngf // 4, nc, kernel_size=3, padding=1, bias=False),
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
        # Embed conditions
        cond_embed = self.condition_embedding(labels)  # (B, 128)
        
        # Concatenate noise and condition
        input_vec = torch.cat([noise, cond_embed], dim=1)  # (B, nz + 128)
        
        # Project to initial feature map
        x = self.fc(input_vec)
        x = x.view(-1, self.ngf * 8, 6, 6, 6)
        
        # Initial feature extraction with conditional fusion
        x = self.initial_conv(x)
        x = self.cond_fusion_init(x, cond_embed)
        
        # Progressive upsampling with conditional fusion
        x = self.upsample1(x)  # 6->12
        x = self.cond_fusion1(x, cond_embed)
        
        x = self.upsample2(x)  # 12->24
        x = self.cond_fusion2(x, cond_embed)
        
        x = self.upsample3(x)  # 24->48
        x = self.cond_fusion3(x, cond_embed)
        
        x = self.upsample4(x)  # 48->96 (temporal dimension)
        
        x = self.upsample_spatial(x)  # 96x96x96 -> 96x128x128 (spatial only)
        
        # Final output
        out = self.final_conv(x)
        
        # Scale from [-1, 1] to [0, 1]
        out = (out + 1) / 2.0
        
        return out


class C3DResUNetDiscriminator(nn.Module):
    """C3D Residual Discriminator with conditional input"""
    def __init__(self, nc=1, ndf=64, n_classes=20, video_length=96, video_size=128):
        super().__init__()
        self.n_classes = n_classes
        
        # Conditional embedding
        self.condition_embedding = ConditionalEmbedding(n_classes, embed_dim=128)
        
        # Video processing layers
        self.conv1 = nn.Sequential(
            nn.Conv3d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv3d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Combine video features with condition
        self.fc = nn.Linear(ndf * 8 + 128, 1)
    
    def forward(self, video, labels):
        """
        Args:
            video: Video tensor (batch_size, nc, T, H, W)
            labels: Class labels (batch_size,)
        Returns:
            Logits (batch_size, 1) - no sigmoid
        """
        # Process video
        x = self.conv1(video)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view(x.size(0), -1)  # (B, ndf * 8)
        
        # Get condition embedding
        cond_embed = self.condition_embedding(labels)  # (B, 128)
        
        # Concatenate and classify
        combined = torch.cat([x, cond_embed], dim=1)
        out = self.fc(combined)
        
        return out


# ============================================================================
# DATASET AND DATA LOADING
# ============================================================================

class ConditionalVideoDataset(Dataset):
    """Dataset for conditional video generation"""
    def __init__(self, manifest_path, video_dir, video_length=96, video_size=128, 
                 class_to_idx=None, filter_groups=None):
        self.manifest = pd.read_csv(manifest_path)
        self.video_dir = Path(video_dir)
        self.video_length = video_length
        self.video_size = video_size
        
        # Create class labels if not present
        if 'class_label' not in self.manifest.columns:
            if all(col in self.manifest.columns for col in ['view', 'sex', 'age_bin']):
                self.manifest['class_label'] = (
                    self.manifest['view'].astype(str) + '_' +
                    self.manifest['sex'].astype(str) + '_' +
                    self.manifest['age_bin'].astype(str)
                )
            else:
                raise ValueError("Cannot create class_label: missing required columns")
        
        # Resolve video paths
        if 'resolved_path' not in self.manifest.columns:
            def resolve_path(row):
                processed = row.get('processed_path', None)
                if pd.notna(processed) and processed and Path(processed).exists():
                    return processed
                file_path = row.get('file_path', None)
                if pd.notna(file_path) and file_path and Path(file_path).exists():
                    return file_path
                return None
            self.manifest['resolved_path'] = self.manifest.apply(resolve_path, axis=1)
        
        # Filter existing videos
        self.manifest = self.manifest[
            self.manifest['resolved_path'].apply(lambda x: x is not None and Path(x).exists())
        ].reset_index(drop=True)
        
        # Filter groups if specified
        if filter_groups is not None:
            self.manifest = self.manifest[
                self.manifest['class_label'].isin(filter_groups)
            ].reset_index(drop=True)
        
        # Create class mapping
        if class_to_idx is None:
            unique_labels = sorted(self.manifest['class_label'].unique())
            self.class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.class_to_idx = class_to_idx
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
    
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        video_path = Path(row['resolved_path'])
        class_label = row['class_label']
        class_idx = self.class_to_idx[class_label]
        
        # Load video
        video = self.load_video(video_path)
        
        return video, class_idx, class_label
    
    def load_video(self, video_path):
        """Load and preprocess video"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (self.video_size, self.video_size))
            frames.append(frame)
        cap.release()
        
        # Uniform sampling to video_length
        if len(frames) > self.video_length:
            indices = np.linspace(0, len(frames) - 1, self.video_length, dtype=int)
            frames = [frames[i] for i in indices]
        while len(frames) < self.video_length:
            frames.append(frames[-1] if frames else np.zeros((self.video_size, self.video_size), dtype=np.uint8))
        
        # Normalize to [0, 1]
        video = np.array(frames[:self.video_length], dtype=np.float32) / 255.0
        video = torch.from_numpy(video).unsqueeze(0)  # (1, T, H, W)
        
        return video


# ============================================================================
# DATASET BALANCING ANALYSIS
# ============================================================================

def analyze_dataset_imbalance(manifest_path, target_samples=500):
    """Analyze dataset imbalance and identify underrepresented groups"""
    df = pd.read_csv(manifest_path)
    
    # Create class labels if not present
    if 'class_label' not in df.columns:
        if all(col in df.columns for col in ['view', 'sex', 'age_bin']):
            df['class_label'] = (
                df['view'].astype(str) + '_' +
                df['sex'].astype(str) + '_' +
                df['age_bin'].astype(str)
            )
        else:
            raise ValueError("Cannot create class_label")
    
    # Count samples per group
    group_counts = df.groupby('class_label').size().sort_values()
    
    # Identify underrepresented groups
    underrepresented = group_counts[group_counts < target_samples].to_dict()
    needed_samples = {group: max(0, target_samples - count) 
                      for group, count in underrepresented.items()}
    
    print(f"\n=== DATASET IMBALANCE ANALYSIS ===")
    print(f"Total groups: {len(group_counts)}")
    print(f"Underrepresented groups (<{target_samples} samples): {len(underrepresented)}")
    print(f"Total samples needed: {sum(needed_samples.values())}")
    print(f"\nTop 10 underrepresented groups:")
    for group, count in list(underrepresented.items())[:10]:
        print(f"  {group}: {count} samples (need {needed_samples[group]})")
    
    return underrepresented, needed_samples, group_counts


# ============================================================================
# VIDEO GENERATION
# ============================================================================

def generate_videos_for_balancing(
    generator,
    class_to_idx,
    underrepresented_groups,
    needed_samples,
    output_dir,
    device='cuda',
    batch_size=8
):
    """Generate synthetic videos to balance underrepresented groups"""
    generator.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_manifest = []
    
    print(f"\n=== GENERATING SYNTHETIC VIDEOS ===")
    
    with torch.no_grad():
        for group_label in tqdm(underrepresented_groups, desc="Groups"):
            if group_label not in class_to_idx:
                print(f"Warning: {group_label} not in class mapping, skipping")
                continue
            
            class_idx = class_to_idx[group_label]
            n_samples = needed_samples[group_label]
            
            if n_samples <= 0:
                continue
            
            # Generate in batches
            for batch_start in range(0, n_samples, batch_size):
                batch_size_actual = min(batch_size, n_samples - batch_start)
                
                # Generate noise
                noise = torch.randn(batch_size_actual, generator.nz, device=device)
                labels = torch.full((batch_size_actual,), class_idx, device=device, dtype=torch.long)
                
                # Generate videos
                fake_videos = generator(noise, labels)  # (B, 1, T, H, W)
                
                # Save each video
                for i in range(batch_size_actual):
                    video_idx = batch_start + i
                    video_tensor = fake_videos[i, 0].detach().cpu()  # (T, H, W)
                    
                    # Clamp and convert to uint8
                    video_tensor = torch.clamp(video_tensor, 0.0, 1.0)
                    video_np = (video_tensor.numpy() * 255).astype(np.uint8)
                    
                    # Save video
                    filename = f"synthetic_{group_label}_{video_idx:05d}.mp4"
                    filepath = output_dir / filename
                    save_video_frames(video_np, str(filepath))
                    
                    # Parse group label
                    parts = group_label.split('_')
                    view = parts[0] if len(parts) > 0 else None
                    sex = parts[1] if len(parts) > 1 else None
                    age_bin = parts[2] if len(parts) > 2 else None
                    
                    generated_manifest.append({
                        'view': view,
                        'sex': sex,
                        'age_bin': age_bin,
                        'class_label': group_label,
                        'file_name': filename,
                        'file_path': str(filepath),
                        'is_synthetic': True,
                        'source': 'C3DResUNetGAN'
                    })
    
    # Save manifest
    manifest_df = pd.DataFrame(generated_manifest)
    manifest_path = output_dir / 'generated_manifest.csv'
    manifest_df.to_csv(manifest_path, index=False)
    
    print(f"\n=== GENERATION COMPLETE ===")
    print(f"Generated {len(generated_manifest)} synthetic videos")
    print(f"Manifest saved to: {manifest_path}")
    
    return manifest_df


def save_video_frames(frames, output_path, fps=30):
    """Save video frames to file"""
    try:
        # frames: (T, H, W) uint8
        T, H, W = frames.shape
        frames_rgb = np.zeros((T, H, W, 3), dtype=np.uint8)
        for t in range(T):
            frames_rgb[t] = np.stack([frames[t]] * 3, axis=-1)
        
        imageio.mimwrite(output_path, frames_rgb, fps=fps, codec='libx264', quality=8)
    except Exception as e:
        print(f"Error saving video {output_path}: {e}")


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_pretrained_generator(checkpoint_path, config_path=None, device='cuda'):
    """Load pretrained generator from checkpoint"""
    print(f"Loading pretrained generator from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    class_to_idx = checkpoint.get('class_to_idx', checkpoint.get('class_mapping', {}))
    n_classes = len(class_to_idx)
    
    # Load model config
    model_cfg = checkpoint.get('model_cfg', None)
    if config_path:
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            if cfg and 'model' in cfg:
                model_cfg = cfg['model']
        except Exception:
            pass
    
    # Default config
    if model_cfg is None:
        model_cfg = {
            'nz': 100,
            'ngf': 64,
            'nc': 1,
            'video_length': 96,
            'video_size': 128
        }
    
    # Create generator
    generator = C3DResUNetGenerator(
        nz=model_cfg.get('nz', 100),
        ngf=model_cfg.get('ngf', 64),
        nc=model_cfg.get('nc', 1),
        n_classes=n_classes,
        video_length=model_cfg.get('video_length', 96),
        video_size=model_cfg.get('video_size', 128)
    ).to(device)
    
    # Load weights
    if 'netG_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['netG_state_dict'])
    elif 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
    else:
        generator.load_state_dict(checkpoint)
    
    generator.eval()
    
    print(f"Generator loaded successfully (n_classes={n_classes})")
    
    return generator, class_to_idx


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='C3D ResU-Net GAN for Data Augmentation')
    parser.add_argument('--mode', type=str, choices=['analyze', 'generate', 'train'], 
                       default='generate', help='Operation mode')
    parser.add_argument('--manifest', type=str, 
                       default='data/processed_full/manifest_full.csv',
                       help='Path to manifest CSV')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to pretrained model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')
    parser.add_argument('--output_dir', type=str, default='data_augmentation_output',
                       help='Output directory for generated videos')
    parser.add_argument('--target_samples', type=int, default=500,
                       help='Target number of samples per group')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for generation')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.mode == 'analyze':
        # Analyze dataset imbalance
        underrepresented, needed_samples, group_counts = analyze_dataset_imbalance(
            args.manifest, args.target_samples
        )
        
        # Save analysis
        analysis = {
            'underrepresented_groups': underrepresented,
            'needed_samples': needed_samples,
            'total_needed': sum(needed_samples.values())
        }
        output_path = Path(args.output_dir) / 'imbalance_analysis.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nAnalysis saved to: {output_path}")
    
    elif args.mode == 'generate':
        if args.checkpoint is None:
            print("Error: --checkpoint required for generation mode")
            return
        
        # Load pretrained generator
        generator, class_to_idx = load_pretrained_generator(
            args.checkpoint, args.config, device
        )
        
        # Analyze imbalance
        underrepresented, needed_samples, _ = analyze_dataset_imbalance(
            args.manifest, args.target_samples
        )
        
        # Generate videos
        underrepresented_groups = list(underrepresented.keys())
        manifest_df = generate_videos_for_balancing(
            generator,
            class_to_idx,
            underrepresented_groups,
            needed_samples,
            args.output_dir,
            device,
            args.batch_size
        )
        
        print(f"\n✓ Data augmentation complete!")
        print(f"  Generated {len(manifest_df)} synthetic videos")
        print(f"  Output directory: {args.output_dir}")
    
    elif args.mode == 'train':
        print("Training mode not yet implemented in this script.")
        print("Please use the training scripts in c3dgan/ directory for training.")
        print("After training, use --mode generate with --checkpoint to generate videos.")


if __name__ == '__main__':
    main()
