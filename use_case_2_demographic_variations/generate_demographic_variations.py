"""
Generate 3 demographic variations from a real video using Perfect Reconstruction C3D-GAN

For each real video, generates:
1. Age variation (change age, keep sex and BMI same)
2. Sex variation (change sex, keep age and BMI same)
3. BMI variation (change BMI, keep age and sex same)

Preserves cardiac motion while changing demographic attributes.
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
import imageio


# ============================================================================
# PERFECT RECONSTRUCTION GENERATOR ARCHITECTURE
# ============================================================================

class ResidualBlock3D(nn.Module):
    """3D Residual Block for better gradient flow"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        # Squeeze-and-Excitation for channel attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // 16, 1),
            nn.ReLU(),
            nn.Conv3d(channels // 16, channels, 1),
            nn.Sigmoid()
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
            nn.ReLU()
        )

    def forward(self, demographics):
        return self.embedding(demographics)


class SpatialDemographicFusion(nn.Module):
    """Fuse demographic information spatially"""
    def __init__(self, feature_channels, demo_embed_dim=128):
        super().__init__()
        self.demo_proj = nn.Linear(demo_embed_dim, feature_channels)
        self.fusion = nn.Sequential(
            nn.Conv3d(feature_channels * 2, feature_channels, kernel_size=1),
            nn.BatchNorm3d(feature_channels),
            nn.ReLU()
        )

    def forward(self, features, demo_embed):
        B, C, T, H, W = features.shape
        demo_features = self.demo_proj(demo_embed)
        demo_spatial = demo_features.view(B, C, 1, 1, 1).expand(B, C, T, H, W)
        combined = torch.cat([features, demo_spatial], dim=1)
        fused = self.fusion(combined)
        return fused


class PerfectReconstructionGenerator(nn.Module):
    """C3D-GAN Generator for video reconstruction with demographic conditioning"""
    def __init__(self, base_channels=64):
        super().__init__()
        self.demo_embedding = DemographicEmbedding(demo_dim=11, embed_dim=128)
        
        # ENCODER
        self.enc1 = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            ResidualBlock3D(base_channels),
            ResidualBlock3D(base_channels)
        )
        self.demo_fusion1 = SpatialDemographicFusion(base_channels, 128)
        
        self.enc2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels*2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(),
            ResidualBlock3D(base_channels*2),
            ResidualBlock3D(base_channels*2)
        )
        self.demo_fusion2 = SpatialDemographicFusion(base_channels*2, 128)
        
        self.enc3 = nn.Sequential(
            nn.Conv3d(base_channels*2, base_channels*4, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels*4),
            nn.ReLU(),
            ResidualBlock3D(base_channels*4),
            ResidualBlock3D(base_channels*4)
        )
        self.demo_fusion3 = SpatialDemographicFusion(base_channels*4, 128)
        
        self.enc4 = nn.Sequential(
            nn.Conv3d(base_channels*4, base_channels*8, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels*8),
            nn.ReLU(),
            ResidualBlock3D(base_channels*8),
            ResidualBlock3D(base_channels*8)
        )
        
        # BOTTLENECK
        self.bottleneck = nn.Sequential(
            ResidualBlock3D(base_channels*8),
            ResidualBlock3D(base_channels*8),
            ResidualBlock3D(base_channels*8),
            ResidualBlock3D(base_channels*8)
        )
        self.demo_fusion_bottleneck = SpatialDemographicFusion(base_channels*8, 128)
        
        # DECODER
        self.dec4 = nn.Sequential(
            nn.ConvTranspose3d(base_channels*8, base_channels*4, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels*4),
            nn.ReLU(),
            ResidualBlock3D(base_channels*4)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(base_channels*8, base_channels*2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(),
            ResidualBlock3D(base_channels*2)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(base_channels*4, base_channels, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            ResidualBlock3D(base_channels)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv3d(base_channels*2, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            ResidualBlock3D(base_channels),
            ResidualBlock3D(base_channels)
        )
        
        self.output = nn.Sequential(
            nn.Conv3d(base_channels, base_channels//2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels//2),
            nn.ReLU(),
            nn.Conv3d(base_channels//2, 1, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, x, demographics):
        """Args: x [B, 1, T, H, W], demographics [B, 11]"""
        demo_embed = self.demo_embedding(demographics)
        
        # ENCODER
        e1 = self.enc1(x)
        e1 = self.demo_fusion1(e1, demo_embed)
        e2 = self.enc2(e1)
        e2 = self.demo_fusion2(e2, demo_embed)
        e3 = self.enc3(e2)
        e3 = self.demo_fusion3(e3, demo_embed)
        e4 = self.enc4(e3)
        
        # BOTTLENECK
        b = self.bottleneck(e4)
        b = self.demo_fusion_bottleneck(b, demo_embed)
        
        # DECODER
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e3], dim=1)
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.dec1(d2)
        
        out = self.output(d1)
        return out


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def encode_demographics(sex, age, bmi, sex_map={'F': 0, 'M': 1, 'O': 0}, 
                       age_bins=[0, 5, 10, 15, 18], 
                       bmi_map={'underweight': 0, 'normal': 1, 'overweight': 2, 'obese': 3}):
    """One-hot encode demographics to 11-dim vector"""
    sex_val = sex_map.get(sex, 0)
    sex_onehot = torch.zeros(2)
    sex_onehot[sex_val] = 1
    
    age_bin = np.digitize(age, age_bins)
    age_onehot = torch.zeros(5)
    age_onehot[min(age_bin, 4)] = 1
    
    bmi_idx = bmi_map.get(bmi, 1)
    bmi_onehot = torch.zeros(4)
    bmi_onehot[bmi_idx] = 1
    
    return torch.cat([sex_onehot, age_onehot, bmi_onehot])


def load_video(video_path, video_length=32, video_size=64):
    """Load and preprocess video"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (video_size, video_size))
        frames.append(frame)
    cap.release()
    
    # Uniform sampling
    if len(frames) > video_length:
        indices = np.linspace(0, len(frames)-1, video_length, dtype=int)
        frames = [frames[i] for i in indices]
    while len(frames) < video_length:
        frames.append(frames[-1] if frames else np.zeros((video_size, video_size)))
    
    # Normalize [-1, 1]
    video = np.array(frames[:video_length], dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(video).unsqueeze(0)  # [1, T, H, W]


def save_video(frames, output_path, fps=30):
    """Save video frames"""
    T, H, W = frames.shape
    frames_rgb = np.zeros((T, H, W, 3), dtype=np.uint8)
    for t in range(T):
        frames_rgb[t] = frames[t, :, :, np.newaxis]
    imageio.mimwrite(output_path, frames_rgb, fps=fps, codec='libx264', quality=8)


def get_demographic_variations(original_sex, original_age, original_bmi):
    """Generate 3 demographic variations"""
    sex_map = {'F': 0, 'M': 1, 'O': 0}
    age_bins = [0, 5, 10, 15, 18]
    bmi_map = {'underweight': 0, 'normal': 1, 'overweight': 2, 'obese': 3}
    
    # Get original indices
    sex_idx = sex_map.get(original_sex, 0)
    age_bin_idx = min(np.digitize(original_age, age_bins), 4)
    bmi_idx = bmi_map.get(original_bmi, 1)
    
    variations = []
    
    # Variation 1: Change age (keep sex and BMI)
    new_age_bin = (age_bin_idx + 1) % 5  # Next age bin
    new_age = (age_bins[new_age_bin] + age_bins[min(new_age_bin+1, 4)]) / 2 if new_age_bin < 4 else 16.5
    variations.append({
        'type': 'age_variation',
        'sex': original_sex,
        'age': new_age,
        'bmi': original_bmi,
        'description': f'Age changed from {original_age:.1f} to {new_age:.1f}'
    })
    
    # Variation 2: Change sex (keep age and BMI)
    new_sex_idx = 1 - sex_idx  # Flip sex
    new_sex = 'M' if new_sex_idx == 1 else 'F'
    variations.append({
        'type': 'sex_variation',
        'sex': new_sex,
        'age': original_age,
        'bmi': original_bmi,
        'description': f'Sex changed from {original_sex} to {new_sex}'
    })
    
    # Variation 3: Change BMI (keep age and sex)
    new_bmi_idx = (bmi_idx + 1) % 4  # Next BMI category
    bmi_categories = ['underweight', 'normal', 'overweight', 'obese']
    new_bmi = bmi_categories[new_bmi_idx]
    variations.append({
        'type': 'bmi_variation',
        'sex': original_sex,
        'age': original_age,
        'bmi': new_bmi,
        'description': f'BMI changed from {original_bmi} to {new_bmi}'
    })
    
    return variations


# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_demographic_variations(
    manifest_csv,
    checkpoint_path,
    output_dir='demographic_variations',
    video_length=32,
    video_size=64,
    device='cuda',
    max_videos=None
):
    """
    Generate 3 demographic variations for each real video
    
    Args:
        manifest_csv: Path to manifest CSV with video paths and demographics
        checkpoint_path: Path to perfect reconstruction checkpoint
        output_dir: Directory to save variations
        video_length: Number of frames (default: 32)
        video_size: Spatial resolution (default: 64)
        device: Device to use
        max_videos: Maximum number of videos to process (None = all)
    """
    print("="*70)
    print("GENERATING DEMOGRAPHIC VARIATIONS FROM REAL VIDEOS")
    print("="*70)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading model: {checkpoint_path}")
    netG = PerfectReconstructionGenerator(base_channels=64).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    netG.load_state_dict(checkpoint['generator'])
    netG.eval()
    print("✓ Model loaded")
    
    # Load manifest
    df = pd.read_csv(manifest_csv)
    if max_videos:
        df = df.head(max_videos)
    
    print(f"\nProcessing {len(df)} videos...")
    print(f"Each video will generate 3 variations (age, sex, BMI)")
    print(f"Total outputs: {len(df) * 3} synthetic videos\n")
    
    results = []
    
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating variations"):
            try:
                # Load original video
                video_path = Path(row.get('processed_path', row.get('video_path', '')))
                if not video_path.exists():
                    print(f"Warning: Video not found: {video_path}, skipping")
                    continue
                
                video = load_video(video_path, video_length, video_size)
                video = video.unsqueeze(0).to(device)  # [1, 1, T, H, W]
                
                # Get original demographics
                original_sex = row.get('sex', row.get('Sex', 'F'))
                original_age = float(row.get('age', row.get('Age', 10)))
                original_bmi = row.get('bmi_category', 'normal')
                
                # If BMI not in manifest, compute from weight/height
                if pd.isna(original_bmi) or original_bmi == '':
                    weight = row.get('weight', 50)
                    height = row.get('height', 150)
                    if pd.notna(weight) and pd.notna(height) and height > 0:
                        bmi_val = weight / ((height / 100.0) ** 2)
                        if bmi_val < 18.5:
                            original_bmi = 'underweight'
                        elif bmi_val < 25:
                            original_bmi = 'normal'
                        elif bmi_val < 30:
                            original_bmi = 'overweight'
                        else:
                            original_bmi = 'obese'
                    else:
                        original_bmi = 'normal'
                
                # Get demographic variations
                variations = get_demographic_variations(original_sex, original_age, original_bmi)
                
                # Generate each variation
                for var_idx, variation in enumerate(variations):
                    # Encode demographics
                    demo_vector = encode_demographics(
                        variation['sex'], 
                        variation['age'], 
                        variation['bmi']
                    ).unsqueeze(0).to(device)
                    
                    # Generate variation (preserves cardiac motion from original video)
                    synthetic = netG(video, demo_vector)
                    
                    # Convert to numpy
                    synthetic_np = synthetic[0, 0].cpu().numpy()  # [T, H, W]
                    synthetic_np = ((synthetic_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
                    
                    # Save
                    output_filename = f"video_{idx:04d}_var{var_idx+1}_{variation['type']}.mp4"
                    output_path = output_dir / output_filename
                    save_video(synthetic_np, str(output_path))
                    
                    results.append({
                        'original_id': idx,
                        'original_path': str(video_path),
                        'variation_type': variation['type'],
                        'variation_description': variation['description'],
                        'original_sex': original_sex,
                        'original_age': original_age,
                        'original_bmi': original_bmi,
                        'variation_sex': variation['sex'],
                        'variation_age': variation['age'],
                        'variation_bmi': variation['bmi'],
                        'synthetic_path': str(output_path),
                        'EF': row.get('EF', row.get('ef', None))
                    })
            
            except Exception as e:
                print(f"\nError processing video {idx}: {e}")
                continue
    
    # Save manifest
    results_df = pd.DataFrame(results)
    manifest_path = output_dir / 'variations_manifest.csv'
    results_df.to_csv(manifest_path, index=False)
    
    print(f"\n✅ GENERATION COMPLETE!")
    print(f"Total variations generated: {len(results)}")
    print(f"Saved to: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"\nVariation breakdown:")
    print(f"  Age variations: {(results_df['variation_type'] == 'age_variation').sum()}")
    print(f"  Sex variations: {(results_df['variation_type'] == 'sex_variation').sum()}")
    print(f"  BMI variations: {(results_df['variation_type'] == 'bmi_variation').sum()}")
    
    return results_df


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate demographic variations from real videos"
    )
    parser.add_argument('--manifest', type=str, required=True,
                       help='Path to manifest CSV with video paths and demographics')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to perfect reconstruction checkpoint')
    parser.add_argument('--output_dir', type=str, default='demographic_variations',
                       help='Output directory for variations')
    parser.add_argument('--video_length', type=int, default=32,
                       help='Number of frames (default: 32; aligns with C3D DCGAN T=32)')
    parser.add_argument('--video_size', type=int, default=64,
                       help='Spatial resolution (default: 64)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--max_videos', type=int, default=None,
                       help='Maximum number of videos to process (None = all)')
    
    args = parser.parse_args()
    
    generate_demographic_variations(
        manifest_csv=args.manifest,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        video_length=args.video_length,
        video_size=args.video_size,
        device=args.device,
        max_videos=args.max_videos
    )
