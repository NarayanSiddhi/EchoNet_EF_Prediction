"""
Data Augmentation using Perfect Reconstruction C3D-GAN
Designed for dataset balancing with Grad-CAM compatible video generation

This script uses PerfectReconstructionGenerator to balance the dataset by:
1. Analyzing dataset imbalance across demographic groups
2. Generating multiple variations (perfect copies + demographic variations) from real videos
3. Creating high-quality synthetic videos suitable for Grad-CAM visualization

USAGE EXAMPLES:

1. Analyze dataset imbalance:
   python Data_Augmentation.py --mode analyze --manifest data/processed_full/manifest_full.csv

2. Generate synthetic videos to balance dataset:
   python Data_Augmentation.py --mode generate \
       --checkpoint perfect_reconstruction_c3dgan/c3dgan_best.pt \
       --manifest data/processed_full/manifest_full.csv \
       --output_dir data_augmentation_output \
       --target_samples 500

ARCHITECTURE:
- Generator: PerfectReconstructionGenerator (U-Net encoder-decoder with residual blocks)
- Features: Squeeze-and-Excitation attention, skip connections for detail preservation
- Output: High-quality videos (32 frames by default, 64x64) suitable for Grad-CAM analysis
- Strategy: Generate multiple variations per real video to balance underrepresented groups
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
import sys

# Import PerfectReconstructionGenerator from use_case_3
# Try multiple import paths
try:
    from use_case_3_perfect_reconstruction.models import PerfectReconstructionGenerator
except ImportError:
    try:
        sys.path.insert(0, str(Path(__file__).parent / 'use_case_3_perfect_reconstruction'))
        from models import PerfectReconstructionGenerator
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent / 'use_case_2_demographic_variations'))
        from generate_demographic_variations import PerfectReconstructionGenerator


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


def save_video(frames, output_path, fps=18):
    """
    Save video frames.
    
    Default FPS is 18 so clip duration stays comparable when T increased:
    - With 32 frames at 18 fps ≈ 1.78 s; use --fps or post-process if you need 30 fps.
    """
    T, H, W = frames.shape
    frames_rgb = np.zeros((T, H, W, 3), dtype=np.uint8)
    for t in range(T):
        frames_rgb[t] = frames[t, :, :, np.newaxis]
    imageio.mimwrite(output_path, frames_rgb, fps=fps, codec='libx264', quality=8)


def parse_group_label(group_label):
    """
    Parse group label into components.
    
    Handles two formats:
    1. 'A4C_F_0-1' (view_sex_age_bin)
    2. 'A4C_F_0-1_normal' (view_sex_age_bin_bmi)
    """
    parts = group_label.split('_')
    if len(parts) >= 3:
        result = {
            'view': parts[0],
            'sex': parts[1],
        }
        
        # Check if BMI is included (4 parts = view_sex_age_bin_bmi)
        if len(parts) == 4:
            result['age_bin'] = parts[2]
            result['bmi'] = parts[3]
        else:
            # Age bin might have dashes (e.g., '11-15')
            result['age_bin'] = '_'.join(parts[2:])
            result['bmi'] = None  # Not specified
        
        return result
    return None


def get_age_from_bin(age_bin):
    """Convert age bin string to numeric age (midpoint)"""
    age_bin_map = {
        '0-1': 0.5,
        '1-2': 1.5,
        '2-3': 2.5,
        '2-5': 3.5,
        '3-5': 4.0,
        '5-8': 6.5,
        '6-10': 8.0,
        '8-12': 10.0,
        '11-15': 13.0,
        '12-15': 13.5,
        '15-18': 16.5,
        '16-18': 17.0
    }
    return age_bin_map.get(age_bin, 10.0)


def find_source_videos_for_target(manifest_df, target_group_info, needed_count, current_distribution=None):
    """
    Find source videos that can be varied to target a specific underrepresented group.
    
    Strategy:
    1. Find videos with same view (required)
    2. Find videos that can be varied to match target demographics
    3. Prioritize videos that need minimal changes
    4. Exclude videos that are already in the target group (generating from them doesn't help)
    """
    target_view = target_group_info['view']
    target_sex = target_group_info['sex']
    target_age_bin = target_group_info['age_bin']
    
    # Filter by view first (required)
    candidate_videos = manifest_df[manifest_df['view'] == target_view].copy()
    
    if len(candidate_videos) == 0:
        return pd.DataFrame()
    
    # Exclude videos already in the target group (generating from them doesn't help balance)
    if 'class_label' in candidate_videos.columns:
        candidate_videos = candidate_videos[candidate_videos['class_label'] != f"{target_view}_{target_sex}_{target_age_bin}"].copy()
    
    if len(candidate_videos) == 0:
        # If no other videos available, use any video from same view
        candidate_videos = manifest_df[manifest_df['view'] == target_view].copy()
    
    # Compute BMI if needed
    if 'bmi_category' not in candidate_videos.columns:
        candidate_videos['bmi_category'] = candidate_videos.apply(
            lambda row: compute_bmi_category(row.get('weight', 50), row.get('height', 150)),
            axis=1
        )
    
    # Score candidates: prefer videos that need fewer changes
    def score_video(row):
        score = 0
        # Same sex is better (no change needed)
        if row.get('sex') == target_sex:
            score += 10
        # Same age bin is better
        if row.get('age_bin') == target_age_bin:
            score += 10
        # Prefer videos with different BMI (so we can vary it)
        if row.get('bmi_category') != 'normal':  # Prefer non-normal to vary
            score += 5
        return score
    
    candidate_videos['score'] = candidate_videos.apply(score_video, axis=1)
    candidate_videos = candidate_videos.sort_values('score', ascending=False)
    
    # Return top candidates (get more than needed for variety)
    return candidate_videos.head(min(needed_count * 5, len(candidate_videos)))


def get_target_variation(target_group_info, source_sex, source_age, source_bmi):
    """
    Generate a variation that targets the underrepresented group.
    Returns demographics for the target group.
    """
    # Use target BMI if specified, otherwise use source BMI (or default to normal)
    target_bmi = target_group_info.get('bmi', source_bmi) if target_group_info.get('bmi') else source_bmi
    if not target_bmi or target_bmi == 'None':
        target_bmi = 'normal'
    
    return {
        'type': 'targeted_balancing',
        'sex': target_group_info['sex'],
        'age': get_age_from_bin(target_group_info['age_bin']),
        'bmi': target_bmi,
        'description': f'Targeting group {target_group_info["view"]}_{target_group_info["sex"]}_{target_group_info["age_bin"]}'
    }


def compute_bmi_category(weight, height):
    """Compute BMI category from weight and height"""
    if pd.isna(weight) or pd.isna(height) or height <= 0:
        return 'normal'
    bmi_val = weight / ((height / 100.0) ** 2)
    if bmi_val < 18.5:
        return 'underweight'
    elif bmi_val < 25:
        return 'normal'
    elif bmi_val < 30:
        return 'overweight'
    else:
        return 'obese'


# ============================================================================
# DATASET BALANCING ANALYSIS
# ============================================================================

def analyze_dataset_imbalance(manifest_path, target_samples=500, include_bmi=False):
    """
    Analyze dataset imbalance and identify underrepresented groups.
    
    If include_bmi=True, also analyzes BMI distribution and creates BMI-specific groups.
    """
    df = pd.read_csv(manifest_path)
    
    # Compute BMI if needed
    if 'bmi_category' not in df.columns:
        df['bmi_category'] = df.apply(
            lambda row: compute_bmi_category(row.get('weight', 50), row.get('height', 150)),
            axis=1
        )
    
    # Create class labels if not present
    if 'class_label' not in df.columns:
        if all(col in df.columns for col in ['view', 'sex', 'age_bin']):
            if include_bmi:
                df['class_label'] = (
                    df['view'].astype(str) + '_' +
                    df['sex'].astype(str) + '_' +
                    df['age_bin'].astype(str) + '_' +
                    df['bmi_category'].astype(str)
                )
            else:
                df['class_label'] = (
                    df['view'].astype(str) + '_' +
                    df['sex'].astype(str) + '_' +
                    df['age_bin'].astype(str)
                )
        else:
            raise ValueError("Cannot create class_label: missing required columns")
    
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
    
    # Also analyze BMI distribution if requested
    if include_bmi:
        print(f"\n=== BMI DISTRIBUTION ===")
        bmi_dist = df['bmi_category'].value_counts()
        for bmi, count in bmi_dist.items():
            print(f"  {bmi}: {count} samples ({count/len(df)*100:.1f}%)")
    
    return underrepresented, needed_samples, group_counts, df


# ============================================================================
# VIDEO GENERATION FOR BALANCING
# ============================================================================

def generate_videos_for_balancing(
    generator,
    manifest_df,
    underrepresented_groups,
    needed_samples,
    output_dir,
    device='cuda',
    video_length=32,
    video_size=64,
    variations_per_video=4
):
    """
    Generate synthetic videos to balance underrepresented groups.
    
    CORRECTED STRATEGY:
    - For each underrepresented group, find source videos that can be varied to target that group
    - Generate variations that specifically fill the gaps
    - Don't generate from videos IN the underrepresented group (that doesn't help)
    """
    generator.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_manifest = []
    
    print(f"\n=== GENERATING SYNTHETIC VIDEOS FOR BALANCING (CORRECTED STRATEGY) ===")
    print(f"Strategy: Target underrepresented groups by finding source videos that can become them")
    
    # Sort groups by how many samples they need (prioritize largest gaps)
    sorted_groups = sorted(needed_samples.items(), key=lambda x: x[1], reverse=True)
    
    with torch.no_grad():
        for group_label, n_needed in tqdm(sorted_groups, desc="Targeting groups"):
            if n_needed <= 0:
                continue
            
            # Parse target group
            target_info = parse_group_label(group_label)
            if target_info is None:
                print(f"Warning: Could not parse group label {group_label}, skipping")
                continue
            
            print(f"\n=== Targeting Group: {group_label} (need {n_needed} more samples) ===")
            print(f"  Target: view={target_info['view']}, sex={target_info['sex']}, age_bin={target_info['age_bin']}")
            
            # Find source videos that can be varied to target this group
            source_videos = find_source_videos_for_target(manifest_df, target_info, n_needed)
            
            if len(source_videos) == 0:
                print(f"  Warning: No suitable source videos found for {group_label}, skipping")
                continue
            
            print(f"  Found {len(source_videos)} candidate source videos")
            
            generated_count = 0
            source_idx = 0
            
            # Generate variations from source videos until we have enough
            while generated_count < n_needed and source_idx < len(source_videos):
                row = source_videos.iloc[source_idx]
                source_idx += 1
                
                try:
                    # Load source video
                    video_path = Path(row.get('processed_path', row.get('file_path', '')))
                    if not video_path.exists():
                        continue
                    
                    video = load_video(video_path, video_length, video_size)
                    video = video.unsqueeze(0).to(device)  # [1, 1, T, H, W]
                    
                    # Get target variation (demographics for the underrepresented group)
                    source_sex = row.get('sex', 'F')
                    source_age = float(row.get('age', 10))
                    source_bmi = row.get('bmi_category', 'normal')
                    if pd.isna(source_bmi) or source_bmi == '':
                        source_bmi = compute_bmi_category(row.get('weight', 50), row.get('height', 150))
                    
                    # Generate variation targeting the underrepresented group
                    variation = get_target_variation(target_info, source_sex, source_age, source_bmi)
                    
                    # Encode target demographics
                    demo_vector = encode_demographics(
                        variation['sex'], 
                        variation['age'], 
                        variation['bmi']
                    ).unsqueeze(0).to(device)
                    
                    # Generate synthetic video
                    synthetic = generator(video, demo_vector)
                    
                    # Convert to numpy
                    synthetic_np = synthetic[0, 0].cpu().numpy()  # [T, H, W]
                    synthetic_np = ((synthetic_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
                    
                    # Save
                    filename = f"synthetic_{group_label}_{generated_count:05d}_targeted.mp4"
                    filepath = output_dir / filename
                    save_video(synthetic_np, str(filepath))
                    
                    # Add to manifest
                    generated_manifest.append({
                        'view': target_info['view'],
                        'sex': variation['sex'],
                        'age_bin': target_info['age_bin'],
                        'class_label': group_label,
                        'file_name': filename,
                        'file_path': str(filepath),
                        'is_synthetic': True,
                        'source': 'PerfectReconstructionGAN',
                        'variation_type': 'targeted_balancing',
                        'original_id': row.name,
                        'EF': row.get('EF', row.get('ef', None))
                    })
                    
                    generated_count += 1
                    
                    if generated_count % 10 == 0:
                        print(f"  Generated {generated_count}/{n_needed} for {group_label}")
                
                except Exception as e:
                    print(f"\n  Error processing source video {row.name} for group {group_label}: {e}")
                    continue
            
            print(f"  ✓ Generated {generated_count}/{n_needed} videos for {group_label}")
    
    # Save manifest
    manifest_df_generated = pd.DataFrame(generated_manifest)
    manifest_path = output_dir / 'generated_manifest.csv'
    manifest_df_generated.to_csv(manifest_path, index=False)
    
    print(f"\n=== GENERATION COMPLETE ===")
    print(f"Generated {len(generated_manifest)} synthetic videos")
    print(f"Manifest saved to: {manifest_path}")
    
    # Calculate new distribution
    if len(generated_manifest) > 0:
        print(f"\n=== NEW DISTRIBUTION ===")
        new_dist = manifest_df_generated.groupby('class_label').size()
        print(f"Top 10 groups by generated count:")
        for group, count in new_dist.sort_values(ascending=False).head(10).items():
            print(f"  {group}: {count} synthetic videos")
    
    return manifest_df_generated


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_pretrained_generator(
    checkpoint_path,
    device='cuda',
    base_channels=64,
    spatial_size=64,
    conditioning='concat',
):
    """Load pretrained PerfectReconstructionGenerator. Use conditioning/video_size that match training."""
    print(f"Loading pretrained generator from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'generator' in checkpoint:
        spatial_size = int(checkpoint.get('video_size', spatial_size))
        conditioning = checkpoint.get('conditioning', conditioning)
        state = checkpoint['generator']
    elif 'netG_state_dict' in checkpoint:
        state = checkpoint['netG_state_dict']
    elif 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
    else:
        state = checkpoint

    generator = PerfectReconstructionGenerator(
        base_channels=base_channels,
        spatial_size=spatial_size,
        conditioning=conditioning,
    ).to(device)
    generator.load_state_dict(state)
    
    generator.eval()
    
    print(f"✓ Generator loaded successfully")
    
    return generator


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Data Augmentation using Perfect Reconstruction GAN')
    parser.add_argument('--mode', type=str, choices=['analyze', 'generate'], 
                       default='generate', help='Operation mode')
    parser.add_argument('--manifest', type=str, 
                       default='data/processed_full/manifest_full.csv',
                       help='Path to manifest CSV')
    parser.add_argument('--checkpoint', type=str,
                       default='use_case_3_perfect_reconstruction/ckpt_uc3_128x128_T32/recon_best.pt',
                       help='Path to pretrained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='data_augmentation_output',
                       help='Output directory for generated videos')
    parser.add_argument('--target_samples', type=int, default=500,
                       help='Target number of samples per group')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--video_length', type=int, default=32,
                       help='Number of frames (default: 32; aligns with C3D DCGAN T=32)')
    parser.add_argument('--video_size', type=int, default=128,
                       help='Spatial resolution (default: 128 for the retrained 128×128 checkpoint)')
    parser.add_argument('--variations_per_video', type=int, default=4,
                       help='Maximum variations to generate per real video')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.mode == 'analyze':
        # Analyze dataset imbalance
        underrepresented, needed_samples, group_counts, df = analyze_dataset_imbalance(
            args.manifest, args.target_samples
        )
        
        # Save analysis
        analysis = {
            'underrepresented_groups': {str(k): int(v) for k, v in underrepresented.items()},
            'needed_samples': {str(k): int(v) for k, v in needed_samples.items()},
            'total_needed': int(sum(needed_samples.values()))
        }
        output_path = Path(args.output_dir) / 'imbalance_analysis.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nAnalysis saved to: {output_path}")
    
    elif args.mode == 'generate':
        if args.checkpoint is None:
            print("Error: --checkpoint required for generation mode")
            print("Example: --checkpoint use_case_3_perfect_reconstruction/ckpt_uc3_128x128_T32/recon_best.pt")
            return
        
        # Load pretrained generator
        generator = load_pretrained_generator(args.checkpoint, device)
        
        # Analyze imbalance
        underrepresented, needed_samples, group_counts, manifest_df = analyze_dataset_imbalance(
            args.manifest, args.target_samples
        )
        
        # Generate videos
        underrepresented_groups = list(underrepresented.keys())
        manifest_df_generated = generate_videos_for_balancing(
            generator,
            manifest_df,
            underrepresented_groups,
            needed_samples,
            args.output_dir,
            device,
            args.video_length,
            args.video_size,
            args.variations_per_video
        )
        
        print(f"\n✓ Data augmentation complete!")
        print(f"  Generated {len(manifest_df_generated)} synthetic videos")
        print(f"  Output directory: {args.output_dir}")
        print(f"\nNext steps:")
        print(f"  1. Combine generated manifest with original manifest")
        print(f"  2. Train EF prediction model on balanced dataset")
        print(f"  3. Validate with Grad-CAM analysis")


if __name__ == '__main__':
    main()
