"""
Generate high-quality synthetic videos with sharpening and improved model.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('c3dgan')
from generate import load_generator, save_video
from tqdm import tqdm
from scipy import ndimage

def sharpen_video(video_tensor):
    """
    Apply unsharp masking to enhance video sharpness.
    
    Args:
        video_tensor: numpy array of shape (T, H, W) with values in [0, 255]
    Returns:
        Sharpened video array
    """
    # Convert to float for processing
    video_float = video_tensor.astype(np.float32)
    
    # Apply unsharp masking frame by frame
    sharpened = np.zeros_like(video_float)
    for t in range(video_tensor.shape[0]):
        frame = video_float[t]
        # Unsharp masking: original + (original - blurred) * amount
        blurred = ndimage.gaussian_filter(frame, sigma=1.0)
        sharpened[t] = np.clip(frame + (frame - blurred) * 1.5, 0, 255)
    
    return sharpened.astype(np.uint8)

def enhance_brightness_and_contrast(video_tensor, target_mean=0.3, contrast_factor=1.5):
    """
    Enhance video brightness and contrast.
    
    Args:
        video_tensor: Tensor of shape (T, H, W) with values in [0, 1]
        target_mean: Target mean brightness (0-1)
        contrast_factor: Contrast enhancement factor
    """
    current_mean = video_tensor.mean()
    
    if current_mean > 0:
        brightness_scale = min(target_mean / current_mean, 10.0)
        video_tensor = video_tensor * brightness_scale
    else:
        video_tensor = video_tensor + 0.1
    
    # Apply contrast
    video_tensor = (video_tensor - video_tensor.mean()) * contrast_factor + video_tensor.mean()
    video_tensor = torch.clamp(video_tensor, 0.0, 1.0)
    
    return video_tensor

def generate_high_quality_videos(
    checkpoint_path: str = "c3dgan/checkpoints/checkpoint_epoch_200.pth",
    class_mapping_path: str = "c3dgan/outputs/class_mapping.csv",
    output_dir: str = "c3dgan/generated_videos_high_quality",
    underrepresented_groups: list = None,
    n_samples_per_group: dict = None,
    device: str = "cuda",
    apply_sharpening: bool = True,
    apply_brightness: bool = True
):
    """Generate high-quality synthetic videos with post-processing."""
    
    print("Loading trained generator...")
    netG, class_to_idx = load_generator(checkpoint_path, device)
    
    if underrepresented_groups is None or n_samples_per_group is None:
        # Load from manifest
        import yaml
        with open("c3dgan/config.yaml", 'r') as f:
            cfg = yaml.safe_load(f)
        
        manifest_path = cfg['paths']['manifest_path']
        df = pd.read_csv(manifest_path)
        df['class_label'] = (
            df['view'].astype(str) + '_' + 
            df['sex'].astype(str) + '_' + 
            df['age_bin'].astype(str)
        )
        
        target_samples = cfg['augmentation']['target_samples_per_group']
        group_counts = df.groupby('class_label').size()
        underrepresented_groups = group_counts[group_counts < target_samples].index.tolist()
        
        n_samples_per_group = {}
        for group in underrepresented_groups:
            current_count = group_counts[group]
            needed = target_samples - current_count
            n_samples_per_group[group] = needed
    
    print(f"\nGenerating videos for {len(underrepresented_groups)} groups...")
    print(f"Total videos to generate: {sum(n_samples_per_group.values())}")
    print(f"Post-processing: Sharpening={apply_sharpening}, Brightness={apply_brightness}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_manifest = []
    total_generated = 0
    
    netG.eval()
    with torch.no_grad():
        for group_label in tqdm(underrepresented_groups, desc="Groups"):
            if group_label not in class_to_idx:
                print(f"Warning: {group_label} not in class mapping, skipping")
                continue
            
            class_idx = class_to_idx[group_label]
            n_samples = n_samples_per_group.get(group_label, 0)
            
            if n_samples == 0:
                continue
            
            print(f"\nGenerating {n_samples} videos for {group_label}...")
            
            for sample_idx in tqdm(range(n_samples), desc=f"  {group_label}", leave=False):
                # Generate random noise
                noise = torch.randn(1, 100, device=device)
                label = torch.tensor([class_idx], device=device)
                
                # Generate video
                fake_video = netG(noise, label)
                video_tensor = fake_video[0, 0].detach().cpu()  # (T, H, W)
                
                # Enhance brightness if needed
                if apply_brightness:
                    orig_mean = video_tensor.mean().item()
                    if orig_mean < 0.1:
                        video_tensor = enhance_brightness_and_contrast(
                            video_tensor, target_mean=0.3, contrast_factor=1.5
                        )
                
                # Convert to uint8
                video_tensor = torch.clamp(video_tensor, 0.0, 1.0)
                video_tensor = (video_tensor * 255.0).clamp(0, 255)
                fake_video = video_tensor.byte().numpy()
                
                # Apply sharpening
                if apply_sharpening:
                    fake_video = sharpen_video(fake_video)
                
                # Save video
                filename = f"{group_label}_synthetic_{sample_idx:04d}.mp4"
                filepath = output_dir / filename
                save_video(fake_video, str(filepath), fps=30)
                
                # Add to manifest
                generated_manifest.append({
                    'view': group_label.split('_')[0],
                    'sex': group_label.split('_')[1],
                    'age_bin': group_label.split('_')[2],
                    'class_label': group_label,
                    'file_name': filename,
                    'file_path': str(filepath),
                    'is_synthetic': True,
                    'source': 'C3DGAN_HQ',
                    'sharpened': apply_sharpening,
                    'brightness_enhanced': apply_brightness
                })
                
                total_generated += 1
                if total_generated % 100 == 0:
                    print(f"  Progress: {total_generated} videos generated...")
    
    # Save manifest
    manifest_df = pd.DataFrame(generated_manifest)
    manifest_path = output_dir / 'generated_manifest.csv'
    manifest_df.to_csv(manifest_path, index=False)
    
    print(f"\n=== GENERATION COMPLETE ===")
    print(f"Generated {len(generated_manifest)} high-quality synthetic videos")
    print(f"Output directory: {output_dir}")
    print(f"Manifest saved to: {manifest_path}")
    print(f"\nQuality enhancements applied:")
    print(f"  - Sharpening: {apply_sharpening}")
    print(f"  - Brightness enhancement: {apply_brightness}")
    
    return manifest_df

if __name__ == "__main__":
    print("=" * 60)
    print("HIGH-QUALITY VIDEO GENERATION")
    print("=" * 60)
    print("\nThis will generate ~5000 videos with:")
    print("  ✓ Sharpening enhancement")
    print("  ✓ Brightness/contrast adjustment")
    print("  ✓ Improved model architecture (no adaptive pooling)")
    print("\nStarting generation...\n")
    
    generate_high_quality_videos(
        apply_sharpening=True,
        apply_brightness=True
    )
