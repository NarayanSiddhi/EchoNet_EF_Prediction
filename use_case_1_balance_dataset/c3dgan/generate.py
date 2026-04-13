"""
Generate synthetic videos for underrepresented groups using trained C3DGAN.
"""

import argparse
import os
import torch
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import yaml
from tqdm import tqdm

from c3dgan.models import ConditionalC3DGenerator


def load_generator(checkpoint_path: str, config_path: str = None, device: str = "cuda"):
    """Load trained generator from checkpoint (uses config if provided)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    class_to_idx = checkpoint['class_to_idx']
    n_classes = len(class_to_idx)
    
    # Load model config from checkpoint or config file
    model_cfg = checkpoint.get('model_cfg', None)
    if config_path:
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            if cfg and 'model' in cfg:
                model_cfg = cfg['model']
        except Exception:
            pass

    # Fallback defaults (for older checkpoints)
    if model_cfg is None:
        model_cfg = {
            'nz': 100,
            'ngf': 64,
            'nc': 1,
            'video_length': 96,
            'video_size': 128
        }

    netG = ConditionalC3DGenerator(
        nz=model_cfg.get('nz', 100),
        ngf=model_cfg.get('ngf', 64),
        nc=model_cfg.get('nc', 1),
        n_classes=n_classes,
        video_length=model_cfg.get('video_length', 96),
        video_size=model_cfg.get('video_size', 128)
    ).to(device)
    
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netG.eval()
    
    return netG, class_to_idx


def generate_videos_for_groups(
    checkpoint_path: str,
    class_mapping_path: str,
    output_dir: str,
    underrepresented_groups: list,
    n_samples_per_group: dict,
    device: str = "cuda",
    config_path: str = None
):
    """
    Generate synthetic videos for specific underrepresented groups.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        class_mapping_path: Path to class mapping CSV
        output_dir: Directory to save generated videos
        underrepresented_groups: List of class labels to generate for
        n_samples_per_group: Number of videos to generate per group
        device: Device to use
    """
    
    # Load generator
    print("Loading trained generator...")
    netG, class_to_idx = load_generator(checkpoint_path, config_path=config_path, device=device)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Load class mapping
    class_mapping = pd.read_csv(class_mapping_path)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating videos for {len(underrepresented_groups)} groups...")
    print(f"Samples per group: {n_samples_per_group}")
    
    generated_manifest = []
    
    for group_label in tqdm(underrepresented_groups, desc="Generating"):
        if group_label not in class_to_idx:
            print(f"Warning: {group_label} not in class mapping, skipping")
            continue
        
        class_idx = class_to_idx[group_label]
        n_samples = n_samples_per_group.get(group_label, 0)
        
        if n_samples == 0:
            continue
        
        # Generate videos for this group
        with torch.no_grad():
            for sample_idx in range(n_samples):
                # Generate random noise
                noise = torch.randn(1, 100, device=device)
                label = torch.tensor([class_idx], device=device)
                
                # Generate video
                fake_video = netG(noise, label)
                
                # Convert to numpy using PyTorch operations to avoid NumPy compatibility issues
                video_tensor = fake_video[0, 0].detach().cpu()  # (T, H, W)
                
                # Use PyTorch operations to avoid NumPy compatibility issues
                # Clamp to [0, 1] range using torch.clamp
                video_tensor = torch.clamp(video_tensor, 0.0, 1.0)
                
                # Scale to [0, 255] and convert to uint8 using torch operations
                video_tensor = (video_tensor * 255.0).clamp(0, 255)
                
                # Convert to numpy uint8
                fake_video = video_tensor.byte().numpy()  # .byte() converts to uint8
                
                # Save video
                filename = f"{group_label}_synthetic_{sample_idx:04d}.mp4"
                filepath = output_dir / filename
                save_video(fake_video, str(filepath), fps=30)
                
                # Add to manifest
                parts = group_label.split('_')
                view = parts[0] if len(parts) > 0 else None
                sex = parts[1] if len(parts) > 1 else None
                age_bin = parts[2] if len(parts) > 2 else None
                bmi_bin = parts[3] if len(parts) > 3 else None
                generated_manifest.append({
                    'view': view,
                    'sex': sex,
                    'age_bin': age_bin,
                    'bmi_bin': bmi_bin,
                    'class_label': group_label,
                    'file_name': filename,
                    'file_path': str(filepath),
                    'is_synthetic': True,
                    'source': 'C3DGAN'
                })
    
    # Save manifest
    manifest_df = pd.DataFrame(generated_manifest)
    manifest_path = output_dir / 'generated_manifest.csv'
    manifest_df.to_csv(manifest_path, index=False)
    
    print(f"\n=== GENERATION COMPLETE ===")
    print(f"Generated {len(generated_manifest)} synthetic videos")
    print(f"Manifest saved to: {manifest_path}")
    
    return manifest_df


def save_video(frames: np.ndarray, output_path: str, fps: int = 30):
    """Save video frames to file using imageio (more reliable than OpenCV 4.11.0).
    
    Args:
        frames: numpy array of shape (T, H, W) with dtype uint8, values in [0, 255]
        output_path: path to save the video file
        fps: frames per second for the video
    """
    try:
        import imageio
    except ImportError:
        raise ImportError("imageio is required for video saving. Install with: pip install imageio imageio-ffmpeg")
    
    # Validate input
    if frames is None or frames.size == 0:
        raise ValueError("Frames array is empty or None")
    
    # Ensure frames are in correct format: (T, H, W) uint8 [0, 255]
    if frames.dtype != np.uint8:
        # Use torch operations to avoid NumPy compatibility issues
        import torch
        frames_tensor = torch.from_numpy(frames).float()
        frames_tensor = torch.clamp(frames_tensor, 0, 255)
        frames = frames_tensor.byte().numpy()  # Convert to uint8 via torch
    
    # Ensure 3D array (T, H, W)
    if len(frames.shape) != 3:
        raise ValueError(f"Expected 3D array (T, H, W), got shape {frames.shape}")
    
    # Validate dimensions
    if frames.shape[0] == 0:
        raise ValueError("Video has 0 frames")
    if frames.shape[1] == 0 or frames.shape[2] == 0:
        raise ValueError(f"Invalid frame dimensions: {frames.shape[1]}x{frames.shape[2]}")
    
    # Use imageio instead of OpenCV (OpenCV 4.11.0 has bugs)
    # Convert grayscale frames to RGB for imageio
    T, H, W = frames.shape
    frames_rgb = np.zeros((T, H, W, 3), dtype=np.uint8)
    for t in range(T):
        frames_rgb[t, :, :, 0] = frames[t]  # R
        frames_rgb[t, :, :, 1] = frames[t]  # G
        frames_rgb[t, :, :, 2] = frames[t]  # B
    
    # Save using imageio
    try:
        imageio.mimwrite(str(output_path), frames_rgb, fps=fps, codec='libx264', quality=8)
    except Exception as e:
        # Fallback: try without codec specification
        try:
            imageio.mimwrite(str(output_path), frames_rgb, fps=fps)
        except Exception as e2:
            raise RuntimeError(f"Failed to save video with imageio: {e2}") from e2
    
    # Verify file was created and has reasonable size
    output_path_obj = Path(output_path)
    if not output_path_obj.exists():
        raise RuntimeError(f"Video file was not created: {output_path}")
    
    file_size = output_path_obj.stat().st_size
    if file_size < 1000:  # Less than 1KB is suspicious
        raise RuntimeError(f"Video file is too small ({file_size} bytes) - likely corrupted")
    
    return T


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic videos for underrepresented groups")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--class_mapping", type=str, default="c3dgan/outputs/class_mapping.csv")
    parser.add_argument("--output_dir", type=str, default="c3dgan/generated_videos")
    parser.add_argument("--config", type=str, default="c3dgan/config.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Load config to get underrepresented groups
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load manifest to identify underrepresented groups
    manifest_path = cfg['paths']['manifest_path']
    df = pd.read_csv(manifest_path)
    df['class_label'] = (
        df['view'].astype(str) + '_' + 
        df['sex'].astype(str) + '_' + 
        df['age_bin'].astype(str)
    )
    
    target_samples = cfg['augmentation']['target_samples_per_group']
    group_counts = df.groupby('class_label').size()
    underrepresented = group_counts[group_counts < target_samples].index.tolist()
    
    # Calculate samples needed per group
    n_samples_per_group = {}
    for group in underrepresented:
        current_count = group_counts[group]
        needed = target_samples - current_count
        n_samples_per_group[group] = needed
    
    print(f"Found {len(underrepresented)} underrepresented groups")
    
    # Generate videos
    generate_videos_for_groups(
        checkpoint_path=args.checkpoint,
        class_mapping_path=args.class_mapping,
        output_dir=args.output_dir,
        underrepresented_groups=underrepresented,
        n_samples_per_group=n_samples_per_group,
        device=args.device,
        config_path=args.config
    )

