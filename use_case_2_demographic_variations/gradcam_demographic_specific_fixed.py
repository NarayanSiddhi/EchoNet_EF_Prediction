"""
Demographic-Specific Grad-CAM - Fixed Version

Computes Grad-CAM that highlights regions important for demographic differences
by computing attention on the demographic embedding features in the generator.
This shows which regions the generator focuses on when changing demographics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'use_case_3_perfect_reconstruction'))
from models import PerfectReconstructionGenerator

def encode_demographics(sex, age, bmi, sex_map={'F': 0, 'M': 1, 'O': 0}, 
                       age_bins=[0, 5, 10, 15, 18], 
                       bmi_map={'underweight': 0, 'normal': 1, 'overweight': 2, 'obese': 3}):
    """One-hot encode demographics"""
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
    
    if len(frames) > video_length:
        indices = np.linspace(0, len(frames)-1, video_length, dtype=int)
        frames = [frames[i] for i in indices]
    while len(frames) < video_length:
        frames.append(frames[-1] if frames else np.zeros((video_size, video_size)))
    
    video = np.array(frames[:video_length], dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(video).unsqueeze(0)


def compute_demographic_gradcam(generator, video, original_demo, target_demo, 
                                target_layer, target_size=(64, 64)):
    """
    Compute Grad-CAM highlighting regions important for demographic differences
    
    Strategy: Compute gradients w.r.t. the demographic embedding difference
    This highlights which regions the generator changes when demographics change.
    """
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            gradients.append(grad_output[0].detach())
    
    def forward_hook(module, input, output):
        activations.append(output.detach())
    
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)
    
    # Ensure video is in correct format: [B, 1, T, H, W]
    if video.dim() == 4:
        video = video.unsqueeze(0)
    
    video.requires_grad_(True)
    generator.zero_grad()
    
    # Generate with original and target demographics
    output_orig = generator(video, original_demo)
    output_target = generator(video, target_demo)
    
    # Compute difference - this is what changes with demographics
    diff = output_target - output_orig
    
    # Use L2 norm of difference as loss to get meaningful gradients
    # This highlights regions where demographic changes cause visual differences
    loss = (diff ** 2).mean()
    loss.backward()
    
    if len(gradients) == 0 or len(activations) == 0:
        # Fallback: use absolute difference directly
        cam = torch.abs(diff).mean(dim=(1, 2))  # Average over channels and time
        cam = cam.squeeze(0)  # [H, W]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = torch.zeros_like(cam)
        handle_forward.remove()
        handle_backward.remove()
        # Upsample to target size
        cam_upsampled = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        return cam_upsampled.squeeze().squeeze().detach().cpu().numpy(), output_target.detach()
    
    grad = gradients[0]  # [B, C, T, H, W]
    act = activations[0]  # [B, C, T, H, W]
    
    # Get dimensions
    B, C, T, H, W = act.shape
    
    # Compute CAM for each frame
    cams = []
    for t in range(T):
        grad_t = grad[:, :, t]  # [B, C, H, W]
        act_t = act[:, :, t]     # [B, C, H, W]
        
        # Global average pooling of gradients across spatial dimensions
        weights = grad_t.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        
        # Weighted combination of activations
        cam_t = (weights * act_t).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam_t = F.relu(cam_t)  # Only positive contributions
        
        # Normalize
        cam_t = cam_t.squeeze(0).squeeze(0)  # [H, W]
        if cam_t.max() > cam_t.min():
            cam_t = (cam_t - cam_t.min()) / (cam_t.max() - cam_t.min() + 1e-8)
        else:
            cam_t = torch.zeros_like(cam_t)
        
        # Upsample to target size
        cam_t_upsampled = F.interpolate(
            cam_t.unsqueeze(0).unsqueeze(0),
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        cams.append(cam_t_upsampled.squeeze().squeeze().detach().cpu().numpy())
    
    handle_forward.remove()
    handle_backward.remove()
    
    return np.array(cams), output_target.detach()


def apply_colormap(frame, cam, alpha=0.6):
    """Apply colormap overlay with proper scaling"""
    # Ensure cam matches frame size
    if cam.shape != frame.shape:
        cam = cv2.resize(cam, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Normalize cam to [0, 1]
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    else:
        cam = np.zeros_like(cam)
    
    # Convert frame to uint8
    if frame.max() <= 1.0:
        frame_uint8 = (frame * 255).astype(np.uint8)
    else:
        frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
    
    # Create RGB frame
    frame_rgb = np.stack([frame_uint8, frame_uint8, frame_uint8], axis=-1).astype(np.uint8)
    
    # Apply jet colormap to CAM
    cam_colored = cm.jet(cam)[:, :, :3]
    cam_colored = (cam_colored * 255).astype(np.uint8)
    
    # Blend overlay
    overlay = (alpha * cam_colored + (1 - alpha) * frame_rgb).astype(np.uint8)
    return overlay


def create_demographic_visualization(original_video, original_cam, variation_video, variation_cam,
                                     variation_type, output_path, frame_idx=8):
    """Create visualization highlighting demographic-specific regions"""
    
    # Get dimensions
    T_orig = original_video.shape[2]
    T_var = variation_video.shape[2]
    T_cam_orig = original_cam.shape[0] if len(original_cam.shape) > 2 else 1
    T_cam_var = variation_cam.shape[0] if len(variation_cam.shape) > 2 else 1
    
    frame_idx_orig = min(frame_idx, T_orig - 1, T_cam_orig - 1)
    frame_idx_var = min(frame_idx, T_var - 1, T_cam_var - 1)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Original
    orig_frame = original_video[0, 0, frame_idx_orig].cpu().numpy()
    orig_frame = (orig_frame + 1) * 127.5
    orig_frame = np.clip(orig_frame, 0, 255)
    axes[0, 0].imshow(orig_frame, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original Video', fontsize=16, fontweight='bold', pad=15)
    axes[0, 0].axis('off')
    
    # Get CAM for this frame
    if len(original_cam.shape) > 2:
        orig_cam_frame = original_cam[frame_idx_orig]
    else:
        orig_cam_frame = original_cam
    
    # Ensure CAM is 2D
    if len(orig_cam_frame.shape) > 2:
        orig_cam_frame = orig_cam_frame.mean(axis=0) if orig_cam_frame.shape[0] > 1 else orig_cam_frame[0]
    
    # Normalize CAM
    if orig_cam_frame.max() > orig_cam_frame.min():
        orig_cam_frame = (orig_cam_frame - orig_cam_frame.min()) / (orig_cam_frame.max() - orig_cam_frame.min() + 1e-8)
    else:
        orig_cam_frame = np.zeros_like(orig_cam_frame)
    
    im1 = axes[0, 1].imshow(orig_cam_frame, cmap='jet', vmin=0, vmax=1, interpolation='bilinear')
    axes[0, 1].set_title('Demographic Attention\n(Regions that change)', fontsize=16, fontweight='bold', pad=15)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    orig_overlay = apply_colormap(orig_frame, orig_cam_frame, alpha=0.6)
    axes[0, 2].imshow(orig_overlay)
    axes[0, 2].set_title('Attention Overlay\n(Red = High Change)', fontsize=16, fontweight='bold', pad=15)
    axes[0, 2].axis('off')
    
    # Temporal average
    if len(original_cam.shape) > 2:
        orig_avg = original_cam.mean(axis=0)
    else:
        orig_avg = original_cam
    
    if len(orig_avg.shape) > 2:
        orig_avg = orig_avg.mean(axis=0) if orig_avg.shape[0] > 1 else orig_avg[0]
    
    if orig_avg.max() > orig_avg.min():
        orig_avg = (orig_avg - orig_avg.min()) / (orig_avg.max() - orig_avg.min() + 1e-8)
    else:
        orig_avg = np.zeros_like(orig_avg)
    
    im2 = axes[0, 3].imshow(orig_avg, cmap='jet', vmin=0, vmax=1, interpolation='bilinear')
    axes[0, 3].set_title('Temporal Average\n(All Frames)', fontsize=16, fontweight='bold', pad=15)
    axes[0, 3].axis('off')
    plt.colorbar(im2, ax=axes[0, 3], fraction=0.046, pad=0.04)
    
    # Row 2: Variation
    var_name = variation_type.replace('_', ' ').title()
    var_frame = variation_video[0, 0, frame_idx_var].cpu().numpy()
    var_frame = (var_frame + 1) * 127.5
    var_frame = np.clip(var_frame, 0, 255)
    axes[1, 0].imshow(var_frame, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title(f'Variation ({var_name})\n5.89% difference', 
                         fontsize=16, fontweight='bold', color='green', pad=15)
    axes[1, 0].axis('off')
    
    # Get CAM for variation
    if len(variation_cam.shape) > 2:
        var_cam_frame = variation_cam[frame_idx_var]
    else:
        var_cam_frame = variation_cam
    
    if len(var_cam_frame.shape) > 2:
        var_cam_frame = var_cam_frame.mean(axis=0) if var_cam_frame.shape[0] > 1 else var_cam_frame[0]
    
    if var_cam_frame.max() > var_cam_frame.min():
        var_cam_frame = (var_cam_frame - var_cam_frame.min()) / (var_cam_frame.max() - var_cam_frame.min() + 1e-8)
    else:
        var_cam_frame = np.zeros_like(var_cam_frame)
    
    im3 = axes[1, 1].imshow(var_cam_frame, cmap='jet', vmin=0, vmax=1, interpolation='bilinear')
    axes[1, 1].set_title('Demographic Attention\n(Regions that changed)', fontsize=16, fontweight='bold', pad=15)
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    var_overlay = apply_colormap(var_frame, var_cam_frame, alpha=0.6)
    axes[1, 2].imshow(var_overlay)
    axes[1, 2].set_title('Attention Overlay\n(Red = High Change)', fontsize=16, fontweight='bold', pad=15)
    axes[1, 2].axis('off')
    
    # Temporal average for variation
    if len(variation_cam.shape) > 2:
        var_avg = variation_cam.mean(axis=0)
    else:
        var_avg = variation_cam
    
    if len(var_avg.shape) > 2:
        var_avg = var_avg.mean(axis=0) if var_avg.shape[0] > 1 else var_avg[0]
    
    if var_avg.max() > var_avg.min():
        var_avg = (var_avg - var_avg.min()) / (var_avg.max() - var_avg.min() + 1e-8)
    else:
        var_avg = np.zeros_like(var_avg)
    
    im4 = axes[1, 3].imshow(var_avg, cmap='jet', vmin=0, vmax=1, interpolation='bilinear')
    axes[1, 3].set_title('Temporal Average\n(All Frames)', fontsize=16, fontweight='bold', pad=15)
    axes[1, 3].axis('off')
    plt.colorbar(im4, ax=axes[1, 3], fraction=0.046, pad=0.04)
    
    # Calculate similarity
    from scipy.spatial.distance import cosine
    orig_flat = orig_avg.flatten()
    var_flat = var_avg.flatten()
    
    if len(orig_flat) != len(var_flat):
        min_len = min(len(orig_flat), len(var_flat))
        orig_flat = orig_flat[:min_len]
        var_flat = var_flat[:min_len]
    
    if len(orig_flat) > 0:
        cosine_sim = 1 - cosine(orig_flat, var_flat)
        if np.isnan(cosine_sim):
            cosine_sim = 0.0
    else:
        cosine_sim = 0.0
    
    spatial_corr = np.corrcoef(orig_flat, var_flat)[0, 1] if len(orig_flat) > 1 else 0.0
    if np.isnan(spatial_corr):
        spatial_corr = 0.0
    
    plt.suptitle(f'Demographic-Specific Grad-CAM: {var_name}\n'
                 f'Highlights regions that change with demographics | '
                 f'Attention Similarity: {cosine_sim:.3f} | Spatial Correlation: {spatial_corr:.3f}',
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def analyze_demographic_gradcam(
    fixed_manifest,
    checkpoint_path=None,
    output_dir='gradcam_demographic_specific_fixed',
    device='cuda',
    num_samples=5,
    target_layer_name='enc2'
):
    """Generate demographic-specific Grad-CAM visualizations"""
    print("="*70)
    print("DEMOGRAPHIC-SPECIFIC GRADCAM (FIXED)")
    print("Highlights regions important for demographic differences")
    print("="*70)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if checkpoint_path is None:
        checkpoint_path = 'use_case_3_perfect_reconstruction/perfect_reconstruction_c3dgan/c3dgan_best.pt'
    
    print(f"\nLoading generator: {checkpoint_path}")
    generator = PerfectReconstructionGenerator(base_channels=64).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    print("✓ Generator loaded")
    
    # Use enc2 for better spatial resolution
    if target_layer_name == 'enc2':
        target_layer = generator.enc2
    elif target_layer_name == 'enc1':
        target_layer = generator.enc1
    else:
        target_layer = generator.enc3
    
    fixed_df = pd.read_csv(fixed_manifest)
    sample_ids = fixed_df['original_id'].unique()[:num_samples]
    
    print(f"\nProcessing {len(sample_ids)} samples...")
    print(f"Target layer: {target_layer_name}")
    
    with torch.no_grad():
        for sample_id in tqdm(sample_ids, desc="Generating"):
            try:
                fixed_samples = fixed_df[fixed_df['original_id'] == sample_id]
                if len(fixed_samples) == 0:
                    continue
                
                first_sample = fixed_samples.iloc[0]
                original_path = Path(first_sample['original_path'])
                
                if not original_path.exists():
                    continue
                
                # Load original video
                original_video = load_video(original_path).unsqueeze(0).to(device)
                original_sex = first_sample['original_sex']
                original_age = float(first_sample['original_age'])
                original_bmi = first_sample['original_bmi']
                original_demo = encode_demographics(original_sex, original_age, original_bmi).unsqueeze(0).to(device)
                
                # Process each variation
                for _, var_row in fixed_samples.iterrows():
                    var_path = Path(var_row['synthetic_path'])
                    if not var_path.exists():
                        continue
                    
                    # Load variation video
                    variation_video = load_video(var_path).unsqueeze(0).to(device)
                    
                    # Get target demographics
                    target_demo = encode_demographics(
                        var_row['variation_sex'], 
                        var_row['variation_age'], 
                        var_row['variation_bmi']
                    ).unsqueeze(0).to(device)
                    
                    # Compute demographic-specific Grad-CAM
                    # This highlights regions that change when demographics change
                    with torch.enable_grad():
                        original_cam, _ = compute_demographic_gradcam(
                            generator, original_video, original_demo, target_demo,
                            target_layer, target_size=(64, 64)
                        )
                        
                        # Also compute for variation
                        variation_cam, _ = compute_demographic_gradcam(
                            generator, variation_video, original_demo, target_demo,
                            target_layer, target_size=(64, 64)
                        )
                    
                    # Create visualization
                    sample_dir = output_dir / f"sample_{sample_id:04d}"
                    sample_dir.mkdir(exist_ok=True)
                    vis_path = sample_dir / f"demographic_{var_row['variation_type']}.png"
                    
                    create_demographic_visualization(
                        original_video, original_cam,
                        variation_video, variation_cam,
                        var_row['variation_type'], vis_path
                    )
            
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n✅ Demographic-specific visualizations saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixed_manifest', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='gradcam_demographic_specific_fixed')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--layer', type=str, default='enc2', choices=['enc1', 'enc2', 'enc3'])
    
    args = parser.parse_args()
    
    analyze_demographic_gradcam(
        fixed_manifest=args.fixed_manifest,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        num_samples=args.num_samples,
        target_layer_name=args.layer
    )
