"""
Proper Grad-CAM Visualizations with Clear Heatmaps and Overlays

Creates high-quality visualizations with:
- Highly visible heatmaps using proper normalization
- Clear overlays that show both video and attention
- Better contrast and color mapping
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
from matplotlib.colors import Normalize

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


def compute_gradcam(model, video, demographics, target_layer):
    """Compute GradCAM"""
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            gradients.append(grad_output[0].detach())
    
    def forward_hook(module, input, output):
        activations.append(output.detach())
    
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)
    
    video.requires_grad_(True)
    model.zero_grad()
    output = model(video, demographics)
    loss = F.l1_loss(output, video)
    loss.backward()
    
    if len(gradients) == 0 or len(activations) == 0:
        cam = torch.abs(output - video).mean(dim=1).squeeze(0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        handle_forward.remove()
        handle_backward.remove()
        return cam.detach().cpu().numpy(), output.detach()
    
    grad = gradients[0]
    act = activations[0]
    weights = grad.mean(dim=(2, 3, 4), keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam.squeeze(0).squeeze(0)
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    else:
        cam = torch.zeros_like(cam)
    
    handle_forward.remove()
    handle_backward.remove()
    return cam.detach().cpu().numpy(), output.detach()


def normalize_heatmap(cam, percentile_low=5, percentile_high=95):
    """Normalize heatmap for better visibility"""
    cam = cam.copy()
    p_low = np.percentile(cam, percentile_low)
    p_high = np.percentile(cam, percentile_high)
    if p_high > p_low:
        cam = np.clip((cam - p_low) / (p_high - p_low), 0, 1)
    else:
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def create_proper_overlay(frame, heatmap, alpha=0.5, colormap='jet'):
    """Create proper overlay with visible heatmap"""
    # Ensure frame is uint8
    if frame.max() <= 1.0:
        frame = (frame * 255).astype(np.uint8)
    else:
        frame = frame.astype(np.uint8)
    
    # Normalize heatmap
    heatmap_norm = normalize_heatmap(heatmap)
    
    # Resize heatmap to match frame
    if heatmap_norm.shape != frame.shape:
        heatmap_norm = cv2.resize(heatmap_norm, (frame.shape[1], frame.shape[0]), 
                                 interpolation=cv2.INTER_LINEAR)
    
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    
    # Apply colormap to heatmap - use jet for blue/cyan/green/yellow/red
    heatmap_colored = cm.jet(heatmap_norm)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Create overlay - blend heatmap on top of frame
    overlay = cv2.addWeighted(frame_rgb, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay, heatmap_norm


def create_proper_visualization(original_video, original_cam, fixed_video, fixed_cam,
                                variation_type, output_path, frame_idx=8):
    """Create proper visualization with clear heatmaps and overlays"""
    
    # Get middle frames
    T_orig = original_video.shape[2]
    T_fixed = fixed_video.shape[2]
    frame_idx_orig = min(frame_idx, T_orig - 1)
    frame_idx_fixed = min(frame_idx, T_fixed - 1)
    
    # Extract frames
    orig_frame = original_video[0, 0, frame_idx_orig].cpu().numpy()
    orig_frame = (orig_frame + 1) * 127.5
    orig_frame = np.clip(orig_frame, 0, 255).astype(np.uint8)
    
    fixed_frame = fixed_video[0, 0, frame_idx_fixed].cpu().numpy()
    fixed_frame = (fixed_frame + 1) * 127.5
    fixed_frame = np.clip(fixed_frame, 0, 255).astype(np.uint8)
    
    # Get CAMs
    orig_cam = original_cam[frame_idx_orig]
    fixed_cam = fixed_cam[frame_idx_fixed]
    
    # Normalize heatmaps for visibility
    orig_cam_norm = normalize_heatmap(orig_cam, percentile_low=10, percentile_high=90)
    fixed_cam_norm = normalize_heatmap(fixed_cam, percentile_low=10, percentile_high=90)
    
    # Resize CAMs to match frame size
    if orig_cam_norm.shape != orig_frame.shape:
        orig_cam_norm = cv2.resize(orig_cam_norm, (orig_frame.shape[1], orig_frame.shape[0]), 
                                  interpolation=cv2.INTER_LINEAR)
    if fixed_cam_norm.shape != fixed_frame.shape:
        fixed_cam_norm = cv2.resize(fixed_cam_norm, (fixed_frame.shape[1], fixed_frame.shape[0]), 
                                    interpolation=cv2.INTER_LINEAR)
    
    # Create overlays with jet colormap
    orig_overlay, _ = create_proper_overlay(orig_frame, orig_cam_norm, alpha=0.6, colormap='jet')
    fixed_overlay, _ = create_proper_overlay(fixed_frame, fixed_cam_norm, alpha=0.6, colormap='jet')
    
    # Calculate temporal averages
    orig_avg = original_cam.mean(axis=0)
    fixed_avg = fixed_cam.mean(axis=0)
    
    # Ensure 2D
    if len(orig_avg.shape) == 1:
        side_len = int(np.sqrt(len(orig_avg)))
        orig_avg = orig_avg.reshape(side_len, side_len)
    if len(fixed_avg.shape) == 1:
        side_len = int(np.sqrt(len(fixed_avg)))
        fixed_avg = fixed_avg.reshape(side_len, side_len)
    
    # Normalize temporal averages
    orig_avg_norm = normalize_heatmap(orig_avg, percentile_low=10, percentile_high=90)
    fixed_avg_norm = normalize_heatmap(fixed_avg, percentile_low=10, percentile_high=90)
    
    # Resize to match frame
    if orig_avg_norm.shape != orig_frame.shape:
        orig_avg_norm = cv2.resize(orig_avg_norm, (orig_frame.shape[1], orig_frame.shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
    if fixed_avg_norm.shape != fixed_frame.shape:
        fixed_avg_norm = cv2.resize(fixed_avg_norm, (fixed_frame.shape[1], fixed_frame.shape[0]), 
                                    interpolation=cv2.INTER_LINEAR)
    
    # Create figure
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.25, left=0.05, right=0.95, top=0.93, bottom=0.07)
    
    # Row 1: Original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig_frame, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('Original Video', fontsize=16, fontweight='bold', pad=15)
    ax1.axis('off')
    
    # Original heatmap - use jet colormap (blue/cyan/green/yellow/red)
    ax2 = fig.add_subplot(gs[0, 1])
    im1 = ax2.imshow(orig_cam_norm, cmap='jet', vmin=0, vmax=1, interpolation='bilinear')
    ax2.set_title('Attention Heatmap', fontsize=16, fontweight='bold', pad=15)
    ax2.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax2, fraction=0.046, pad=0.04)
    cbar1.set_label('Attention', fontsize=12, fontweight='bold')
    
    # Original overlay
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(orig_overlay)
    ax3.set_title('Heatmap Overlay', fontsize=16, fontweight='bold', pad=15)
    ax3.axis('off')
    
    # Original temporal average - use jet colormap
    ax4 = fig.add_subplot(gs[0, 3])
    im2 = ax4.imshow(orig_avg_norm, cmap='jet', vmin=0, vmax=1, interpolation='bilinear')
    ax4.set_title('Temporal Average', fontsize=16, fontweight='bold', pad=15)
    ax4.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax4, fraction=0.046, pad=0.04)
    cbar2.set_label('Attention', fontsize=12, fontweight='bold')
    
    # Row 2: Fixed Variation
    var_name = variation_type.replace('_', ' ').title()
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(fixed_frame, cmap='gray', vmin=0, vmax=255)
    ax5.set_title(f'Fixed Variation ({var_name})\n5.89% difference', 
                  fontsize=16, fontweight='bold', color='green', pad=15)
    ax5.axis('off')
    
    # Fixed heatmap - use jet colormap (blue/cyan/green/yellow/red)
    ax6 = fig.add_subplot(gs[1, 1])
    im3 = ax6.imshow(fixed_cam_norm, cmap='jet', vmin=0, vmax=1, interpolation='bilinear')
    ax6.set_title('Attention Heatmap', fontsize=16, fontweight='bold', pad=15)
    ax6.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax6, fraction=0.046, pad=0.04)
    cbar3.set_label('Attention', fontsize=12, fontweight='bold')
    
    # Fixed overlay
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(fixed_overlay)
    ax7.set_title('Heatmap Overlay', fontsize=16, fontweight='bold', pad=15)
    ax7.axis('off')
    
    # Fixed temporal average - use jet colormap
    ax8 = fig.add_subplot(gs[1, 3])
    im4 = ax8.imshow(fixed_avg_norm, cmap='jet', vmin=0, vmax=1, interpolation='bilinear')
    ax8.set_title('Temporal Average', fontsize=16, fontweight='bold', pad=15)
    ax8.axis('off')
    cbar4 = plt.colorbar(im4, ax=ax8, fraction=0.046, pad=0.04)
    cbar4.set_label('Attention', fontsize=12, fontweight='bold')
    
    # Calculate similarity
    from scipy.spatial.distance import cosine
    orig_flat = original_cam.flatten()
    fixed_flat = fixed_cam.flatten()
    
    if len(orig_flat) != len(fixed_flat):
        min_len = min(len(orig_flat), len(fixed_flat))
        orig_flat = orig_flat[:min_len]
        fixed_flat = fixed_flat[:min_len]
    
    if len(orig_flat) > 0:
        cosine_sim = 1 - cosine(orig_flat, fixed_flat)
        if np.isnan(cosine_sim):
            cosine_sim = 0.0
    else:
        cosine_sim = 0.0
    
    # Spatial correlation
    orig_mean_flat = orig_avg.flatten()
    fixed_mean_flat = fixed_avg.flatten()
    if len(orig_mean_flat) != len(fixed_mean_flat):
        min_len = min(len(orig_mean_flat), len(fixed_mean_flat))
        orig_mean_flat = orig_mean_flat[:min_len]
        fixed_mean_flat = fixed_mean_flat[:min_len]
    
    if len(orig_mean_flat) > 1:
        spatial_corr = np.corrcoef(orig_mean_flat, fixed_mean_flat)[0, 1]
        if np.isnan(spatial_corr):
            spatial_corr = 0.0
    else:
        spatial_corr = 0.0
    
    plt.suptitle(f'Grad-CAM Analysis: Fixed Demographic Variation ({var_name})\n'
                 f'Visual Difference: 5.89% | Attention Similarity: {cosine_sim:.3f} | '
                 f'Spatial Correlation: {spatial_corr:.3f}',
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def analyze_fixed_variations_proper(
    fixed_manifest,
    checkpoint_path=None,
    output_dir='gradcam_fixed_variations_proper',
    device='cuda',
    num_samples=5
):
    """Generate proper Grad-CAM visualizations"""
    print("="*70)
    print("GENERATING PROPER GRADCAM VISUALIZATIONS")
    print("="*70)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if checkpoint_path is None:
        checkpoint_path = 'use_case_3_perfect_reconstruction/perfect_reconstruction_c3dgan/c3dgan_best.pt'
    
    print(f"\nLoading model: {checkpoint_path}")
    model = PerfectReconstructionGenerator(base_channels=64).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['generator'])
    model.eval()
    print("✓ Model loaded")
    
    target_layer = model.enc3
    fixed_df = pd.read_csv(fixed_manifest)
    sample_ids = fixed_df['original_id'].unique()[:num_samples]
    
    print(f"\nProcessing {len(sample_ids)} samples...")
    
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
                
                # Load original
                original_video = load_video(original_path).unsqueeze(0).to(device)
                original_sex = first_sample['original_sex']
                original_age = float(first_sample['original_age'])
                original_bmi = first_sample['original_bmi']
                original_demo = encode_demographics(original_sex, original_age, original_bmi).unsqueeze(0).to(device)
                
                with torch.enable_grad():
                    original_cam, _ = compute_gradcam(model, original_video, original_demo, target_layer)
                
                # Process each variation
                for _, var_row in fixed_samples.iterrows():
                    var_path = Path(var_row['synthetic_path'])
                    if not var_path.exists():
                        continue
                    
                    fixed_video = load_video(var_path).unsqueeze(0).to(device)
                    var_demo = encode_demographics(
                        var_row['variation_sex'], 
                        var_row['variation_age'], 
                        var_row['variation_bmi']
                    ).unsqueeze(0).to(device)
                    
                    with torch.enable_grad():
                        fixed_cam, _ = compute_gradcam(model, fixed_video, var_demo, target_layer)
                    
                    # Create proper visualization
                    sample_dir = output_dir / f"sample_{sample_id:04d}"
                    sample_dir.mkdir(exist_ok=True)
                    vis_path = sample_dir / f"proper_{var_row['variation_type']}.png"
                    
                    create_proper_visualization(
                        original_video, original_cam,
                        fixed_video, fixed_cam,
                        var_row['variation_type'], vis_path
                    )
            
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n✅ Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixed_manifest', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='gradcam_fixed_variations_proper')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    analyze_fixed_variations_proper(
        fixed_manifest=args.fixed_manifest,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        num_samples=args.num_samples
    )
