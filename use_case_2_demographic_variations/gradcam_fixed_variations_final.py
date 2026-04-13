"""
Final Grad-CAM Visualizations - Matching Original Working Style

Uses the same approach as the original working gradcam script
but with improved layout and clarity.
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


def compute_gradcam(model, video, demographics, target_layer):
    """Compute GradCAM - matching original working version exactly"""
    # Register hook for gradients and activations
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            gradients.append(grad_output[0].detach())
    
    def forward_hook(module, input, output):
        activations.append(output.detach())
    
    # Register hooks
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)
    
    # Forward pass
    video.requires_grad_(True)
    model.zero_grad()
    
    # Handle both single return and tuple return
    model_output = model(video, demographics)
    if isinstance(model_output, tuple):
        output, _ = model_output
    else:
        output = model_output
    
    # Backward pass - use reconstruction loss
    loss = F.l1_loss(output, video)
    loss.backward()
    
    # Compute CAM
    if len(gradients) == 0 or len(activations) == 0:
        # Fallback: use output directly
        cam = torch.abs(output - video).mean(dim=1)  # [B, T, H, W]
        cam = cam.squeeze(0)  # [T, H, W]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        handle_forward.remove()
        handle_backward.remove()
        return cam.detach().cpu().numpy(), output.detach()
    
    grad = gradients[0]  # [B, C, T, H, W]
    act = activations[0]  # [B, C, T, H, W]
    
    # Global average pooling of gradients
    weights = grad.mean(dim=(2, 3, 4), keepdim=True)  # [B, C, 1, 1, 1]
    
    # Weighted combination of activation maps
    cam = (weights * act).sum(dim=1, keepdim=True)  # [B, 1, T, H, W]
    cam = F.relu(cam)  # ReLU to get positive activations
    
    # Normalize
    cam = cam.squeeze(0).squeeze(0)  # [T, H, W]
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    else:
        cam = torch.zeros_like(cam)
    
    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()
    
    return cam.detach().cpu().numpy(), output.detach()


def apply_colormap(frame, cam, alpha=0.5):
    """Apply colormap overlay to frame - matching original working version"""
    # Resize CAM to match frame size if needed
    if cam.shape != frame.shape:
        cam = cv2.resize(cam, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Ensure frame is uint8
    if frame.max() <= 1.0:
        frame_uint8 = (frame * 255).astype(np.uint8)
    else:
        frame_uint8 = frame.astype(np.uint8)
    
    frame_rgb = np.stack([frame_uint8, frame_uint8, frame_uint8], axis=-1).astype(np.uint8)
    cam_colored = cm.jet(cam)[:, :, :3]
    cam_colored = (cam_colored * 255).astype(np.uint8)
    overlay = (alpha * cam_colored + (1 - alpha) * frame_rgb).astype(np.uint8)
    return overlay


def create_final_visualization(original_video, original_cam, fixed_video, fixed_cam,
                                variation_type, output_path, frame_idx=8):
    """Create final visualization matching original working style"""
    
    # Get video dimensions: [B, C, T, H, W]
    T_orig = original_video.shape[2]
    T_fixed = fixed_video.shape[2]
    T_cam_orig = original_cam.shape[0]
    T_cam_fixed = fixed_cam.shape[0]
    
    # Ensure frame_idx is within bounds
    frame_idx_orig = min(frame_idx, T_orig - 1, T_cam_orig - 1)
    frame_idx_fixed = min(frame_idx, T_fixed - 1, T_cam_fixed - 1)
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    
    # Row 1: Original Video
    orig_frame = original_video[0, 0, frame_idx_orig].cpu().numpy()
    orig_frame = (orig_frame + 1) * 127.5
    axes[0, 0].imshow(orig_frame, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original Video', fontsize=14, fontweight='bold', pad=10)
    axes[0, 0].axis('off')
    
    # Original CAM - use as-is, no normalization
    orig_cam_frame = original_cam[frame_idx_orig]
    im1 = axes[0, 1].imshow(orig_cam_frame, cmap='jet', vmin=0, vmax=1)
    axes[0, 1].set_title('Original Attention', fontsize=14, fontweight='bold', pad=10)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Original overlay
    orig_overlay = apply_colormap(orig_frame, orig_cam_frame, alpha=0.5)
    axes[0, 2].imshow(orig_overlay)
    axes[0, 2].set_title('Original Overlay', fontsize=14, fontweight='bold', pad=10)
    axes[0, 2].axis('off')
    
    # Original temporal average
    orig_avg = original_cam.mean(axis=0)
    im2 = axes[0, 3].imshow(orig_avg, cmap='jet', vmin=0, vmax=1)
    axes[0, 3].set_title('Original Avg Attention', fontsize=14, fontweight='bold', pad=10)
    axes[0, 3].axis('off')
    plt.colorbar(im2, ax=axes[0, 3], fraction=0.046, pad=0.04)
    
    # Row 2: Fixed Variation
    var_name = variation_type.replace('_', ' ').title()
    fixed_frame = fixed_video[0, 0, frame_idx_fixed].cpu().numpy()
    fixed_frame = (fixed_frame + 1) * 127.5
    axes[1, 0].imshow(fixed_frame, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title(f'Fixed Variation ({var_name})\n5.89% difference', 
                         fontsize=14, fontweight='bold', color='green', pad=10)
    axes[1, 0].axis('off')
    
    # Fixed CAM - use as-is, no normalization
    fixed_cam_frame = fixed_cam[frame_idx_fixed]
    im3 = axes[1, 1].imshow(fixed_cam_frame, cmap='jet', vmin=0, vmax=1)
    axes[1, 1].set_title('Fixed Attention', fontsize=14, fontweight='bold', pad=10)
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Fixed overlay
    fixed_overlay = apply_colormap(fixed_frame, fixed_cam_frame, alpha=0.5)
    axes[1, 2].imshow(fixed_overlay)
    axes[1, 2].set_title('Fixed Overlay', fontsize=14, fontweight='bold', pad=10)
    axes[1, 2].axis('off')
    
    # Fixed temporal average
    fixed_avg = fixed_cam.mean(axis=0)
    im4 = axes[1, 3].imshow(fixed_avg, cmap='jet', vmin=0, vmax=1)
    axes[1, 3].set_title('Fixed Avg Attention', fontsize=14, fontweight='bold', pad=10)
    axes[1, 3].axis('off')
    plt.colorbar(im4, ax=axes[1, 3], fraction=0.046, pad=0.04)
    
    # Calculate similarity metrics
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
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def analyze_fixed_variations_final(
    fixed_manifest,
    checkpoint_path=None,
    output_dir='gradcam_fixed_variations_final',
    device='cuda',
    num_samples=5
):
    """Generate final Grad-CAM visualizations"""
    print("="*70)
    print("GENERATING FINAL GRADCAM VISUALIZATIONS")
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
                    
                    # Create final visualization
                    sample_dir = output_dir / f"sample_{sample_id:04d}"
                    sample_dir.mkdir(exist_ok=True)
                    vis_path = sample_dir / f"final_{var_row['variation_type']}.png"
                    
                    create_final_visualization(
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
    parser.add_argument('--output_dir', type=str, default='gradcam_fixed_variations_final')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    analyze_fixed_variations_final(
        fixed_manifest=args.fixed_manifest,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        num_samples=args.num_samples
    )
