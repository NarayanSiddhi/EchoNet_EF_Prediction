"""
Cardiac-Specific Grad-CAM using EF Prediction Model

Uses the EF prediction model (PTEFNet) to compute Grad-CAM, which highlights
regions important for ejection fraction estimation - i.e., cardiac structures
like the left ventricle, rather than reconstruction artifacts.
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
import yaml

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'ef_prediction'))
from models.pt_efnet import PTEFNet

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


def compute_cardiac_gradcam(ef_model, video, target_layer, target_size=(64, 64), flip_h=False, flip_v=False):
    """
    Compute Grad-CAM using EF prediction model - cardiac-specific attention
    
    This highlights regions important for EF estimation (cardiac structures)
    rather than reconstruction artifacts.
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
    ef_model.zero_grad()
    
    # Forward pass - predict EF
    ef_pred = ef_model(video)
    
    # Backward pass - gradient w.r.t. EF prediction
    # This highlights regions important for EF estimation
    ef_pred.backward(torch.ones_like(ef_pred))
    
    if len(gradients) == 0 or len(activations) == 0:
        # Fallback: use output directly
        cam = torch.abs(video).mean(dim=(1, 2))
        cam = cam.squeeze(0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        handle_forward.remove()
        handle_backward.remove()
        # Upsample to target size
        cam_upsampled = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
        return cam_upsampled.squeeze().detach().cpu().numpy(), ef_pred.detach()
    
    # Get activations and gradients
    # For ResNet18 in EF model, video is reshaped to [B*T, 3, H, W]
    # So activations are [B*T, C, H, W]
    grad = gradients[0]
    act = activations[0]
    
    B, C, T, H, W = video.shape
    
    # Reshape from [B*T, C, H, W] back to [B, T, C, H, W]
    if grad.dim() == 4 and grad.shape[0] == B * T:
        grad = grad.view(B, T, grad.shape[1], grad.shape[2], grad.shape[3])
        act = act.view(B, T, act.shape[1], act.shape[2], act.shape[3])
    elif grad.dim() == 5:
        pass  # Already [B, T, C, H, W]
    else:
        # Fallback
        cam = torch.abs(video).mean(dim=(1, 2))
        cam = cam.squeeze(0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        handle_forward.remove()
        handle_backward.remove()
        cam_upsampled = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
        return cam_upsampled.squeeze().detach().cpu().numpy(), ef_pred.detach()
    
    # Compute CAM for each frame
    cams = []
    for t in range(T):
        grad_t = grad[:, t]  # [B, C, H, W]
        act_t = act[:, t]     # [B, C, H, W]
        
        # Global average pooling of gradients
        weights = grad_t.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        
        # Weighted combination
        cam_t = (weights * act_t).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam_t = F.relu(cam_t)
        
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
        cam_t_np = cam_t_upsampled.squeeze().detach().cpu().numpy()
        
        # Apply flips if needed to align with video orientation
        if flip_h:
            cam_t_np = np.fliplr(cam_t_np)
        if flip_v:
            cam_t_np = np.flipud(cam_t_np)
        
        cams.append(cam_t_np)
    
    handle_forward.remove()
    handle_backward.remove()
    
    return np.array(cams), ef_pred.detach()


def apply_colormap(frame, cam, alpha=0.5):
    """Apply colormap overlay to frame"""
    if cam.shape != frame.shape:
        cam = cv2.resize(cam, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    if frame.max() <= 1.0:
        frame_uint8 = (frame * 255).astype(np.uint8)
    else:
        frame_uint8 = frame.astype(np.uint8)
    
    frame_rgb = np.stack([frame_uint8, frame_uint8, frame_uint8], axis=-1).astype(np.uint8)
    cam_colored = cm.jet(cam)[:, :, :3]
    cam_colored = (cam_colored * 255).astype(np.uint8)
    overlay = (alpha * cam_colored + (1 - alpha) * frame_rgb).astype(np.uint8)
    return overlay


def create_cardiac_visualization(original_video, original_cam, fixed_video, fixed_cam,
                                 original_ef, fixed_ef, variation_type, output_path, frame_idx=8):
    """Create visualization with cardiac-specific Grad-CAM"""
    
    T_orig = original_video.shape[2]
    T_fixed = fixed_video.shape[2]
    T_cam_orig = original_cam.shape[0] if len(original_cam.shape) > 2 else 1
    T_cam_fixed = fixed_cam.shape[0] if len(fixed_cam.shape) > 2 else 1
    
    frame_idx_orig = min(frame_idx, T_orig - 1, T_cam_orig - 1)
    frame_idx_fixed = min(frame_idx, T_fixed - 1, T_cam_fixed - 1)
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    
    # Row 1: Original
    orig_frame = original_video[0, 0, frame_idx_orig].cpu().numpy()
    orig_frame = (orig_frame + 1) * 127.5
    axes[0, 0].imshow(orig_frame, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title(f'Original Video\n(EF: {original_ef:.2f})', fontsize=14, fontweight='bold', pad=10)
    axes[0, 0].axis('off')
    
    # Get CAM for middle frame
    if len(original_cam.shape) > 2:
        orig_cam_frame = original_cam[frame_idx_orig]
    else:
        orig_cam_frame = original_cam
    
    im1 = axes[0, 1].imshow(orig_cam_frame, cmap='jet', vmin=0, vmax=1)
    axes[0, 1].set_title('Cardiac Attention\n(EF-focused)', fontsize=14, fontweight='bold', pad=10)
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    orig_overlay = apply_colormap(orig_frame, orig_cam_frame, alpha=0.5)
    axes[0, 2].imshow(orig_overlay)
    axes[0, 2].set_title('Cardiac Overlay', fontsize=14, fontweight='bold', pad=10)
    axes[0, 2].axis('off')
    
    # Temporal average
    if len(original_cam.shape) > 2:
        orig_avg = original_cam.mean(axis=0)
    else:
        orig_avg = original_cam
    im2 = axes[0, 3].imshow(orig_avg, cmap='jet', vmin=0, vmax=1)
    axes[0, 3].set_title('Temporal Avg\n(All Frames)', fontsize=14, fontweight='bold', pad=10)
    axes[0, 3].axis('off')
    plt.colorbar(im2, ax=axes[0, 3], fraction=0.046, pad=0.04)
    
    # Row 2: Fixed Variation
    var_name = variation_type.replace('_', ' ').title()
    fixed_frame = fixed_video[0, 0, frame_idx_fixed].cpu().numpy()
    fixed_frame = (fixed_frame + 1) * 127.5
    axes[1, 0].imshow(fixed_frame, cmap='gray', vmin=0, vmax=255)
    axes[1, 0].set_title(f'Fixed Variation ({var_name})\n(EF: {fixed_ef:.2f})', 
                         fontsize=14, fontweight='bold', color='green', pad=10)
    axes[1, 0].axis('off')
    
    if len(fixed_cam.shape) > 2:
        fixed_cam_frame = fixed_cam[frame_idx_fixed]
    else:
        fixed_cam_frame = fixed_cam
    
    im3 = axes[1, 1].imshow(fixed_cam_frame, cmap='jet', vmin=0, vmax=1)
    axes[1, 1].set_title('Cardiac Attention\n(EF-focused)', fontsize=14, fontweight='bold', pad=10)
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    fixed_overlay = apply_colormap(fixed_frame, fixed_cam_frame, alpha=0.5)
    axes[1, 2].imshow(fixed_overlay)
    axes[1, 2].set_title('Cardiac Overlay', fontsize=14, fontweight='bold', pad=10)
    axes[1, 2].axis('off')
    
    if len(fixed_cam.shape) > 2:
        fixed_avg = fixed_cam.mean(axis=0)
    else:
        fixed_avg = fixed_cam
    im4 = axes[1, 3].imshow(fixed_avg, cmap='jet', vmin=0, vmax=1)
    axes[1, 3].set_title('Temporal Avg\n(All Frames)', fontsize=14, fontweight='bold', pad=10)
    axes[1, 3].axis('off')
    plt.colorbar(im4, ax=axes[1, 3], fraction=0.046, pad=0.04)
    
    # Calculate similarity
    from scipy.spatial.distance import cosine
    orig_flat = orig_avg.flatten()
    fixed_flat = fixed_avg.flatten()
    
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
    
    spatial_corr = np.corrcoef(orig_flat, fixed_flat)[0, 1] if len(orig_flat) > 1 else 0.0
    if np.isnan(spatial_corr):
        spatial_corr = 0.0
    
    plt.suptitle(f'Cardiac-Specific Grad-CAM: Fixed Demographic Variation ({var_name})\n'
                 f'EF: {original_ef:.2f} → {fixed_ef:.2f} | Attention Similarity: {cosine_sim:.3f} | '
                 f'Spatial Correlation: {spatial_corr:.3f}',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def analyze_cardiac_gradcam(
    fixed_manifest,
    ef_model_path,
    output_dir='gradcam_cardiac_specific',
    device='cuda',
    num_samples=5,
    flip_horizontal=False,
    flip_vertical=False
):
    """Generate cardiac-specific Grad-CAM visualizations"""
    print("="*70)
    print("CARDIAC-SPECIFIC GRADCAM USING EF PREDICTION MODEL")
    print("="*70)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load EF prediction model
    print(f"\nLoading EF prediction model: {ef_model_path}")
    ef_model = PTEFNet(hidden_dim=256).to(device)
    checkpoint = torch.load(ef_model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        ef_model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        ef_model.load_state_dict(checkpoint['state_dict'])
    else:
        ef_model.load_state_dict(checkpoint)
    # Keep in eval mode but allow gradients for Grad-CAM
    ef_model.eval()
    for param in ef_model.parameters():
        param.requires_grad = True
    print("✓ EF model loaded")
    
    # Use ResNet18 layer4 (last conv layer) for Grad-CAM
    # This captures high-level cardiac features
    # ResNet18 structure: conv layers -> layer1 -> layer2 -> layer3 -> layer4 -> AdaptiveAvgPool
    # We want layer4 (last conv block) before pooling
    if hasattr(ef_model.cnn, 'layer4'):
        target_layer = ef_model.cnn.layer4[-1]  # Last conv in layer4
    else:
        # Fallback: find last conv layer
        for module in reversed(list(ef_model.cnn.modules())):
            if isinstance(module, nn.Conv2d):
                target_layer = module
                break
        else:
            target_layer = ef_model.cnn[-2]  # Second to last (before pooling)
    print(f"Target layer: {target_layer}")
    
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
                
                # Load original video
                original_video = load_video(original_path).unsqueeze(0).to(device)
                
                # Compute EF and cardiac Grad-CAM for original
                # Set to train mode temporarily for LSTM backward pass
                ef_model.train()
                with torch.enable_grad():
                    original_cam, original_ef_pred = compute_cardiac_gradcam(
                        ef_model, original_video, target_layer, target_size=(64, 64),
                        flip_h=flip_horizontal, flip_v=flip_vertical
                    )
                ef_model.eval()
                original_ef = original_ef_pred.item() * 100  # Convert to percentage
                
                # Process each variation
                for _, var_row in fixed_samples.iterrows():
                    var_path = Path(var_row['synthetic_path'])
                    if not var_path.exists():
                        continue
                    
                    fixed_video = load_video(var_path).unsqueeze(0).to(device)
                    
                    # Compute EF and cardiac Grad-CAM for fixed variation
                    ef_model.train()
                    with torch.enable_grad():
                        fixed_cam, fixed_ef_pred = compute_cardiac_gradcam(
                            ef_model, fixed_video, target_layer, target_size=(64, 64),
                            flip_h=flip_horizontal, flip_v=flip_vertical
                        )
                    ef_model.eval()
                    fixed_ef = fixed_ef_pred.item() * 100
                    
                    # Create visualization
                    sample_dir = output_dir / f"sample_{sample_id:04d}"
                    sample_dir.mkdir(exist_ok=True)
                    vis_path = sample_dir / f"cardiac_{var_row['variation_type']}.png"
                    
                    create_cardiac_visualization(
                        original_video, original_cam,
                        fixed_video, fixed_cam,
                        original_ef, fixed_ef,
                        var_row['variation_type'], vis_path
                    )
            
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n✅ Cardiac-specific visualizations saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixed_manifest', type=str, required=True)
    parser.add_argument('--ef_model', type=str, 
                       default='ef_prediction/checkpoints/best.pth')
    parser.add_argument('--output_dir', type=str, default='gradcam_cardiac_specific')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--flip_h', action='store_true', 
                       help='Flip attention map horizontally (if on wrong side)')
    parser.add_argument('--flip_v', action='store_true',
                       help='Flip attention map vertically (if upside down)')
    
    args = parser.parse_args()
    
    analyze_cardiac_gradcam(
        fixed_manifest=args.fixed_manifest,
        ef_model_path=args.ef_model,
        output_dir=args.output_dir,
        device=args.device,
        num_samples=args.num_samples,
        flip_horizontal=args.flip_h,
        flip_vertical=args.flip_v
    )
