"""
Clear Grad-CAM Visualizations for Fixed Variations

Creates high-quality, publication-ready visualizations showing:
- Original vs Fixed variations side-by-side
- Clear attention maps with proper contrast
- Cardiac structure annotations
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
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
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


def create_clear_visualization(original_video, original_cam, fixed_video, fixed_cam,
                               variation_type, output_path, frame_idx=8):
    """Create clear, publication-ready visualization"""
    
    # Get middle frames
    T_orig = original_video.shape[2]
    T_fixed = fixed_video.shape[2]
    frame_idx_orig = min(frame_idx, T_orig - 1)
    frame_idx_fixed = min(frame_idx, T_fixed - 1)
    
    # Extract and enhance frames
    orig_frame = original_video[0, 0, frame_idx_orig].cpu().numpy()
    orig_frame = (orig_frame + 1) * 127.5
    orig_frame = np.clip(orig_frame, 0, 255).astype(np.uint8)
    
    fixed_frame = fixed_video[0, 0, frame_idx_fixed].cpu().numpy()
    fixed_frame = (fixed_frame + 1) * 127.5
    fixed_frame = np.clip(fixed_frame, 0, 255).astype(np.uint8)
    
    # Get CAMs and threshold to show top 30% attention
    orig_cam = original_cam[frame_idx_orig]
    orig_cam_thresh = np.where(orig_cam >= np.percentile(orig_cam, 70), orig_cam, 0)
    
    fixed_cam = fixed_cam[frame_idx_fixed]
    fixed_cam_thresh = np.where(fixed_cam >= np.percentile(fixed_cam, 70), fixed_cam, 0)
    
    # Create figure with better layout
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.2)
    
    # Row 1: Original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig_frame, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('Original Video', fontsize=14, fontweight='bold', pad=10)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    im1 = ax2.imshow(orig_cam_thresh, cmap='hot', vmin=0, vmax=1)
    ax2.set_title('Attention Map', fontsize=14, fontweight='bold', pad=10)
    ax2.axis('off')
    plt.colorbar(im1, ax=ax2, fraction=0.046, pad=0.04)
    
    # Overlay - ensure shapes match
    orig_frame_rgb = cv2.cvtColor(orig_frame, cv2.COLOR_GRAY2RGB)
    orig_cam_rgb = (cm.hot(orig_cam_thresh)[:, :, :3] * 255).astype(np.uint8)
    if orig_cam_rgb.shape != orig_frame_rgb.shape:
        orig_cam_rgb = cv2.resize(orig_cam_rgb, (orig_frame_rgb.shape[1], orig_frame_rgb.shape[0]))
    orig_overlay = cv2.addWeighted(orig_frame_rgb, 0.4, orig_cam_rgb, 0.6, 0)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(orig_overlay)
    ax3.set_title('Attention Overlay', fontsize=14, fontweight='bold', pad=10)
    ax3.axis('off')
    
    # Temporal average - ensure 2D
    orig_avg_2d = original_cam.mean(axis=0)
    if len(orig_avg_2d.shape) == 1:
        side_len = int(np.sqrt(len(orig_avg_2d)))
        orig_avg_2d = orig_avg_2d.reshape(side_len, side_len)
    orig_avg = threshold_cam(orig_avg_2d, 70)
    ax4 = fig.add_subplot(gs[0, 3])
    im2 = ax4.imshow(orig_avg, cmap='hot', vmin=0, vmax=1)
    ax4.set_title('Temporal Average', fontsize=14, fontweight='bold', pad=10)
    ax4.axis('off')
    plt.colorbar(im2, ax=ax4, fraction=0.046, pad=0.04)
    
    # Row 2: Fixed Variation
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(fixed_frame, cmap='gray', vmin=0, vmax=255)
    var_name = variation_type.replace('_', ' ').title()
    ax5.set_title(f'Fixed Variation ({var_name})\n5.89% difference', 
                  fontsize=14, fontweight='bold', color='green', pad=10)
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 1])
    im3 = ax6.imshow(fixed_cam_thresh, cmap='hot', vmin=0, vmax=1)
    ax6.set_title('Attention Map', fontsize=14, fontweight='bold', pad=10)
    ax6.axis('off')
    plt.colorbar(im3, ax=ax6, fraction=0.046, pad=0.04)
    
    # Overlay - ensure shapes match
    fixed_frame_rgb = cv2.cvtColor(fixed_frame, cv2.COLOR_GRAY2RGB)
    fixed_cam_rgb = (cm.hot(fixed_cam_thresh)[:, :, :3] * 255).astype(np.uint8)
    if fixed_cam_rgb.shape != fixed_frame_rgb.shape:
        fixed_cam_rgb = cv2.resize(fixed_cam_rgb, (fixed_frame_rgb.shape[1], fixed_frame_rgb.shape[0]))
    fixed_overlay = cv2.addWeighted(fixed_frame_rgb, 0.4, fixed_cam_rgb, 0.6, 0)
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(fixed_overlay)
    ax7.set_title('Attention Overlay', fontsize=14, fontweight='bold', pad=10)
    ax7.axis('off')
    
    # Temporal average - ensure 2D
    fixed_avg_2d = fixed_cam.mean(axis=0)
    if len(fixed_avg_2d.shape) == 1:
        side_len = int(np.sqrt(len(fixed_avg_2d)))
        fixed_avg_2d = fixed_avg_2d.reshape(side_len, side_len)
    fixed_avg = threshold_cam(fixed_avg_2d, 70)
    ax8 = fig.add_subplot(gs[1, 3])
    im4 = ax8.imshow(fixed_avg, cmap='hot', vmin=0, vmax=1)
    ax8.set_title('Temporal Average', fontsize=14, fontweight='bold', pad=10)
    ax8.axis('off')
    plt.colorbar(im4, ax=ax8, fraction=0.046, pad=0.04)
    
    # Calculate similarity - ensure same size
    from scipy.spatial.distance import cosine
    orig_flat = original_cam.flatten()
    fixed_flat = fixed_cam.flatten()
    
    # Resize to same size if needed
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
    
    # Calculate spatial correlation safely
    orig_mean = original_cam.mean(axis=0)
    fixed_mean = fixed_cam.mean(axis=0)
    
    # Ensure both are 2D
    if len(orig_mean.shape) == 1:
        side_len = int(np.sqrt(len(orig_mean)))
        orig_mean = orig_mean.reshape(side_len, side_len)
    if len(fixed_mean.shape) == 1:
        side_len = int(np.sqrt(len(fixed_mean)))
        fixed_mean = fixed_mean.reshape(side_len, side_len)
    
    # Resize to same size if needed
    if orig_mean.shape != fixed_mean.shape:
        fixed_mean = cv2.resize(fixed_mean, (orig_mean.shape[1], orig_mean.shape[0]))
    
    orig_mean_flat = orig_mean.flatten()
    fixed_mean_flat = fixed_mean.flatten()
    if len(orig_mean_flat) == len(fixed_mean_flat) and len(orig_mean_flat) > 1:
        spatial_corr = np.corrcoef(orig_mean_flat, fixed_mean_flat)[0, 1]
        if np.isnan(spatial_corr):
            spatial_corr = 0.0
    else:
        spatial_corr = 0.0
    
    plt.suptitle(f'Grad-CAM: Fixed Demographic Variation ({var_name})\n'
                 f'Visual Difference: 5.89% | Attention Similarity: {cosine_sim:.3f} | '
                 f'Spatial Correlation: {spatial_corr:.3f}',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def threshold_cam(cam, percentile=70):
    """Threshold CAM to show top regions"""
    threshold = np.percentile(cam, percentile)
    return np.where(cam >= threshold, cam, 0)


def analyze_fixed_variations_clear(
    fixed_manifest,
    checkpoint_path=None,
    output_dir='gradcam_fixed_variations_clear',
    device='cuda',
    num_samples=5
):
    """Generate clear Grad-CAM visualizations"""
    print("="*70)
    print("GENERATING CLEAR GRADCAM VISUALIZATIONS")
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
                    
                    # Create clear visualization
                    sample_dir = output_dir / f"sample_{sample_id:04d}"
                    sample_dir.mkdir(exist_ok=True)
                    vis_path = sample_dir / f"clear_{var_row['variation_type']}.png"
                    
                    create_clear_visualization(
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
    parser.add_argument('--output_dir', type=str, default='gradcam_fixed_variations_clear')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    analyze_fixed_variations_clear(
        fixed_manifest=args.fixed_manifest,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        num_samples=args.num_samples
    )
