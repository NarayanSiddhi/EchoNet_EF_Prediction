"""
Create 2x4 table visualization:
Row 1: Original video, Age-varied, Sex-varied, BMI-varied
Row 2: Grad-CAM overlay (original), (age), (sex), (BMI)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add paths in correct order
sys.path.insert(0, str(Path(__file__).parent / 'use_case_3_perfect_reconstruction'))
from models import PerfectReconstructionGenerator

sys.path.insert(0, str(Path(__file__).parent / 'use_case_2_demographic_variations'))
from gradcam_cardiac_specific import load_video, compute_cardiac_gradcam

def apply_colormap(frame, cam, alpha=0.6):
    """Apply colormap overlay"""
    import matplotlib.cm as cm
    if cam.shape != frame.shape:
        cam = cv2.resize(cam, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    if frame.max() <= 1.0:
        frame_uint8 = (frame * 255).astype(np.uint8)
    else:
        frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
    frame_rgb = np.stack([frame_uint8, frame_uint8, frame_uint8], axis=-1).astype(np.uint8)
    cam_colored = cm.jet(cam)[:, :, :3]
    cam_colored = (cam_colored * 255).astype(np.uint8)
    overlay = (alpha * cam_colored + (1 - alpha) * frame_rgb).astype(np.uint8)
    return overlay

import torch
import torch.nn.functional as F
import yaml

def load_ef_model():
    """Load EF prediction model for cardiac-specific Grad-CAM"""
    sys.path.insert(0, str(Path(__file__).parent / 'ef_prediction'))
    from models.pt_efnet import PTEFNet
    
    config_path = Path(__file__).parent / 'ef_prediction' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    ef_model = PTEFNet(config['model'])
    checkpoint_path = Path(__file__).parent / 'ef_prediction' / 'checkpoints' / 'best_model.pt'
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        ef_model.load_state_dict(checkpoint['model_state_dict'])
    
    ef_model.eval()
    return ef_model

def create_table_visualization(sample_id, output_path, device='cuda'):
    """Create 2x4 table with videos and Grad-CAM overlays"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load models
    print("Loading models...")
    ef_model = load_ef_model().to(device)
    
    generator_path = Path('use_case_3_perfect_reconstruction/perfect_reconstruction_c3dgan/c3dgan_best.pt')
    generator = PerfectReconstructionGenerator(base_channels=64).to(device)
    checkpoint = torch.load(generator_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    
    # Load videos
    print(f"Loading videos for sample {sample_id}...")
    
    # Original video
    orig_path = Path(f"paper_gradcam_collection/videos/demographic_variations/originals/sample_{sample_id:04d}_original.mp4")
    if not orig_path.exists():
        print(f"Original video not found: {orig_path}")
        return
    
    # Variation videos
    age_path = Path(f"paper_gradcam_collection/videos/demographic_variations/variations/video_{sample_id:04d}_var1_age_variation.mp4")
    sex_path = Path(f"paper_gradcam_collection/videos/demographic_variations/variations/video_{sample_id:04d}_var2_sex_variation.mp4")
    bmi_path = Path(f"paper_gradcam_collection/videos/demographic_variations/variations/video_{sample_id:04d}_var3_bmi_variation.mp4")
    
    if not all([age_path.exists(), sex_path.exists(), bmi_path.exists()]):
        print("Some variation videos not found")
        return
    
    # Load videos
    orig_video = load_video(orig_path).unsqueeze(0).to(device)
    age_video = load_video(age_path).unsqueeze(0).to(device)
    sex_video = load_video(sex_path).unsqueeze(0).to(device)
    bmi_video = load_video(bmi_path).unsqueeze(0).to(device)
    
    # Get middle frame (frame 8)
    frame_idx = 8
    orig_frame = orig_video[0, 0, frame_idx].cpu().numpy()
    age_frame = age_video[0, 0, frame_idx].cpu().numpy()
    sex_frame = sex_video[0, 0, frame_idx].cpu().numpy()
    bmi_frame = bmi_video[0, 0, frame_idx].cpu().numpy()
    
    # Convert to display format
    orig_frame_disp = (orig_frame + 1) * 127.5
    age_frame_disp = (age_frame + 1) * 127.5
    sex_frame_disp = (sex_frame + 1) * 127.5
    bmi_frame_disp = (bmi_frame + 1) * 127.5
    
    # Compute Grad-CAM for each video
    print("Computing Grad-CAM...")
    target_layer = ef_model.cnn[7]  # layer4 of ResNet18
    
    with torch.no_grad():
        # Original Grad-CAM
        orig_cam, _ = compute_cardiac_gradcam(ef_model, orig_video, target_layer, target_size=(64, 64))
        orig_cam_frame = orig_cam[frame_idx] if len(orig_cam.shape) > 2 else orig_cam
        
        # Age variation Grad-CAM
        age_cam, _ = compute_cardiac_gradcam(ef_model, age_video, target_layer, target_size=(64, 64))
        age_cam_frame = age_cam[frame_idx] if len(age_cam.shape) > 2 else age_cam
        
        # Sex variation Grad-CAM
        sex_cam, _ = compute_cardiac_gradcam(ef_model, sex_video, target_layer, target_size=(64, 64))
        sex_cam_frame = sex_cam[frame_idx] if len(sex_cam.shape) > 2 else sex_cam
        
        # BMI variation Grad-CAM
        bmi_cam, _ = compute_cardiac_gradcam(ef_model, bmi_video, target_layer, target_size=(64, 64))
        bmi_cam_frame = bmi_cam[frame_idx] if len(bmi_cam.shape) > 2 else bmi_cam
    
    # Create overlays
    orig_overlay = apply_colormap(orig_frame_disp, orig_cam_frame, alpha=0.6)
    age_overlay = apply_colormap(age_frame_disp, age_cam_frame, alpha=0.6)
    sex_overlay = apply_colormap(sex_frame_disp, sex_cam_frame, alpha=0.6)
    bmi_overlay = apply_colormap(bmi_frame_disp, bmi_cam_frame, alpha=0.6)
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Videos
    axes[0, 0].imshow(orig_frame_disp, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Original video', fontsize=16, fontweight='bold', pad=10)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(age_frame_disp, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('Age-varied', fontsize=16, fontweight='bold', pad=10)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(sex_frame_disp, cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title('Sex-varied', fontsize=16, fontweight='bold', pad=10)
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(bmi_frame_disp, cmap='gray', vmin=0, vmax=255)
    axes[0, 3].set_title('BMI-varied', fontsize=16, fontweight='bold', pad=10)
    axes[0, 3].axis('off')
    
    # Row 2: Grad-CAM Overlays
    axes[1, 0].imshow(orig_overlay)
    axes[1, 0].set_title('Grad-CAM overlay\n(original)', fontsize=16, fontweight='bold', pad=10)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(age_overlay)
    axes[1, 1].set_title('Grad-CAM overlay\n(age)', fontsize=16, fontweight='bold', pad=10)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(sex_overlay)
    axes[1, 2].set_title('Grad-CAM overlay\n(sex)', fontsize=16, fontweight='bold', pad=10)
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(bmi_overlay)
    axes[1, 3].set_title('Grad-CAM overlay\n(BMI)', fontsize=16, fontweight='bold', pad=10)
    axes[1, 3].axis('off')
    
    plt.suptitle(f'Demographic Variations and Grad-CAM Analysis - Sample {sample_id:04d}',
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ Saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_id', type=int, default=0, help='Sample ID (0-4)')
    parser.add_argument('--output_dir', type=str, default='table_gradcam_visualizations')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"table_gradcam_sample_{args.sample_id:04d}.png"
    
    create_table_visualization(args.sample_id, output_path, args.device)
