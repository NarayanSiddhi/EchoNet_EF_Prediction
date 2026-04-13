"""
Generate proper GradCAM visualizations using EXACT same approach as working code.
No modifications - just following the proven implementation.
"""
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import yaml
import sys
import os

# Add ef_prediction to path
sys.path.insert(0, str(Path(__file__).parent))

from ef_prediction.dataset_demographics import DualVideoEFDataset
from ef_prediction.demographics_utils import row_to_demo_vector
from ef_prediction.models.pt_efnet_real import PTEFNetReal


class GradCAM:
    """Exact copy of working GradCAM implementation"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate(self, video_tensor, demo_vec):
        self.model.zero_grad()

        ef_pred, _ = self.model(video_tensor, demo_vec)
        ef_pred.backward(torch.ones_like(ef_pred))
        
        gradients = self.gradients
        activations = self.activations
        
        # activations shape: (B*T, C, H, W)
        # reshape back to (B, T, C, H, W)
        B = video_tensor.size(0)
        T = video_tensor.size(2)
        
        activations = activations.view(B, T, activations.size(1), activations.size(2), activations.size(3))
        gradients = gradients.view(B, T, gradients.size(1), gradients.size(2), gradients.size(3))
        
        # Take middle frame
        t = T // 2
        act = activations[:, t]
        grad = gradients[:, t]
        
        weights = torch.mean(grad, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * act, dim=1)
        
        cam = torch.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()
        
        cam = cv2.resize(cam, (128, 128))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def create_comparison_visualization(real_video_tensor, real_cam, syn_video_tensor, syn_cam, 
                                   sample_id, ef, output_dir, title_prefix=""):
    """Save individual images instead of grid - CLEAR visible frames and subtle heatmaps"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract middle frame
    t = real_video_tensor.size(2) // 2
    real_frame = real_video_tensor[0, 0, t].detach().cpu().numpy()
    syn_frame = syn_video_tensor[0, 0, t].detach().cpu().numpy()
    
    # Enhance frame contrast for visibility
    def enhance_frame(frame):
        frame = np.clip(frame, 0, 1)
        p2, p98 = np.percentile(frame, [2, 98])
        if p98 > p2:
            frame = (frame - p2) / (p98 - p2)
            frame = np.clip(frame, 0, 1)
        frame = np.power(frame, 0.7)
        return frame
    
    real_frame_enhanced = enhance_frame(real_frame)
    syn_frame_enhanced = enhance_frame(syn_frame)
    
    # Threshold heatmaps to show top regions - use adaptive thresholding
    def threshold_heatmap(cam, percentile=60):
        # Use lower percentile to show more regions, but still focus on important areas
        threshold = np.percentile(cam, percentile)
        cam_thresholded = np.where(cam >= threshold, cam, 0)
        if cam_thresholded.max() > 0:
            # Normalize after thresholding to enhance contrast
            cam_thresholded = (cam_thresholded - cam_thresholded.min()) / (cam_thresholded.max() - cam_thresholded.min() + 1e-8)
            # Apply power function to enhance visibility of important regions
            cam_thresholded = np.power(cam_thresholded, 0.8)
        return cam_thresholded
    
    real_cam_thresh = threshold_heatmap(real_cam, percentile=60)
    syn_cam_thresh = threshold_heatmap(syn_cam, percentile=60)
    
    # Save individual images
    base_name = f"sample_{sample_id:04d}"
    
    # Real video frame
    plt.figure(figsize=(8, 8))
    plt.imshow(real_frame_enhanced, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.title(f'Real Video Frame - Sample {sample_id} - EF: {ef:.1f}%', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_real_frame.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Real heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(real_cam_thresh, cmap='jet', vmin=0, vmax=1)
    plt.axis('off')
    plt.title(f'Real GradCAM Heatmap - Sample {sample_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_real_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Real overlay
    heatmap_real = cv2.applyColorMap(np.uint8(255 * real_cam_thresh), cv2.COLORMAP_JET)
    frame_rgb_real = (real_frame_enhanced * 255).astype(np.uint8)
    frame_rgb_real = np.stack([frame_rgb_real]*3, axis=-1)
    overlay_real = cv2.addWeighted(frame_rgb_real, 0.85, heatmap_real, 0.15, 0)
    cv2.imwrite(str(output_dir / f"{base_name}_real_overlay.png"), overlay_real)
    
    # Synthetic video frame
    plt.figure(figsize=(8, 8))
    plt.imshow(syn_frame_enhanced, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.title(f'Synthetic Video Frame - Sample {sample_id} - EF: {ef:.1f}%', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_synthetic_frame.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Synthetic heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(syn_cam_thresh, cmap='jet', vmin=0, vmax=1)
    plt.axis('off')
    plt.title(f'Synthetic GradCAM Heatmap - Sample {sample_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_synthetic_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Synthetic overlay
    heatmap_syn = cv2.applyColorMap(np.uint8(255 * syn_cam_thresh), cv2.COLORMAP_JET)
    frame_rgb_syn = (syn_frame_enhanced * 255).astype(np.uint8)
    frame_rgb_syn = np.stack([frame_rgb_syn]*3, axis=-1)
    overlay_syn = cv2.addWeighted(frame_rgb_syn, 0.85, heatmap_syn, 0.15, 0)
    cv2.imwrite(str(output_dir / f"{base_name}_synthetic_overlay.png"), overlay_syn)


def create_variations_visualization(real_video_tensor, real_cam, var_video_tensors, var_cams, 
                                   var_types, sample_id, ef, output_dir):
    """Save individual images for 1 real + 3 variations - CLEAR visible frames and subtle heatmaps"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract middle frame
    t = real_video_tensor.size(2) // 2
    real_frame = real_video_tensor[0, 0, t].detach().cpu().numpy()
    
    # Enhance frame contrast for visibility
    def enhance_frame(frame):
        frame = np.clip(frame, 0, 1)
        p2, p98 = np.percentile(frame, [2, 98])
        if p98 > p2:
            frame = (frame - p2) / (p98 - p2)
            frame = np.clip(frame, 0, 1)
        frame = np.power(frame, 0.7)
        return frame
    
    # Threshold heatmaps to show top regions - use adaptive thresholding
    def threshold_heatmap(cam, percentile=60):
        # Use lower percentile to show more regions, but still focus on important areas
        threshold = np.percentile(cam, percentile)
        cam_thresholded = np.where(cam >= threshold, cam, 0)
        if cam_thresholded.max() > 0:
            # Normalize after thresholding to enhance contrast
            cam_thresholded = (cam_thresholded - cam_thresholded.min()) / (cam_thresholded.max() - cam_thresholded.min() + 1e-8)
            # Apply power function to enhance visibility of important regions
            cam_thresholded = np.power(cam_thresholded, 0.8)
        return cam_thresholded
    
    base_name = f"sample_{sample_id:04d}"
    
    # Real video
    real_frame_enhanced = enhance_frame(real_frame)
    real_cam_thresh = threshold_heatmap(real_cam, percentile=60)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(real_frame_enhanced, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.title(f'Real Video - Sample {sample_id} - EF: {ef:.1f}%', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_real_frame.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(real_cam_thresh, cmap='jet', vmin=0, vmax=1)
    plt.axis('off')
    plt.title(f'Real Heatmap - Sample {sample_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_real_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    heatmap_real = cv2.applyColorMap(np.uint8(255 * real_cam_thresh), cv2.COLORMAP_JET)
    frame_rgb_real = (real_frame_enhanced * 255).astype(np.uint8)
    frame_rgb_real = np.stack([frame_rgb_real]*3, axis=-1)
    overlay_real = cv2.addWeighted(frame_rgb_real, 0.85, heatmap_real, 0.15, 0)
    cv2.imwrite(str(output_dir / f"{base_name}_real_overlay.png"), overlay_real)
    
    # Variations
    for var_type, var_video_tensor, var_cam in zip(var_types, var_video_tensors, var_cams):
        var_frame = var_video_tensor[0, 0, t].detach().cpu().numpy()
        var_frame_enhanced = enhance_frame(var_frame)
        var_cam_thresh = threshold_heatmap(var_cam, percentile=60)
        
        var_name = var_type.replace("_", "_")
        
        plt.figure(figsize=(8, 8))
        plt.imshow(var_frame_enhanced, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title(f'{var_type.replace("_", " ").title()} - Sample {sample_id} - EF: {ef:.1f}%', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f"{base_name}_{var_name}_frame.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(8, 8))
        plt.imshow(var_cam_thresh, cmap='jet', vmin=0, vmax=1)
        plt.axis('off')
        plt.title(f'{var_type.replace("_", " ").title()} Heatmap - Sample {sample_id}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f"{base_name}_{var_name}_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        heatmap_var = cv2.applyColorMap(np.uint8(255 * var_cam_thresh), cv2.COLORMAP_JET)
        frame_rgb_var = (var_frame_enhanced * 255).astype(np.uint8)
        frame_rgb_var = np.stack([frame_rgb_var]*3, axis=-1)
        overlay_var = cv2.addWeighted(frame_rgb_var, 0.85, heatmap_var, 0.15, 0)
        cv2.imwrite(str(output_dir / f"{base_name}_{var_name}_overlay.png"), overlay_var)


def load_video_tensor(video_path, video_length=32, video_size=128):
    """Load video EXACTLY like DualVideoEFDataset.load_video"""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Use absolute path for cv2
    abs_path = str(video_path.resolve())
    cap = cv2.VideoCapture(abs_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video (may be corrupted): {abs_path}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (video_size, video_size))
        frames.append(frame)
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"Empty video: {video_path}")
    
    frames = np.stack(frames)
    
    # EXACT same sampling as dataset
    if len(frames) >= video_length:
        idx = np.linspace(0, len(frames) - 1, video_length).astype(int)
        frames = frames[idx]
    else:
        pad = video_length - len(frames)
        frames = np.pad(frames, ((0, pad), (0, 0), (0, 0)), mode="edge")
    
    # EXACT same normalization and tensor creation as dataset
    frames = frames.astype(np.float32) / 255.0
    frames = torch.from_numpy(frames).unsqueeze(0)  # (1, T, H, W)
    
    return frames


def generate_perfect_copy_gradcam(top_n=5):
    """Generate GradCAM for top N perfect synthetic copies"""
    print("="*70)
    print("GENERATING GRADCAM FOR PERFECT SYNTHETIC COPIES")
    print("="*70)
    
    with open("ef_prediction/config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading real-only model...")
    backbone = cfg["model"].get("backbone", "resnet34")
    model = PTEFNetReal(backbone=backbone).to(device)
    checkpoint_path = "ef_prediction/checkpoints/real/best.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.train()
    print("✓ Model loaded")
    
    target_layer = model.cnn[7]
    gradcam = GradCAM(model, target_layer)
    
    # Use the main manifest which has all samples, sorted by SSIM
    manifest_path = "perfect_synthetic_copies/perfect_copies_manifest.csv"
    if not os.path.exists(manifest_path):
        manifest_path = "perfect_synthetic_copies/perfect_copies_val.csv"
    
    df = pd.read_csv(manifest_path)
    # Filter out any rows with missing SSIM or invalid values
    df = df.dropna(subset=['SSIM'])
    df = df[df['SSIM'] > 0]
    # Sort by SSIM descending to get the BEST quality samples
    df = df.sort_values('SSIM', ascending=False)
    
    # Resolve paths like dataset does
    video_root = cfg['data']['original_video_dir']
    
    def resolve_path(root, path):
        if os.path.isabs(path):
            return Path(path)
        if root is None:
            return Path(path)
        if str(path).startswith(str(root)):
            return Path(path)
        return Path(root) / path
    
    # Filter to only samples where both files exist AND can be opened
    valid_samples = []
    for idx, row in df.iterrows():
        real_path = resolve_path(video_root, row['original_path'])
        syn_path = Path(row['synthetic_path'])  # Synthetic paths relative to project root
        
        if real_path.exists() and syn_path.exists():
            # Test if files can actually be opened
            try:
                cap_real = cv2.VideoCapture(str(real_path.resolve()))
                cap_syn = cv2.VideoCapture(str(syn_path.resolve()))
                real_ok = cap_real.isOpened()
                syn_ok = cap_syn.isOpened()
                cap_real.release()
                cap_syn.release()
                
                if real_ok and syn_ok:
                    # Update paths in row to resolved paths
                    row_resolved = row.copy()
                    row_resolved['original_path'] = str(real_path.resolve())
                    row_resolved['synthetic_path'] = str(syn_path.resolve())
                    valid_samples.append(row_resolved)
                    if len(valid_samples) >= top_n:
                        break
            except:
                continue  # Skip if can't test
    
    if len(valid_samples) == 0:
        print("ERROR: No valid samples found with both real and synthetic videos!")
        return []
    
    df = pd.DataFrame(valid_samples)
    
    print(f"\nSelected top {top_n} samples by SSIM:")
    for idx, row in df.iterrows():
        print(f"  Sample {row['original_id']}: SSIM={row['SSIM']:.4f}, EF={row['EF']:.1f}%")
    
    output_dir = Path("best_gradcam_results/perfect_copies")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        sample_id = int(row['original_id'])
        
        try:
            # Paths are already resolved in the dataframe
            real_path = Path(row['original_path'])
            syn_path = Path(row['synthetic_path'])
            
            if not real_path.exists() or not syn_path.exists():
                print(f"  Skipping sample {sample_id}: files not found")
                print(f"    Real: {real_path} (exists: {real_path.exists()})")
                print(f"    Syn: {syn_path} (exists: {syn_path.exists()})")
                continue
            
            real_video = load_video_tensor(real_path, cfg['model']['video_length'], cfg['model']['video_size']).unsqueeze(0).to(device)
            syn_video = load_video_tensor(syn_path, cfg['model']['video_length'], cfg['model']['video_size']).unsqueeze(0).to(device)
            demo_b = torch.from_numpy(row_to_demo_vector(row)).float().unsqueeze(0).to(device)

            # Generate separate GradCAMs
            with torch.enable_grad():
                real_cam = gradcam.generate(real_video, demo_b)
                syn_cam = gradcam.generate(syn_video, demo_b)
            
            create_comparison_visualization(
                real_video, real_cam,
                syn_video, syn_cam,
                sample_id, row['EF'],
                output_dir,
                title_prefix="Perfect Copy - "
            )
            
            results.append({
                'sample_id': sample_id,
                'output_dir': str(output_dir),
                'ssim': row['SSIM'],
                'ef': row['EF']
            })
            
            print(f"  ✓ Saved individual images for sample {sample_id} to {output_dir}")
            
        except Exception as e:
            print(f"  ✗ Error processing sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✓ Generated {len(results)} perfect copy GradCAM visualizations")
    print(f"  Output directory: {output_dir.absolute()}")
    
    return results


def generate_variations_gradcam(top_n=5):
    """Generate GradCAM for top N demographic variations"""
    print("\n" + "="*70)
    print("GENERATING GRADCAM FOR DEMOGRAPHIC VARIATIONS")
    print("="*70)
    
    with open("ef_prediction/config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading real-only model...")
    backbone = cfg["model"].get("backbone", "resnet34")
    model = PTEFNetReal(backbone=backbone).to(device)
    checkpoint_path = "ef_prediction/checkpoints/real/best.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.train()
    print("✓ Model loaded")
    
    target_layer = model.cnn[7]
    gradcam = GradCAM(model, target_layer)
    
    manifest_path = "demographic_variations/variations_manifest.csv"
    df = pd.read_csv(manifest_path)
    sample_ids = df['original_id'].unique()[:top_n]
    
    print(f"\nSelected top {top_n} samples:")
    for sample_id in sample_ids:
        sample_data = df[df['original_id'] == sample_id]
        print(f"  Sample {sample_id}: {len(sample_data)} variations, EF={sample_data.iloc[0]['EF']:.1f}%")
    
    output_dir = Path("best_gradcam_results/variations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for sample_id in tqdm(sample_ids, desc="Processing"):
        try:
            sample_data = df[df['original_id'] == sample_id]
            
            real_path = Path(sample_data.iloc[0]['original_path'])
            if not real_path.exists():
                print(f"  Skipping sample {sample_id}: real video not found")
                continue
            
            real_video = load_video_tensor(real_path, cfg['model']['video_length'], cfg['model']['video_size']).unsqueeze(0).to(device)
            demo_real = torch.from_numpy(row_to_demo_vector(sample_data.iloc[0])).float().unsqueeze(0).to(device)

            with torch.enable_grad():
                real_cam = gradcam.generate(real_video, demo_real)
            
            var_video_tensors = []
            var_cams = []
            var_types = []
            
            for _, var_row in sample_data.iterrows():
                var_path = Path(var_row['synthetic_path'])
                if not var_path.exists():
                    continue
                
                var_video = load_video_tensor(var_path, cfg['model']['video_length'], cfg['model']['video_size']).unsqueeze(0).to(device)
                demo_var = torch.from_numpy(row_to_demo_vector(var_row)).float().unsqueeze(0).to(device)

                with torch.enable_grad():
                    var_cam = gradcam.generate(var_video, demo_var)
                
                var_video_tensors.append(var_video)
                var_cams.append(var_cam)
                var_types.append(var_row['variation_type'])
            
            if len(var_video_tensors) < 3:
                print(f"  Skipping sample {sample_id}: only {len(var_video_tensors)} variations found")
                continue
            
            create_variations_visualization(
                real_video, real_cam,
                var_video_tensors[:3], var_cams[:3], var_types[:3],
                sample_id, sample_data.iloc[0]['EF'],
                output_dir
            )
            
            results.append({
                'sample_id': sample_id,
                'output_dir': str(output_dir),
                'num_variations': len(var_video_tensors),
                'ef': sample_data.iloc[0]['EF']
            })
            
            print(f"  ✓ Saved individual images for sample {sample_id} to {output_dir}")
            
        except Exception as e:
            print(f"  ✗ Error processing sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✓ Generated {len(results)} variations GradCAM visualizations")
    print(f"  Output directory: {output_dir.absolute()}")
    
    return results


def main():
    """Main function"""
    print("="*70)
    print("GENERATING PROPER GRADCAM VISUALIZATIONS")
    print("="*70)
    
    perfect_results = generate_perfect_copy_gradcam(top_n=5)
    variation_results = generate_variations_gradcam(top_n=5)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Perfect Copy GradCAMs: {len(perfect_results)}")
    print(f"Variations GradCAMs: {len(variation_results)}")
    print(f"\nResults saved to: best_gradcam_results/")
    print("="*70)


if __name__ == "__main__":
    main()
