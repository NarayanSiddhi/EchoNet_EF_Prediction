"""
Generate GradCAM visualizations for demographic category subgroups.
Categories:
- Early childhood: 0 ≤ Age ≤ 6
- Middle childhood: 6 < Age ≤ 12
- Underweight: BMI ≤ 18
- Normal weight: 18 < BMI ≤ 25
- Overweight: BMI > 25
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

sys.path.insert(0, str(Path(__file__).parent))

from ef_prediction.demographics_utils import row_to_demo_vector
from ef_prediction.models.pt_efnet_real import PTEFNetReal


class GradCAM:
    """GradCAM implementation for demographic-specific visualization"""
    
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
        
        B = video_tensor.size(0)
        T = video_tensor.size(2)
        
        activations = activations.view(B, T, activations.size(1), activations.size(2), activations.size(3))
        gradients = gradients.view(B, T, gradients.size(1), gradients.size(2), gradients.size(3))
        
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


def load_video_tensor(video_path, video_length=32, video_size=128):
    """Load video exactly like DualVideoEFDataset"""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    abs_path = str(video_path.resolve())
    cap = cv2.VideoCapture(abs_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {abs_path}")
    
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
    
    if len(frames) >= video_length:
        idx = np.linspace(0, len(frames) - 1, video_length).astype(int)
        frames = frames[idx]
    else:
        pad = video_length - len(frames)
        frames = np.pad(frames, ((0, pad), (0, 0), (0, 0)), mode="edge")
    
    frames = frames.astype(np.float32) / 255.0
    frames = torch.from_numpy(frames).float().unsqueeze(0)  # (1, T, H, W)
    
    return frames


def calculate_bmi(weight_kg, height_cm):
    """Calculate BMI from weight (kg) and height (cm)"""
    if pd.isna(weight_kg) or pd.isna(height_cm) or height_cm == 0:
        return None
    height_m = height_cm / 100.0
    return weight_kg / (height_m ** 2)


def enhance_frame(frame):
    """Enhance frame contrast for visibility"""
    frame = np.clip(frame, 0, 1)
    p2, p98 = np.percentile(frame, [2, 98])
    if p98 > p2:
        frame = (frame - p2) / (p98 - p2)
        frame = np.clip(frame, 0, 1)
    frame = np.power(frame, 0.7)
    return frame


def threshold_heatmap(cam, percentile=60):
    """Show only top important regions"""
    threshold = np.percentile(cam, percentile)
    cam_thresholded = np.where(cam >= threshold, cam, 0)
    cam_thresholded = np.power(cam_thresholded, 0.8)
    return cam_thresholded


def generate_category_gradcam(video_tensor, cam, category_name, output_path, age=None, bmi=None):
    """Generate and save GradCAM visualization for a category"""
    t = video_tensor.size(2) // 2
    frame = video_tensor[0, 0, t].detach().cpu().numpy()
    
    frame_enhanced = enhance_frame(frame)
    cam_thresh = threshold_heatmap(cam, percentile=60)
    
    # Create overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_thresh), cv2.COLORMAP_JET)
    frame_rgb = (frame_enhanced * 255).astype(np.uint8)
    frame_rgb = np.stack([frame_rgb]*3, axis=-1)
    overlay = cv2.addWeighted(frame_rgb, 0.85, heatmap, 0.15, 0)
    
    # Save overlay
    cv2.imwrite(str(output_path), overlay)
    
    # Also save individual components
    base_path = output_path.parent / output_path.stem
    
    # Frame
    plt.figure(figsize=(8, 8))
    plt.imshow(frame_enhanced, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    title = f'{category_name.replace("_", " ").title()}'
    if age is not None:
        title += f' - Age: {age:.1f}'
    if bmi is not None:
        title += f' - BMI: {bmi:.1f}'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{base_path}_frame.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(cam_thresh, cmap='jet', vmin=0, vmax=1)
    plt.axis('off')
    plt.title(f'{category_name.replace("_", " ").title()} Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{base_path}_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()


def generate_demographic_category_gradcams():
    """Generate GradCAMs for each demographic category"""
    print("="*70)
    print("GENERATING GRADCAM FOR DEMOGRAPHIC CATEGORIES")
    print("="*70)
    
    with open("ef_prediction/config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
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
    
    # Load data
    variations_df = pd.read_csv("demographic_variations/variations_manifest.csv")
    manifest_df = pd.read_csv("data/processed_full/manifest_full.csv")
    
    # Extract base filename from original_path (e.g., "CR32a95e8-CR32a9abd-000032.mp4" -> "CR32a95e8-CR32a9abd-000032")
    variations_df['base_filename'] = variations_df['original_path'].apply(
        lambda x: Path(x).stem if pd.notna(x) else None
    )
    manifest_df['base_filename'] = manifest_df['file_name'].apply(
        lambda x: Path(x).stem if pd.notna(x) else None
    )
    
    # Merge to get weight/height for BMI calculation
    variations_df = variations_df.merge(
        manifest_df[['base_filename', 'weight', 'height']],
        on='base_filename',
        how='left'
    )
    
    # Calculate BMI for variations
    variations_df['bmi'] = variations_df.apply(
        lambda row: calculate_bmi(row.get('weight'), row.get('height')),
        axis=1
    )
    
    # Define categories
    # For BMI categories, we can use variation_bmi='normal'/'overweight' or calculated BMI
    categories = {
        "early_childhood": {
            "filter": lambda df: (df['variation_age'] >= 0) & (df['variation_age'] <= 6),
            "output_name": "early.png"
        },
        "middle_childhood": {
            "filter": lambda df: (df['variation_age'] > 6) & (df['variation_age'] <= 12),
            "output_name": "middle.png"
        },
        "underweight": {
            "filter": lambda df: (df['bmi'].notna()) & (df['bmi'] <= 18),
            "output_name": "underweight.png"
        },
        "normal_weight": {
            "filter": lambda df: (
                ((df['bmi'].notna()) & (df['bmi'] > 18) & (df['bmi'] <= 25)) |
                (df['variation_bmi'] == 'normal')
            ),
            "output_name": "normal.png"
        },
        "overweight": {
            "filter": lambda df: (
                ((df['bmi'].notna()) & (df['bmi'] > 25)) |
                (df['variation_bmi'] == 'overweight')
            ),
            "output_name": "overweight.png"
        }
    }
    
    output_dir = Path("best_gradcam_results/demographic_categories")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate GradCAM for each category
    for cat_name, cat_config in categories.items():
        print(f"\n{'='*70}")
        print(f"Processing category: {cat_name}")
        print(f"{'='*70}")
        
        # Filter samples for this category
        filtered = variations_df[cat_config["filter"](variations_df)].copy()
        
        if len(filtered) == 0:
            print(f"⚠ No samples found for {cat_name}")
            continue
        
        print(f"Found {len(filtered)} samples")
        
        # Try to find a working video
        sample = None
        video_tensor = None
        
        for idx, row in filtered.iterrows():
            video_path = Path(row['synthetic_path'])
            if not video_path.exists():
                continue
            
            try:
                video_tensor = load_video_tensor(video_path).to(device)
                video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension: (1, 1, T, H, W)
                sample = row
                break
            except Exception as e:
                print(f"  Skipping {video_path}: {e}")
                continue
        
        if sample is None or video_tensor is None:
            print(f"⚠ No working video found for {cat_name}")
            continue
        
        # Generate GradCAM
        try:
            demo_b = torch.from_numpy(row_to_demo_vector(sample)).float().unsqueeze(0).to(device)
            cam = gradcam.generate(video_tensor, demo_b)
            
            # Get age and BMI for title
            age = sample.get('variation_age', None)
            bmi = sample.get('bmi', None)
            
            output_path = output_dir / cat_config["output_name"]
            generate_category_gradcam(
                video_tensor, cam, cat_name, output_path, 
                age=age, bmi=bmi
            )
            
            print(f"✓ Saved: {output_path}")
            bmi_str = f"{bmi:.2f}" if bmi is not None else "N/A"
            print(f"  Sample ID: {sample['original_id']}, Age: {age}, BMI: {bmi_str}")
            
        except Exception as e:
            print(f"✗ Error generating GradCAM for {cat_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    generate_demographic_category_gradcams()
