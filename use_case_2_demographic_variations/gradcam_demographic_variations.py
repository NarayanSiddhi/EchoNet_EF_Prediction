"""
GradCAM Analysis for Demographic Variation Videos

Compares attention patterns between:
- Original videos
- Their demographic variations (age, sex, BMI)

Validates that synthetic variations preserve cardiac motion patterns
and contribute meaningfully to dataset balancing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import cosine
import imageio


# ============================================================================
# PERFECT RECONSTRUCTION GENERATOR ARCHITECTURE (for GradCAM)
# ============================================================================

class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // 16, 1),
            nn.ReLU(),
            nn.Conv3d(channels // 16, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        se_weight = self.se(out)
        out = out * se_weight
        out = out + residual
        return F.relu(out)


class DemographicEmbedding(nn.Module):
    def __init__(self, demo_dim=11, embed_dim=128):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(demo_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )

    def forward(self, demographics):
        return self.embedding(demographics)


class SpatialDemographicFusion(nn.Module):
    def __init__(self, feature_channels, demo_embed_dim=128):
        super().__init__()
        self.demo_proj = nn.Linear(demo_embed_dim, feature_channels)
        self.fusion = nn.Sequential(
            nn.Conv3d(feature_channels * 2, feature_channels, kernel_size=1),
            nn.BatchNorm3d(feature_channels),
            nn.ReLU()
        )

    def forward(self, features, demo_embed):
        B, C, T, H, W = features.shape
        demo_features = self.demo_proj(demo_embed)
        demo_spatial = demo_features.view(B, C, 1, 1, 1).expand(B, C, T, H, W)
        combined = torch.cat([features, demo_spatial], dim=1)
        fused = self.fusion(combined)
        return fused


class PerfectReconstructionGenerator(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        self.demo_embedding = DemographicEmbedding(demo_dim=11, embed_dim=128)
        
        self.enc1 = nn.Sequential(
            nn.Conv3d(1, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            ResidualBlock3D(base_channels),
            ResidualBlock3D(base_channels)
        )
        self.demo_fusion1 = SpatialDemographicFusion(base_channels, 128)
        
        self.enc2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels*2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(),
            ResidualBlock3D(base_channels*2),
            ResidualBlock3D(base_channels*2)
        )
        self.demo_fusion2 = SpatialDemographicFusion(base_channels*2, 128)
        
        self.enc3 = nn.Sequential(
            nn.Conv3d(base_channels*2, base_channels*4, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels*4),
            nn.ReLU(),
            ResidualBlock3D(base_channels*4),
            ResidualBlock3D(base_channels*4)
        )
        self.demo_fusion3 = SpatialDemographicFusion(base_channels*4, 128)
        
        self.enc4 = nn.Sequential(
            nn.Conv3d(base_channels*4, base_channels*8, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels*8),
            nn.ReLU(),
            ResidualBlock3D(base_channels*8),
            ResidualBlock3D(base_channels*8)
        )
        
        self.bottleneck = nn.Sequential(
            ResidualBlock3D(base_channels*8),
            ResidualBlock3D(base_channels*8),
            ResidualBlock3D(base_channels*8),
            ResidualBlock3D(base_channels*8)
        )
        self.demo_fusion_bottleneck = SpatialDemographicFusion(base_channels*8, 128)
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose3d(base_channels*8, base_channels*4, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels*4),
            nn.ReLU(),
            ResidualBlock3D(base_channels*4)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(base_channels*8, base_channels*2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels*2),
            nn.ReLU(),
            ResidualBlock3D(base_channels*2)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(base_channels*4, base_channels, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            ResidualBlock3D(base_channels)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv3d(base_channels*2, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(),
            ResidualBlock3D(base_channels),
            ResidualBlock3D(base_channels)
        )
        
        self.output = nn.Sequential(
            nn.Conv3d(base_channels, base_channels//2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels//2),
            nn.ReLU(),
            nn.Conv3d(base_channels//2, 1, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, x, demographics):
        demo_embed = self.demo_embedding(demographics)
        
        e1 = self.enc1(x)
        e1 = self.demo_fusion1(e1, demo_embed)
        e2 = self.enc2(e1)
        e2 = self.demo_fusion2(e2, demo_embed)
        e3 = self.enc3(e2)
        e3 = self.demo_fusion3(e3, demo_embed)
        e4 = self.enc4(e3)
        
        b = self.bottleneck(e4)
        b = self.demo_fusion_bottleneck(b, demo_embed)
        
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e3], dim=1)
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.dec1(d2)
        
        out = self.output(d1)
        return out, e3  # Return attention features from enc3


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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
    return torch.from_numpy(video).unsqueeze(0)  # [1, T, H, W]


def compute_gradcam(model, video, demographics, target_layer):
    """Compute GradCAM for a video"""
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
    
    output, _ = model(video, demographics)
    
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


def compute_attention_similarity(cam1, cam2):
    """Compute similarity between two attention maps"""
    # Flatten and normalize
    cam1_flat = cam1.flatten()
    cam2_flat = cam2.flatten()
    
    # Cosine similarity
    cosine_sim = 1 - cosine(cam1_flat, cam2_flat)
    
    # Spatial correlation (average across temporal dimension)
    cam1_mean = cam1.mean(axis=0)  # [H, W]
    cam2_mean = cam2.mean(axis=0)  # [H, W]
    spatial_corr = np.corrcoef(cam1_mean.flatten(), cam2_mean.flatten())[0, 1]
    
    return cosine_sim, spatial_corr


def visualize_gradcam_comparison(original_video, original_cam, variation_video, variation_cam, 
                                 variation_type, output_path, frame_idx=8):
    """Create comparison visualization"""
    # Get video dimensions: [B, C, T, H, W]
    T_orig = original_video.shape[2]
    T_var = variation_video.shape[2]
    T_cam_orig = original_cam.shape[0]
    T_cam_var = variation_cam.shape[0]
    
    # Ensure frame_idx is within bounds
    frame_idx_orig = min(frame_idx, T_orig - 1, T_cam_orig - 1)
    frame_idx_var = min(frame_idx, T_var - 1, T_cam_var - 1)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original video frame - correct indexing: [B, C, T, H, W]
    orig_frame = original_video[0, 0, frame_idx_orig].cpu().numpy()
    orig_frame = (orig_frame + 1) * 127.5
    axes[0, 0].imshow(orig_frame, cmap='gray')
    axes[0, 0].set_title('Original Video')
    axes[0, 0].axis('off')
    
    # Original CAM
    orig_cam_frame = original_cam[frame_idx_orig]
    axes[0, 1].imshow(orig_cam_frame, cmap='jet')
    axes[0, 1].set_title('Original Attention')
    axes[0, 1].axis('off')
    
    # Original overlay
    orig_overlay = apply_colormap(orig_frame, orig_cam_frame)
    axes[0, 2].imshow(orig_overlay)
    axes[0, 2].set_title('Original Overlay')
    axes[0, 2].axis('off')
    
    # Original heatmap (temporal average)
    orig_avg = original_cam.mean(axis=0)
    axes[0, 3].imshow(orig_avg, cmap='jet')
    axes[0, 3].set_title('Original Avg Attention')
    axes[0, 3].axis('off')
    
    # Variation video frame - correct indexing: [B, C, T, H, W]
    var_frame = variation_video[0, 0, frame_idx_var].cpu().numpy()
    var_frame = (var_frame + 1) * 127.5
    axes[1, 0].imshow(var_frame, cmap='gray')
    axes[1, 0].set_title(f'Variation: {variation_type}')
    axes[1, 0].axis('off')
    
    # Variation CAM
    var_cam_frame = variation_cam[frame_idx_var]
    axes[1, 1].imshow(var_cam_frame, cmap='jet')
    axes[1, 1].set_title('Variation Attention')
    axes[1, 1].axis('off')
    
    # Variation overlay
    var_overlay = apply_colormap(var_frame, var_cam_frame)
    axes[1, 2].imshow(var_overlay)
    axes[1, 2].set_title('Variation Overlay')
    axes[1, 2].axis('off')
    
    # Variation heatmap (temporal average)
    var_avg = variation_cam.mean(axis=0)
    axes[1, 3].imshow(var_avg, cmap='jet')
    axes[1, 3].set_title('Variation Avg Attention')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def apply_colormap(frame, cam, alpha=0.5):
    """Apply colormap overlay to frame"""
    # Resize CAM to match frame size if needed
    if cam.shape != frame.shape:
        import cv2
        cam = cv2.resize(cam, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    frame_rgb = np.stack([frame, frame, frame], axis=-1).astype(np.uint8)
    cam_colored = cm.jet(cam)[:, :, :3]
    cam_colored = (cam_colored * 255).astype(np.uint8)
    overlay = (alpha * cam_colored + (1 - alpha) * frame_rgb).astype(np.uint8)
    return overlay


def analyze_demographic_variations(
    original_manifest,
    variations_manifest,
    checkpoint_path,
    output_dir='gradcam_variations_analysis',
    device='cuda',
    num_samples=50
):
    """Analyze GradCAM for demographic variations"""
    print("="*70)
    print("GRADCAM ANALYSIS FOR DEMOGRAPHIC VARIATIONS")
    print("="*70)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading model: {checkpoint_path}")
    model = PerfectReconstructionGenerator(base_channels=64).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['generator'])
    model.eval()
    print("✓ Model loaded")
    
    # Select target layer for GradCAM (encoder level 3)
    target_layer = model.enc3
    
    # Load manifests
    original_df = pd.read_csv(original_manifest)
    variations_df = pd.read_csv(variations_manifest)
    
    # Sample videos for analysis
    sample_indices = np.random.choice(len(original_df), min(num_samples, len(original_df)), replace=False)
    
    print(f"\nAnalyzing {len(sample_indices)} video pairs...")
    
    results = []
    attention_similarities = []
    
    with torch.no_grad():
        for idx in tqdm(sample_indices, desc="Computing GradCAM"):
            try:
                original_row = original_df.iloc[idx]
                original_path = Path(original_row.get('processed_path', original_row.get('video_path', '')))
                
                if not original_path.exists():
                    continue
                
                # Load original video
                original_video = load_video(original_path).unsqueeze(0).to(device)
                
                # Get original demographics
                original_sex = original_row.get('sex', original_row.get('Sex', 'F'))
                original_age = float(original_row.get('age', original_row.get('Age', 10)))
                original_bmi = original_row.get('bmi_category', 'normal')
                
                original_demo = encode_demographics(original_sex, original_age, original_bmi).unsqueeze(0).to(device)
                
                # Compute GradCAM for original
                with torch.enable_grad():
                    original_cam, _ = compute_gradcam(model, original_video, original_demo, target_layer)
                
                # Find corresponding variations
                variations = variations_df[variations_df['original_id'] == idx]
                
                for var_idx, var_row in variations.iterrows():
                    var_path = Path(var_row['synthetic_path'])
                    if not var_path.exists():
                        continue
                    
                    variation_type = var_row['variation_type']
                    
                    # Load variation video
                    variation_video = load_video(var_path).unsqueeze(0).to(device)
                    
                    # Get variation demographics
                    var_sex = var_row['variation_sex']
                    var_age = var_row['variation_age']
                    var_bmi = var_row['variation_bmi']
                    
                    var_demo = encode_demographics(var_sex, var_age, var_bmi).unsqueeze(0).to(device)
                    
                    # Compute GradCAM for variation
                    with torch.enable_grad():
                        variation_cam, _ = compute_gradcam(model, variation_video, var_demo, target_layer)
                    
                    # Compute attention similarity
                    cosine_sim, spatial_corr = compute_attention_similarity(original_cam, variation_cam)
                    attention_similarities.append({
                        'original_id': idx,
                        'variation_type': variation_type,
                        'cosine_similarity': cosine_sim,
                        'spatial_correlation': spatial_corr
                    })
                    
                    # Save visualization
                    sample_dir = output_dir / f"sample_{idx:04d}"
                    sample_dir.mkdir(exist_ok=True)
                    
                    vis_path = sample_dir / f"gradcam_comparison_{variation_type}.png"
                    visualize_gradcam_comparison(
                        original_video, original_cam,
                        variation_video, variation_cam,
                        variation_type, vis_path
                    )
                    
                    results.append({
                        'original_id': idx,
                        'variation_type': variation_type,
                        'cosine_similarity': cosine_sim,
                        'spatial_correlation': spatial_corr,
                        'visualization_path': str(vis_path)
                    })
            
            except Exception as e:
                print(f"\nError processing sample {idx}: {e}")
                continue
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = output_dir / 'gradcam_analysis_results.csv'
    results_df.to_csv(results_path, index=False)
    
    # Summary statistics
    print("\n" + "="*70)
    print("GRADCAM ANALYSIS SUMMARY")
    print("="*70)
    
    if len(results_df) > 0:
        print(f"\nTotal comparisons: {len(results_df)}")
        print(f"\nAttention Similarity Metrics:")
        print(f"  Cosine Similarity:")
        print(f"    Mean: {results_df['cosine_similarity'].mean():.4f} ± {results_df['cosine_similarity'].std():.4f}")
        print(f"    Min: {results_df['cosine_similarity'].min():.4f}")
        print(f"    Max: {results_df['cosine_similarity'].max():.4f}")
        
        print(f"\n  Spatial Correlation:")
        print(f"    Mean: {results_df['spatial_correlation'].mean():.4f} ± {results_df['spatial_correlation'].std():.4f}")
        print(f"    Min: {results_df['spatial_correlation'].min():.4f}")
        print(f"    Max: {results_df['spatial_correlation'].max():.4f}")
        
        # By variation type
        print(f"\nBy Variation Type:")
        for var_type in ['age_variation', 'sex_variation', 'bmi_variation']:
            var_results = results_df[results_df['variation_type'] == var_type]
            if len(var_results) > 0:
                print(f"\n  {var_type}:")
                print(f"    Cosine Similarity: {var_results['cosine_similarity'].mean():.4f}")
                print(f"    Spatial Correlation: {var_results['spatial_correlation'].mean():.4f}")
        
        # Validation
        mean_cosine = results_df['cosine_similarity'].mean()
        mean_spatial = results_df['spatial_correlation'].mean()
        
        print(f"\n" + "="*70)
        print("VALIDATION RESULTS")
        print("="*70)
        print(f"✓ Attention Pattern Preservation: {'PASS' if mean_cosine > 0.75 else 'FAIL'}")
        print(f"  Target: Cosine similarity > 0.75")
        print(f"  Actual: {mean_cosine:.4f}")
        print(f"\n✓ Spatial Attention Consistency: {'PASS' if mean_spatial > 0.70 else 'FAIL'}")
        print(f"  Target: Spatial correlation > 0.70")
        print(f"  Actual: {mean_spatial:.4f}")
        
        if mean_cosine > 0.75 and mean_spatial > 0.70:
            print(f"\n✅ DEMOGRAPHIC VARIATIONS PRESERVE CARDIAC MOTION PATTERNS")
            print(f"   Synthetic videos follow same attention patterns as originals")
            print(f"   Variations contribute meaningfully to dataset balancing")
        else:
            print(f"\n⚠️  ATTENTION PATTERNS DIFFER SIGNIFICANTLY")
            print(f"   May need to review generation process")
    
    print(f"\nResults saved to: {results_path}")
    print(f"Visualizations saved to: {output_dir}")
    
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GradCAM analysis for demographic variations")
    parser.add_argument('--original_manifest', type=str, required=True,
                       help='Path to original dataset manifest')
    parser.add_argument('--variations_manifest', type=str, required=True,
                       help='Path to variations manifest')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to perfect reconstruction checkpoint')
    parser.add_argument('--output_dir', type=str, default='gradcam_variations_analysis',
                       help='Output directory for analysis')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to analyze')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    analyze_demographic_variations(
        original_manifest=args.original_manifest,
        variations_manifest=args.variations_manifest,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        num_samples=args.num_samples
    )
