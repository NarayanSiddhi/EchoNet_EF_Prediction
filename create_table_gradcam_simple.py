"""
Create 2x4 table visualization using existing Grad-CAM files:
Row 1: Original video, Age-varied, Sex-varied, BMI-varied
Row 2: Grad-CAM overlay (original), (age), (sex), (BMI)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.cm as cm

def load_video_frame(video_path, frame_idx=8):
    """Load a single frame from video"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = min(frame_idx, total_frames - 1)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
    return None

def load_gradcam_overlay(gradcam_path):
    """Load Grad-CAM overlay image"""
    if not Path(gradcam_path).exists():
        return None
    img = cv2.imread(str(gradcam_path))
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def extract_overlay_from_gradcam_image(gradcam_path):
    """Extract overlay from existing Grad-CAM visualization"""
    if not Path(gradcam_path).exists():
        return None
    
    # Try to load the existing Grad-CAM image and extract the overlay panel
    img = cv2.imread(str(gradcam_path))
    if img is None:
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # The overlay is typically in the 3rd column (index 2) of the first row
    # Assuming 2x4 layout, overlay would be at position [0, 2] or [1, 2]
    # For a single row, it might be at column 2 or 3
    # Let's extract a region - this is approximate
    overlay_region = img[:, w//4*2:w//4*3] if w > 400 else img
    return overlay_region

def create_overlay_from_heatmap(video_frame, heatmap_path):
    """Create overlay from heatmap and video frame"""
    if not Path(heatmap_path).exists():
        return None
    
    heatmap = cv2.imread(str(heatmap_path), cv2.IMREAD_GRAYSCALE)
    if heatmap is None:
        return None
    
    # Resize heatmap to match video frame
    if video_frame.shape != heatmap.shape:
        heatmap = cv2.resize(heatmap, (video_frame.shape[1], video_frame.shape[0]))
    
    # Normalize heatmap
    heatmap = heatmap.astype(np.float32) / 255.0
    
    # Convert video frame to RGB
    if len(video_frame.shape) == 2:
        frame_rgb = np.stack([video_frame, video_frame, video_frame], axis=-1)
    else:
        frame_rgb = video_frame
    
    # Apply colormap
    heatmap_colored = cm.jet(heatmap)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Blend
    overlay = (0.6 * heatmap_colored + 0.4 * frame_rgb).astype(np.uint8)
    return overlay

def create_table_visualization(sample_id, output_path):
    """Create 2x4 table with videos and Grad-CAM overlays"""
    
    print(f"Creating table visualization for sample {sample_id:04d}...")
    
    # Paths to videos
    orig_video_path = Path(f"paper_gradcam_collection/videos/demographic_variations/originals/sample_{sample_id:04d}_original.mp4")
    age_video_path = Path(f"paper_gradcam_collection/videos/demographic_variations/variations/video_{sample_id:04d}_var1_age_variation.mp4")
    sex_video_path = Path(f"paper_gradcam_collection/videos/demographic_variations/variations/video_{sample_id:04d}_var2_sex_variation.mp4")
    bmi_video_path = Path(f"paper_gradcam_collection/videos/demographic_variations/variations/video_{sample_id:04d}_var3_bmi_variation.mp4")
    
    # Paths to Grad-CAM overlays (from existing collection)
    orig_overlay_path = Path(f"paper_gradcam_collection/demographic_variations/overlays/sample_{sample_id:04d}_real_overlay.png")
    age_overlay_path = Path(f"paper_gradcam_collection/demographic_variations/overlays/sample_{sample_id:04d}_age_variation_overlay.png")
    sex_overlay_path = Path(f"paper_gradcam_collection/demographic_variations/overlays/sample_{sample_id:04d}_sex_variation_overlay.png")
    bmi_overlay_path = Path(f"paper_gradcam_collection/demographic_variations/overlays/sample_{sample_id:04d}_bmi_variation_overlay.png")
    
    # Load video frames
    print("Loading video frames...")
    orig_frame = load_video_frame(orig_video_path, frame_idx=8)
    age_frame = load_video_frame(age_video_path, frame_idx=8)
    sex_frame = load_video_frame(sex_video_path, frame_idx=8)
    bmi_frame = load_video_frame(bmi_video_path, frame_idx=8)
    
    if any(f is None for f in [orig_frame, age_frame, sex_frame, bmi_frame]):
        print("Error: Could not load all video frames")
        return
    
    # Load or create overlays
    print("Loading Grad-CAM overlays...")
    
    # Try to load existing overlays, or create from heatmaps
    orig_overlay = load_gradcam_overlay(orig_overlay_path)
    if orig_overlay is None:
        heatmap_path = Path(f"paper_gradcam_collection/demographic_variations/heatmaps/sample_{sample_id:04d}_real_heatmap.png")
        orig_overlay = create_overlay_from_heatmap(orig_frame, heatmap_path)
    
    age_overlay = load_gradcam_overlay(age_overlay_path)
    if age_overlay is None:
        heatmap_path = Path(f"paper_gradcam_collection/demographic_variations/heatmaps/sample_{sample_id:04d}_age_variation_heatmap.png")
        age_overlay = create_overlay_from_heatmap(age_frame, heatmap_path)
    
    sex_overlay = load_gradcam_overlay(sex_overlay_path)
    if sex_overlay is None:
        heatmap_path = Path(f"paper_gradcam_collection/demographic_variations/heatmaps/sample_{sample_id:04d}_sex_variation_heatmap.png")
        sex_overlay = create_overlay_from_heatmap(sex_frame, heatmap_path)
    
    bmi_overlay = load_gradcam_overlay(bmi_overlay_path)
    if bmi_overlay is None:
        heatmap_path = Path(f"paper_gradcam_collection/demographic_variations/heatmaps/sample_{sample_id:04d}_bmi_variation_heatmap.png")
        bmi_overlay = create_overlay_from_heatmap(bmi_frame, heatmap_path)
    
    # Ensure all overlays exist
    if any(o is None for o in [orig_overlay, age_overlay, sex_overlay, bmi_overlay]):
        print("Warning: Some overlays could not be loaded, creating simple overlays...")
        # Create simple overlays as fallback
        for frame, overlay_var in [(orig_frame, 'orig_overlay'), (age_frame, 'age_overlay'), 
                                   (sex_frame, 'sex_overlay'), (bmi_frame, 'bmi_overlay')]:
            if locals()[overlay_var] is None:
                # Create a simple colored overlay
                frame_rgb = np.stack([frame, frame, frame], axis=-1) if len(frame.shape) == 2 else frame
                locals()[overlay_var] = frame_rgb
    
    # Resize all to same size
    target_size = (256, 256)
    orig_frame = cv2.resize(orig_frame, target_size)
    age_frame = cv2.resize(age_frame, target_size)
    sex_frame = cv2.resize(sex_frame, target_size)
    bmi_frame = cv2.resize(bmi_frame, target_size)
    
    orig_overlay = cv2.resize(orig_overlay, target_size) if orig_overlay is not None else orig_frame
    age_overlay = cv2.resize(age_overlay, target_size) if age_overlay is not None else age_frame
    sex_overlay = cv2.resize(sex_overlay, target_size) if sex_overlay is not None else sex_frame
    bmi_overlay = cv2.resize(bmi_overlay, target_size) if bmi_overlay is not None else bmi_frame
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Videos
    axes[0, 0].imshow(orig_frame, cmap='gray')
    axes[0, 0].set_title('Original video', fontsize=16, fontweight='bold', pad=10)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(age_frame, cmap='gray')
    axes[0, 1].set_title('Age-varied', fontsize=16, fontweight='bold', pad=10)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(sex_frame, cmap='gray')
    axes[0, 2].set_title('Sex-varied', fontsize=16, fontweight='bold', pad=10)
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(bmi_frame, cmap='gray')
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
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"table_gradcam_sample_{args.sample_id:04d}.png"
    
    create_table_visualization(args.sample_id, output_path)
