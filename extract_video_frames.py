"""
Extract frames from synthetic videos and save as images
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

def extract_video_frames(video_path, output_dir, num_frames=8, frame_indices=None):
    """Extract frames from video and save as images"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_indices is None:
        # Extract evenly spaced frames
        if total_frames > num_frames:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        else:
            frame_indices = list(range(total_frames))
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
    
    cap.release()
    
    # Save individual frames
    video_name = Path(video_path).stem
    for i, frame in enumerate(frames):
        frame_path = output_dir / f"{video_name}_frame_{i:02d}.png"
        cv2.imwrite(str(frame_path), frame)
    
    # Create a grid visualization
    if len(frames) > 0:
        create_frame_grid(frames, output_dir / f"{video_name}_grid.png")
    
    return frames

def create_frame_grid(frames, output_path, cols=4):
    """Create a grid of frames"""
    rows = (len(frames) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, frame in enumerate(frames):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.imshow(frame, cmap='gray')
        ax.set_title(f'Frame {i}', fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(len(frames), rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def process_synthetic_videos():
    """Process all synthetic videos"""
    base_dir = Path(".")
    
    # Demographic variations
    print("Processing demographic variations...")
    demo_vars_dir = base_dir / "demographic_variations_fixed_v2"
    output_demo = base_dir / "synthetic_video_frames" / "demographic_variations"
    
    if demo_vars_dir.exists():
        for video_file in tqdm(list(demo_vars_dir.glob("*.mp4"))[:9], desc="Demographic variations"):
            extract_video_frames(video_file, output_demo, num_frames=8)
    
    # Perfect reconstruction
    print("\nProcessing perfect reconstruction...")
    perfect_dir = base_dir / "perfect_synthetic_copies"
    output_perfect = base_dir / "synthetic_video_frames" / "perfect_reconstruction"
    
    if perfect_dir.exists():
        for video_file in tqdm(list(perfect_dir.glob("*.mp4"))[:5], desc="Perfect reconstruction"):
            extract_video_frames(video_file, output_perfect, num_frames=8)
    
    # Paper collection videos
    print("\nProcessing paper collection videos...")
    paper_videos = base_dir / "paper_gradcam_collection" / "videos"
    output_paper = base_dir / "synthetic_video_frames" / "paper_collection"
    
    if paper_videos.exists():
        # Variations
        var_dir = paper_videos / "demographic_variations" / "variations"
        if var_dir.exists():
            for video_file in tqdm(list(var_dir.glob("*.mp4"))[:9], desc="Paper variations"):
                extract_video_frames(video_file, output_paper / "variations", num_frames=8)
        
        # Originals
        orig_dir = paper_videos / "demographic_variations" / "originals"
        if orig_dir.exists():
            for video_file in tqdm(list(orig_dir.glob("*.mp4"))[:5], desc="Paper originals"):
                extract_video_frames(video_file, output_paper / "originals", num_frames=8)
    
    print(f"\n✅ Frames extracted to: synthetic_video_frames/")

if __name__ == "__main__":
    process_synthetic_videos()
