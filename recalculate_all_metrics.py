"""
Recalculate SSIM, PSNR, and MSE metrics for all three use cases
This script reads the actual generated video files and recalculates metrics
"""
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import warnings
warnings.filterwarnings('ignore')


def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate PSNR between two images"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def calculate_mse(img1, img2):
    """Calculate MSE between two images"""
    return np.mean((img1 - img2) ** 2)


def load_video_frames(video_path, target_frames=32, target_size=(64, 64)):
    """Load video and return frames as numpy array"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    while len(frames) < target_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, target_size)
        frames.append(frame)
    cap.release()
    
    # Pad if needed
    while len(frames) < target_frames:
        frames.append(frames[-1] if frames else np.zeros(target_size, dtype=np.uint8))
    
    # Normalize to [0, 1]
    video = np.array(frames[:target_frames], dtype=np.float32) / 255.0
    return video


def calculate_video_metrics(original_path, synthetic_path, target_frames=32, target_size=(64, 64)):
    """Calculate SSIM, PSNR, and MSE between two videos"""
    try:
        original = load_video_frames(original_path, target_frames, target_size)
        synthetic = load_video_frames(synthetic_path, target_frames, target_size)
        
        # Ensure same shape
        min_frames = min(len(original), len(synthetic))
        original = original[:min_frames]
        synthetic = synthetic[:min_frames]
        
        # Calculate metrics frame by frame
        ssim_scores = []
        psnr_scores = []
        mse_scores = []
        
        for i in range(min_frames):
            # SSIM
            ssim_val = ssim(original[i], synthetic[i], data_range=1.0)
            ssim_scores.append(ssim_val)
            
            # PSNR
            psnr_val = calculate_psnr(original[i], synthetic[i], max_val=1.0)
            psnr_scores.append(psnr_val)
            
            # MSE
            mse_val = calculate_mse(original[i], synthetic[i])
            mse_scores.append(mse_val)
        
        # Average across frames
        mean_ssim = np.mean(ssim_scores)
        mean_psnr = np.mean(psnr_scores)
        mean_mse = np.mean(mse_scores)
        
        # Handle infinite PSNR
        if np.isinf(mean_psnr):
            mean_psnr = float('inf')
        
        return mean_ssim, mean_psnr, mean_mse
    
    except Exception as e:
        print(f"Error calculating metrics for {original_path} vs {synthetic_path}: {e}")
        return None, None, None


def recalculate_use_case_3():
    """Recalculate metrics for Use Case 3: Perfect Reconstruction"""
    print("="*70)
    print("RECALCULATING METRICS: USE CASE 3 (Perfect Reconstruction)")
    print("="*70)
    
    manifest_path = Path("perfect_synthetic_copies/perfect_copies_manifest.csv")
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        return
    
    df = pd.read_csv(manifest_path)
    print(f"\nTotal samples in manifest: {len(df)}")
    
    # Recalculate metrics
    results = []
    errors = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Recalculating metrics"):
        original_path = Path(row['original_path'])
        synthetic_path = Path(row['synthetic_path'])
        
        if not original_path.exists() or not synthetic_path.exists():
            errors += 1
            continue
        
        ssim_val, psnr_val, mse_val = calculate_video_metrics(
            original_path, synthetic_path, target_frames=16, target_size=(64, 64)
        )
        
        if ssim_val is not None:
            results.append({
                'original_id': row['original_id'],
                'original_path': str(original_path),
                'synthetic_path': str(synthetic_path),
                'EF': row['EF'],
                'demographics': row['demographics'],
                'SSIM': ssim_val,
                'PSNR': psnr_val,
                'MSE': mse_val
            })
        else:
            errors += 1
    
    # Create new dataframe
    new_df = pd.DataFrame(results)
    
    # Statistics
    print(f"\n{'='*70}")
    print("RECALCULATED METRICS - USE CASE 3")
    print(f"{'='*70}")
    print(f"Successfully calculated: {len(new_df)}/{len(df)}")
    print(f"Errors: {errors}")
    
    if len(new_df) > 0:
        print(f"\nSSIM Statistics:")
        print(f"  Mean: {new_df['SSIM'].mean():.6f}")
        print(f"  Std: {new_df['SSIM'].std():.6f}")
        print(f"  Min: {new_df['SSIM'].min():.6f}")
        print(f"  Max: {new_df['SSIM'].max():.6f}")
        print(f"  >0.99: {(new_df['SSIM'] > 0.99).sum()} ({(new_df['SSIM'] > 0.99).sum()/len(new_df)*100:.1f}%)")
        
        print(f"\nPSNR Statistics:")
        psnr_finite = new_df[new_df['PSNR'] != float('inf')]['PSNR']
        print(f"  Mean (finite): {psnr_finite.mean():.2f}")
        print(f"  Std (finite): {psnr_finite.std():.2f}")
        print(f"  Min: {new_df['PSNR'].min():.2f}")
        print(f"  Infinite count: {(new_df['PSNR'] == float('inf')).sum()}")
        print(f"  >48 dB: {(new_df['PSNR'] > 48).sum()} ({(new_df['PSNR'] > 48).sum()/len(new_df)*100:.1f}%)")
        
        print(f"\nMSE Statistics:")
        print(f"  Mean: {new_df['MSE'].mean():.6f}")
        print(f"  Std: {new_df['MSE'].std():.6f}")
        print(f"  Min: {new_df['MSE'].min():.6f}")
        print(f"  Max: {new_df['MSE'].max():.6f}")
        print(f"  <1.0: {(new_df['MSE'] < 1.0).sum()} ({(new_df['MSE'] < 1.0).sum()/len(new_df)*100:.1f}%)")
    
    # Save updated manifest
    output_path = manifest_path.parent / "perfect_copies_manifest_recalculated.csv"
    new_df.to_csv(output_path, index=False)
    print(f"\n✓ Updated manifest saved to: {output_path}")
    
    return new_df


def recalculate_use_case_2():
    """Recalculate metrics for Use Case 2: Demographic Variations"""
    print("\n" + "="*70)
    print("RECALCULATING METRICS: USE CASE 2 (Demographic Variations)")
    print("="*70)
    
    manifest_path = Path("use_case_2_demographic_variations/demographic_variations/variations_manifest.csv")
    
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        return
    
    df = pd.read_csv(manifest_path)
    
    print(f"\nTotal variation samples: {len(df)}")
    
    # Recalculate metrics
    results = []
    errors = 0
    
    # Get base directory for resolving relative paths (use_case_2_demographic_variations)
    base_dir = Path("use_case_2_demographic_variations")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Recalculating metrics"):
        # Get original_id from the row
        original_id = row.get('original_id', idx)
        
        # Use original_path from the variations manifest (it already has the correct path)
        original_path_str = row.get('original_path', '')
        if not original_path_str:
            errors += 1
            continue
        
        # Resolve original path - check if it's absolute or relative
        original_path = Path(original_path_str)
        if original_path.is_absolute():
            # Absolute path from different machine - try to find in data/processed/videos
            video_name = original_path.name
            original_path = Path('data/processed/videos') / video_name
        else:
            # Relative path - resolve from data/processed/videos
            if 'videos' in str(original_path):
                original_path = Path('data/processed/videos') / original_path.name
            else:
                original_path = Path('data/processed/videos') / Path(original_path_str).name
        
        # Resolve synthetic path relative to base_dir
        synthetic_path_str = row['synthetic_path']
        if Path(synthetic_path_str).is_absolute():
            synthetic_path = Path(synthetic_path_str)
        else:
            # Path is relative, resolve from base_dir
            synthetic_path = base_dir / synthetic_path_str
        
        if not original_path.exists() or not synthetic_path.exists():
            errors += 1
            continue
        
        ssim_val, psnr_val, mse_val = calculate_video_metrics(
            original_path, synthetic_path, target_frames=16, target_size=(64, 64)
        )
        
        if ssim_val is not None:
            results.append({
                'original_id': original_id,
                'original_path': str(original_path),
                'variation_type': row['variation_type'],
                'synthetic_path': str(synthetic_path),
                'EF': row['EF'],
                'SSIM': ssim_val,
                'PSNR': psnr_val,
                'MSE': mse_val
            })
        else:
            errors += 1
    
    # Create new dataframe
    new_df = pd.DataFrame(results)
    
    # Statistics
    print(f"\n{'='*70}")
    print("RECALCULATED METRICS - USE CASE 2")
    print(f"{'='*70}")
    print(f"Successfully calculated: {len(new_df)}/{len(df)}")
    print(f"Errors: {errors}")
    
    if len(new_df) > 0:
        print(f"\nOverall Statistics:")
        print(f"  SSIM Mean: {new_df['SSIM'].mean():.6f} ± {new_df['SSIM'].std():.6f}")
        psnr_finite = new_df[new_df['PSNR'] != float('inf')]['PSNR']
        print(f"  PSNR Mean (finite): {psnr_finite.mean():.2f} ± {psnr_finite.std():.2f}")
        print(f"  MSE Mean: {new_df['MSE'].mean():.6f} ± {new_df['MSE'].std():.6f}")
        
        print(f"\nBy Variation Type:")
        for var_type in new_df['variation_type'].unique():
            var_df = new_df[new_df['variation_type'] == var_type]
            print(f"  {var_type}:")
            print(f"    SSIM: {var_df['SSIM'].mean():.6f} ± {var_df['SSIM'].std():.6f}")
            psnr_finite_var = var_df[var_df['PSNR'] != float('inf')]['PSNR']
            print(f"    PSNR: {psnr_finite_var.mean():.2f} ± {psnr_finite_var.std():.2f}")
            print(f"    MSE: {var_df['MSE'].mean():.6f} ± {var_df['MSE'].std():.6f}")
    
    # Save updated manifest
    output_path = manifest_path.parent / "variations_manifest_recalculated.csv"
    new_df.to_csv(output_path, index=False)
    print(f"\n✓ Updated manifest saved to: {output_path}")
    
    return new_df


def recalculate_use_case_1():
    """Recalculate metrics for Use Case 1: Dataset Balancing (if original videos available)"""
    print("\n" + "="*70)
    print("RECALCULATING METRICS: USE CASE 1 (Dataset Balancing)")
    print("="*70)
    print("Note: Use Case 1 generates from random noise, so direct comparison")
    print("with originals is not applicable. Quality assessment would require")
    print("different metrics (e.g., FID, IS) or visual inspection.")
    print("Skipping Use Case 1 metric recalculation.")
    return None


def main():
    """Main function to recalculate all metrics"""
    print("\n" + "="*70)
    print("RECALCULATING ALL METRICS FOR ALL USE CASES")
    print("="*70)
    
    # Use Case 3
    uc3_results = recalculate_use_case_3()
    
    # Use Case 2
    uc2_results = recalculate_use_case_2()
    
    # Use Case 1
    uc1_results = recalculate_use_case_1()
    
    print("\n" + "="*70)
    print("RECALCULATION COMPLETE")
    print("="*70)
    print("\nSummary:")
    if uc3_results is not None:
        print(f"  Use Case 3: {len(uc3_results)} samples recalculated")
    if uc2_results is not None:
        print(f"  Use Case 2: {len(uc2_results)} samples recalculated")
    if uc1_results is not None:
        print(f"  Use Case 1: Metrics recalculated")
    else:
        print(f"  Use Case 1: Skipped (generates from noise)")


if __name__ == "__main__":
    main()
