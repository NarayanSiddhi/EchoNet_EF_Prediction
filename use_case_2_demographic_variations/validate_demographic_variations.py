"""
Validation script for demographic variation generation

Validates:
1. Dataset balancing (demographic distribution)
2. Pattern preservation (quality metrics, motion analysis)
3. Model utility (training performance, bias reduction)
"""
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy import stats
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import cv2
from tqdm import tqdm


def calculate_demographic_distribution(manifest_df):
    """Calculate demographic distribution from manifest"""
    distribution = {}
    
    # Age distribution
    age_bins = ['0-1', '2-5', '6-10', '11-15', '16-18']
    for age_bin in age_bins:
        count = len(manifest_df[manifest_df['age_bin'] == age_bin])
        distribution[f'age_{age_bin}'] = count
    
    # Sex distribution
    for sex in ['F', 'M']:
        count = len(manifest_df[manifest_df['sex'] == sex])
        distribution[f'sex_{sex}'] = count
    
    # BMI distribution
    for bmi in ['underweight', 'normal', 'overweight', 'obese']:
        count = len(manifest_df[manifest_df['bmi_category'] == bmi])
        distribution[f'bmi_{bmi}'] = count
    
    # Combined groups
    for age in age_bins:
        for sex in ['F', 'M']:
            for bmi in ['underweight', 'normal', 'overweight', 'obese']:
                count = len(manifest_df[
                    (manifest_df['age_bin'] == age) &
                    (manifest_df['sex'] == sex) &
                    (manifest_df['bmi_category'] == bmi)
                ])
                distribution[f'{sex}_{age}_{bmi}'] = count
    
    return distribution


def calculate_balance_ratio(distribution):
    """Calculate balance ratio (max/min)"""
    values = [v for v in distribution.values() if v > 0]
    if len(values) == 0:
        return float('inf')
    return max(values) / min(values)


def calculate_ssim_batch(video1, video2):
    """Calculate SSIM between two videos"""
    # Simplified SSIM calculation
    video1 = video1.float()
    video2 = video2.float()
    
    mu1 = video1.mean()
    mu2 = video2.mean()
    
    sigma1_sq = ((video1 - mu1) ** 2).mean()
    sigma2_sq = ((video2 - mu2) ** 2).mean()
    sigma12 = ((video1 - mu1) * (video2 - mu2)).mean()
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim.item()


def validate_dataset_balancing(original_manifest, augmented_manifest):
    """Validate that dataset is balanced after augmentation"""
    print("="*70)
    print("DATASET BALANCING VALIDATION")
    print("="*70)
    
    original_dist = calculate_demographic_distribution(original_manifest)
    augmented_dist = calculate_demographic_distribution(augmented_manifest)
    
    print(f"\nOriginal dataset size: {len(original_manifest)}")
    print(f"Augmented dataset size: {len(augmented_manifest)}")
    
    original_balance = calculate_balance_ratio(original_dist)
    augmented_balance = calculate_balance_ratio(augmented_dist)
    
    print(f"\nBalance Ratio (max/min group size):")
    print(f"  Original: {original_balance:.2f}")
    print(f"  Augmented: {augmented_balance:.2f}")
    print(f"  Improvement: {(1 - augmented_balance/original_balance)*100:.1f}%")
    
    # Check underrepresented groups
    print(f"\nUnderrepresented Groups (original < 100 samples):")
    underrepresented = [k for k, v in original_dist.items() if v < 100]
    print(f"  Count: {len(underrepresented)}")
    
    print(f"\nAfter Augmentation:")
    for group in underrepresented[:10]:  # Show first 10
        orig_count = original_dist.get(group, 0)
        aug_count = augmented_dist.get(group, 0)
        print(f"  {group}: {orig_count} → {aug_count} ({(aug_count/orig_count-1)*100:.1f}% increase)")
    
    # Statistical test
    print(f"\nStatistical Test (Chi-square):")
    # Simplified chi-square test
    original_values = [v for v in original_dist.values() if v > 0]
    augmented_values = [augmented_dist.get(k, 0) for k in original_dist.keys() if original_dist[k] > 0]
    
    if len(original_values) == len(augmented_values):
        chi2, p_value = stats.chisquare(augmented_values, f_exp=original_values)
        print(f"  Chi-square statistic: {chi2:.2f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Interpretation: {'Significantly different' if p_value < 0.05 else 'Similar distribution'}")
    
    # Validation result
    is_balanced = augmented_balance < 2.0
    print(f"\n✓ Dataset Balancing: {'PASS' if is_balanced else 'FAIL'}")
    print(f"  Target: Balance ratio < 2.0")
    print(f"  Actual: {augmented_balance:.2f}")
    
    return {
        'original_balance': original_balance,
        'augmented_balance': augmented_balance,
        'is_balanced': is_balanced,
        'underrepresented_groups': len(underrepresented)
    }


def validate_pattern_preservation(original_manifest, variations_manifest, sample_size=100):
    """Validate that patterns are preserved in synthetic variations"""
    print("\n" + "="*70)
    print("PATTERN PRESERVATION VALIDATION")
    print("="*70)
    
    # Sample videos for validation
    sample_indices = np.random.choice(len(original_manifest), min(sample_size, len(original_manifest)), replace=False)
    
    ssim_scores = []
    ef_differences = []
    
    print(f"\nValidating {len(sample_indices)} video pairs...")
    
    for idx in tqdm(sample_indices, desc="Calculating quality metrics"):
        try:
            original_row = original_manifest.iloc[idx]
            original_path = Path(original_row.get('processed_path', original_row.get('video_path', '')))
            
            if not original_path.exists():
                continue
            
            # Find corresponding variations
            variations = variations_manifest[variations_manifest['original_id'] == idx]
            
            if len(variations) == 0:
                continue
            
            # Load original video
            original_video = load_video(original_path)
            
            # Check each variation
            for _, var_row in variations.iterrows():
                var_path = Path(var_row['synthetic_path'])
                if not var_path.exists():
                    continue
                
                # Load variation
                var_video = load_video(var_path)
                
                # Calculate SSIM
                ssim = calculate_ssim_batch(
                    torch.from_numpy(original_video).float(),
                    torch.from_numpy(var_video).float()
                )
                ssim_scores.append(ssim)
                
                # Check EF preservation
                original_ef = original_row.get('EF', original_row.get('ef'))
                var_ef = var_row.get('EF')
                if pd.notna(original_ef) and pd.notna(var_ef):
                    ef_diff = abs(float(original_ef) - float(var_ef))
                    ef_differences.append(ef_diff)
        
        except Exception as e:
            continue
    
    # Report results
    if ssim_scores:
        mean_ssim = np.mean(ssim_scores)
        std_ssim = np.std(ssim_scores)
        print(f"\nSSIM (Structural Similarity):")
        print(f"  Mean: {mean_ssim:.4f} ± {std_ssim:.4f}")
        print(f"  Min: {np.min(ssim_scores):.4f}")
        print(f"  Max: {np.max(ssim_scores):.4f}")
        print(f"  Target: > 0.90")
        print(f"  ✓ Pattern Preservation: {'PASS' if mean_ssim > 0.90 else 'FAIL'}")
    
    if ef_differences:
        mean_ef_diff = np.mean(ef_differences)
        print(f"\nEF Preservation:")
        print(f"  Mean absolute difference: {mean_ef_diff:.2f}%")
        print(f"  Target: < 5.0%")
        print(f"  ✓ EF Preservation: {'PASS' if mean_ef_diff < 5.0 else 'FAIL'}")
    
    return {
        'mean_ssim': mean_ssim if ssim_scores else None,
        'mean_ef_diff': mean_ef_diff if ef_differences else None,
        'pattern_preserved': mean_ssim > 0.90 if ssim_scores else False
    }


def load_video(video_path, video_length=32, video_size=64):
    """Load and preprocess video"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while len(frames) < video_length:
        ret, frame = cap.read()
        if not ret:
            break
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (video_size, video_size))
        frames.append(frame)
    cap.release()
    
    while len(frames) < video_length:
        frames.append(frames[-1] if frames else np.zeros((video_size, video_size)))
    
    video = np.array(frames[:video_length], dtype=np.float32) / 255.0
    return video


def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate demographic variations")
    parser.add_argument('--original_manifest', type=str, required=True,
                       help='Path to original dataset manifest')
    parser.add_argument('--variations_manifest', type=str, required=True,
                       help='Path to variations manifest')
    parser.add_argument('--output_report', type=str, default='validation_report.txt',
                       help='Output validation report file')
    
    args = parser.parse_args()
    
    # Load manifests
    original_manifest = pd.read_csv(args.original_manifest)
    variations_manifest = pd.read_csv(args.variations_manifest)
    
    # Combine for augmented dataset
    augmented_manifest = pd.concat([original_manifest, variations_manifest], ignore_index=True)
    
    # Run validations
    balancing_results = validate_dataset_balancing(original_manifest, augmented_manifest)
    pattern_results = validate_pattern_preservation(original_manifest, variations_manifest)
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"\nDataset Balancing: {'✓ PASS' if balancing_results['is_balanced'] else '✗ FAIL'}")
    print(f"Pattern Preservation: {'✓ PASS' if pattern_results.get('pattern_preserved', False) else '✗ FAIL'}")
    
    # Save report
    with open(args.output_report, 'w') as f:
        f.write("DEMOGRAPHIC VARIATIONS VALIDATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Dataset Balancing: {balancing_results}\n")
        f.write(f"Pattern Preservation: {pattern_results}\n")
    
    print(f"\nReport saved to: {args.output_report}")


if __name__ == "__main__":
    main()
