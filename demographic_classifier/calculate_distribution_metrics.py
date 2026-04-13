"""
Calculate distribution divergence metrics (Option B):
- KL divergence before and after Use Case 2
- Jensen-Shannon divergence
- Entropy increase
This shows redistribution is mathematically grounded.
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json


def calculate_entropy(probs):
    """Calculate Shannon entropy"""
    probs = np.array(probs)
    probs = probs[probs > 0]  # Remove zeros
    return -np.sum(probs * np.log2(probs))


def calculate_kl_divergence(p, q):
    """Calculate KL divergence D_KL(P || Q)"""
    p = np.array(p)
    q = np.array(q)
    
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    
    # Avoid zeros
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    
    return np.sum(p * np.log(p / q))


def calculate_js_divergence(p, q):
    """Calculate Jensen-Shannon divergence"""
    p = np.array(p)
    q = np.array(q)
    
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    
    # Avoid zeros
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    
    m = 0.5 * (p + q)
    js = 0.5 * calculate_kl_divergence(p, m) + 0.5 * calculate_kl_divergence(q, m)
    return js


def get_distribution(df, column, value_counts=None):
    """Get probability distribution for a column"""
    if value_counts is None:
        value_counts = df[column].value_counts()
    
    total = value_counts.sum()
    probs = value_counts / total
    return probs.to_dict(), probs.values


def main():
    print("\n" + "="*80)
    print("DISTRIBUTION DIVERGENCE METRICS CALCULATION")
    print("="*80 + "\n")
    
    # Load original dataset
    original_manifest = "data/processed_full/train_manifest_filtered_clean.csv"
    if not Path(original_manifest).exists():
        original_manifest = "../data/processed_full/train_manifest_filtered_clean.csv"
    if not Path(original_manifest).exists():
        original_manifest = "data/processed/manifest.csv"
        if not Path(original_manifest).exists():
            original_manifest = "../data/processed/manifest.csv"
    
    print(f"Loading original dataset: {original_manifest}")
    original_df = pd.read_csv(original_manifest)
    
    # Calculate BMI category if not present
    if 'bmi_category' not in original_df.columns:
        original_df['bmi'] = original_df.apply(
            lambda row: row['weight'] / ((row['height'] / 100) ** 2) 
            if pd.notna(row['weight']) and pd.notna(row['height']) and row['height'] > 0 
            else np.nan, axis=1
        )
        original_df['bmi_category'] = pd.cut(
            original_df['bmi'],
            bins=[0, 18.5, 25, 30, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese'],
            include_lowest=True
        )
        original_df['bmi_category'] = original_df['bmi_category'].fillna('Normal')
    
    print(f"Original samples: {len(original_df)}\n")
    
    # Load augmented dataset (original + synthetic from Use Case 2)
    variations_manifest = "use_case_2_demographic_variations/demographic_variations/variations_manifest.csv"
    if not Path(variations_manifest).exists():
        variations_manifest = "../use_case_2_demographic_variations/demographic_variations/variations_manifest.csv"
    print(f"Loading variations manifest: {variations_manifest}")
    variations_df = pd.read_csv(variations_manifest)
    
    # Create augmented dataset by combining original and synthetic
    # For each original video, we have 3 synthetic variations
    augmented_df = original_df.copy()
    
    # Add synthetic variations
    synthetic_rows = []
    for _, row in variations_df.iterrows():
        synthetic_row = {
            'sex': row['variation_sex'],
            'age': row['variation_age'],
            'bmi_category': row['variation_bmi'].capitalize() if isinstance(row['variation_bmi'], str) else 'Normal'
        }
        # Map age to age_bin
        age = row['variation_age']
        if age <= 1:
            synthetic_row['age_bin'] = '0-1'
        elif age <= 2:
            synthetic_row['age_bin'] = '1-2'
        elif age <= 3:
            synthetic_row['age_bin'] = '2-3'
        elif age <= 5:
            synthetic_row['age_bin'] = '3-5'
        elif age <= 8:
            synthetic_row['age_bin'] = '5-8'
        elif age <= 12:
            synthetic_row['age_bin'] = '8-12'
        elif age <= 15:
            synthetic_row['age_bin'] = '12-15'
        else:
            synthetic_row['age_bin'] = '15-18'
        synthetic_rows.append(synthetic_row)
    
    synthetic_df = pd.DataFrame(synthetic_rows)
    augmented_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    print(f"Augmented samples (original + synthetic): {len(augmented_df)}\n")
    
    results = {}
    
    # Analyze each demographic dimension
    for dimension in ['sex', 'age_bin', 'bmi_category']:
        print(f"{'='*80}")
        print(f"Analyzing {dimension.upper()}")
        print(f"{'='*80}")
        
        # Original distribution
        orig_counts = original_df[dimension].value_counts()
        orig_probs_dict, orig_probs = get_distribution(original_df, dimension, orig_counts)
        orig_entropy = calculate_entropy(orig_probs)
        
        # Augmented distribution
        aug_counts = augmented_df[dimension].value_counts()
        aug_probs_dict, aug_probs = get_distribution(augmented_df, dimension, aug_counts)
        aug_entropy = calculate_entropy(aug_probs)
        
        # Ensure same categories for comparison
        all_categories = sorted(set(list(orig_probs_dict.keys()) + list(aug_probs_dict.keys())))
        orig_probs_aligned = np.array([orig_probs_dict.get(cat, 0) for cat in all_categories])
        aug_probs_aligned = np.array([aug_probs_dict.get(cat, 0) for cat in all_categories])
        
        # Normalize
        orig_probs_aligned = orig_probs_aligned / orig_probs_aligned.sum()
        aug_probs_aligned = aug_probs_aligned / aug_probs_aligned.sum()
        
        # Calculate metrics
        kl_div = calculate_kl_divergence(orig_probs_aligned, aug_probs_aligned)
        js_div = calculate_js_divergence(orig_probs_aligned, aug_probs_aligned)
        entropy_increase = aug_entropy - orig_entropy
        entropy_ratio = aug_entropy / orig_entropy if orig_entropy > 0 else 0
        
        print(f"\nOriginal Distribution:")
        for cat, count in orig_counts.items():
            pct = (count / len(original_df)) * 100
            print(f"  {cat}: {count} ({pct:.2f}%)")
        print(f"  Entropy: {orig_entropy:.4f}")
        
        print(f"\nAugmented Distribution:")
        for cat, count in aug_counts.items():
            pct = (count / len(augmented_df)) * 100
            print(f"  {cat}: {count} ({pct:.2f}%)")
        print(f"  Entropy: {aug_entropy:.4f}")
        
        print(f"\nMetrics:")
        print(f"  KL Divergence: {kl_div:.6f}")
        print(f"  JS Divergence: {js_div:.6f}")
        print(f"  Entropy Increase: {entropy_increase:.6f} ({entropy_ratio:.4f}x)")
        
        results[dimension] = {
            'original': {
                'distribution': orig_probs_dict,
                'entropy': float(orig_entropy),
                'counts': orig_counts.to_dict()
            },
            'augmented': {
                'distribution': aug_probs_dict,
                'entropy': float(aug_entropy),
                'counts': aug_counts.to_dict()
            },
            'metrics': {
                'kl_divergence': float(kl_div),
                'js_divergence': float(js_div),
                'entropy_increase': float(entropy_increase),
                'entropy_ratio': float(entropy_ratio)
            }
        }
        print()
    
    # Overall summary
    print(f"{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}\n")
    
    avg_entropy_increase = np.mean([results[dim]['metrics']['entropy_increase'] for dim in results])
    avg_kl_div = np.mean([results[dim]['metrics']['kl_divergence'] for dim in results])
    avg_js_div = np.mean([results[dim]['metrics']['js_divergence'] for dim in results])
    
    print(f"Average Entropy Increase: {avg_entropy_increase:.6f}")
    print(f"Average KL Divergence: {avg_kl_div:.6f}")
    print(f"Average JS Divergence: {avg_js_div:.6f}")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    if avg_entropy_increase > 0.1:
        print("✓ STRONG: Significant entropy increase indicates substantial distribution rebalancing.")
    elif avg_entropy_increase > 0.05:
        print("✓ MODERATE: Noticeable entropy increase shows effective redistribution.")
    else:
        print("⚠️ WEAK: Minimal entropy increase suggests limited redistribution effect.")
    
    if avg_js_div < 0.1:
        print("✓ The augmented distribution is similar to original (low JS divergence).")
        print("  This is expected as synthetic videos are derived from real videos.")
    else:
        print("⚠️ The augmented distribution differs from original (higher JS divergence).")
        print("  This indicates significant redistribution occurred.")
    
    # Save results
    results_dir = Path("results")
    if not results_dir.exists():
        results_dir = Path("../demographic_classifier/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'overall_metrics': {
            'average_entropy_increase': float(avg_entropy_increase),
            'average_kl_divergence': float(avg_kl_div),
            'average_js_divergence': float(avg_js_div)
        },
        'detailed_results': results
    }
    
    with open(results_dir / "distribution_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Results saved to {results_dir / 'distribution_metrics.json'}")
    print("\n🎉 Distribution analysis completed!\n")


if __name__ == "__main__":
    main()
