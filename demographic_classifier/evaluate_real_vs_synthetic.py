"""
Evaluate demographic classifier on real vs synthetic videos.
This validates that synthetic videos encode demographic signals correctly.
"""
import os
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, classification_report

from train_demographic_classifier import DemographicVideoDataset, DemographicClassifier3D
from train_demographic_classifier_improved import ImprovedDemographicClassifier3D


def evaluate_on_manifest(model, manifest_path, video_root_dir, device, config, label_prefix="", train_manifest_path=None):
    """Evaluate model on a specific manifest
    
    Args:
        train_manifest_path: Path to training manifest to ensure consistent label encoding
    """
    # For synthetic videos, we need to ensure label encoding matches training data
    if train_manifest_path and "synthetic" in manifest_path.lower():
        # Load training data to get consistent age bin mapping
        train_df = pd.read_csv(train_manifest_path)
        train_age_bins = sorted(train_df['age_bin'].unique())
        
        # Create a custom dataset that uses training age bins
        dataset = DemographicVideoDataset(
            manifest_path=manifest_path,
            video_root_dir=video_root_dir,
            video_length=config["model"]["video_length"],
            video_size=config["model"]["video_size"],
            augment=False
        )
        
        # Override age label encoding to match training data
        train_age_map = {bin_name: idx for idx, bin_name in enumerate(train_age_bins)}
        # Map synthetic age bins to training age bins
        dataset.df['age_label'] = dataset.df['age_bin'].map(train_age_map)
        
        # Fill missing values (age bins not in training data)
        missing_mask = dataset.df['age_label'].isna()
        if missing_mask.any():
            print(f"Warning: {missing_mask.sum()} samples have age bins not in training data")
            # Map to closest bin or default
            for age_bin in dataset.df.loc[missing_mask, 'age_bin'].unique():
                # Find closest training bin
                age_val = float(age_bin.split('-')[0]) if '-' in age_bin else 0
                closest_bin = min(train_age_bins, key=lambda x: abs(float(x.split('-')[0]) - age_val) if '-' in x else abs(0 - age_val))
                dataset.df.loc[dataset.df['age_bin'] == age_bin, 'age_label'] = train_age_map[closest_bin]
        
        dataset.df = dataset.df.dropna(subset=['age_label']).reset_index(drop=True)
    else:
        dataset = DemographicVideoDataset(
            manifest_path=manifest_path,
            video_root_dir=video_root_dir,
            video_length=config["model"]["video_length"],
            video_size=config["model"]["video_size"],
            augment=False
        )
    
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    model.eval()
    sex_preds, sex_labels = [], []
    age_preds, age_labels = [], []
    bmi_preds, bmi_labels = [], []
    
    with torch.no_grad():
        for video, sex_label, age_label, bmi_label in tqdm(loader, desc=f"Evaluating {label_prefix}"):
            video = video.to(device)
            
            sex_logits, age_logits, bmi_logits = model(video)
            
            sex_pred = sex_logits.argmax(dim=1).cpu().numpy()
            age_pred = age_logits.argmax(dim=1).cpu().numpy()
            bmi_pred = bmi_logits.argmax(dim=1).cpu().numpy()
            
            sex_label_np = sex_label.cpu().numpy()
            age_label_np = age_label.cpu().numpy()
            bmi_label_np = bmi_label.cpu().numpy()
            
            sex_preds.extend(sex_pred)
            sex_labels.extend(sex_label_np)
            age_preds.extend(age_pred)
            age_labels.extend(age_label_np)
            bmi_preds.extend(bmi_pred)
            bmi_labels.extend(bmi_label_np)
    
    sex_acc = accuracy_score(sex_labels, sex_preds)
    age_acc = accuracy_score(age_labels, age_preds)
    bmi_acc = accuracy_score(bmi_labels, bmi_preds)
    overall_acc = np.mean([sex_acc, age_acc, bmi_acc])
    
    # Debug: Print some sample predictions vs labels for synthetic videos
    if "synthetic" in label_prefix.lower():
        print(f"\n[DEBUG] Sample predictions vs labels (first 10):")
        print(f"Sex - Preds: {sex_preds[:10]}, Labels: {sex_labels[:10]}")
        print(f"Age - Preds: {age_preds[:10]}, Labels: {age_labels[:10]}")
        print(f"BMI - Preds: {bmi_preds[:10]}, Labels: {bmi_labels[:10]}")
        print(f"Unique sex labels: {np.unique(sex_labels)}, Unique sex preds: {np.unique(sex_preds)}")
        print(f"Unique age labels: {np.unique(age_labels)}, Unique age preds: {np.unique(age_preds)}")
        print(f"Unique BMI labels: {np.unique(bmi_labels)}, Unique BMI preds: {np.unique(bmi_preds)}")
    
    return {
        'sex_accuracy': sex_acc,
        'age_accuracy': age_acc,
        'bmi_accuracy': bmi_acc,
        'overall_accuracy': overall_acc,
        'sex_preds': sex_preds,
        'sex_labels': sex_labels,
        'age_preds': age_preds,
        'age_labels': age_labels,
        'bmi_preds': bmi_preds,
        'bmi_labels': bmi_labels,
        'num_samples': len(dataset)
    }


def prepare_synthetic_manifest(variations_manifest_path, output_path, train_manifest_path):
    """Prepare synthetic manifest with proper demographic labels matching training data format"""
    variations_df = pd.read_csv(variations_manifest_path)
    train_df = pd.read_csv(train_manifest_path)
    
    # Get actual age bins from training data
    actual_age_bins = sorted(train_df['age_bin'].unique())
    print(f"Age bins in training data: {actual_age_bins}")
    
    # Create mapping function for age to age_bin
    def map_age_to_bin(age):
        age = float(age)
        # Match the exact bins from training data
        if '0-1' in actual_age_bins and age <= 1:
            return '0-1'
        elif '1-2' in actual_age_bins and age <= 2:
            return '1-2'
        elif '2-3' in actual_age_bins and age <= 3:
            return '2-3'
        elif '2-5' in actual_age_bins and age <= 5:
            return '2-5'
        elif '3-5' in actual_age_bins and age <= 5:
            return '3-5'
        elif '5-8' in actual_age_bins and age <= 8:
            return '5-8'
        elif '6-10' in actual_age_bins and age <= 10:
            return '6-10'
        elif '8-12' in actual_age_bins and age <= 12:
            return '8-12'
        elif '11-15' in actual_age_bins and age <= 15:
            return '11-15'
        elif '12-15' in actual_age_bins and age <= 15:
            return '12-15'
        elif '15-18' in actual_age_bins or '16-18' in actual_age_bins:
            if '15-18' in actual_age_bins:
                return '15-18'
            else:
                return '16-18'
        else:
            # Default to closest bin
            return actual_age_bins[-1] if actual_age_bins else '15-18'
    
    # Create new manifest with synthetic videos
    synthetic_manifest = []
    
    for _, row in variations_df.iterrows():
        # Map variation demographics to labels
        sex = str(row['variation_sex']).strip()
        age = float(row['variation_age'])
        bmi = str(row['variation_bmi']).strip().lower()
        
        # Map age to age_bin using actual bins
        age_bin = map_age_to_bin(age)
        
        # Normalize BMI category
        bmi_category = bmi.capitalize() if isinstance(bmi, str) else 'Normal'
        if bmi_category.lower() == 'normal':
            bmi_category = 'Normal'
        elif bmi_category.lower() == 'overweight':
            bmi_category = 'Overweight'
        elif bmi_category.lower() == 'underweight':
            bmi_category = 'Underweight'
        elif bmi_category.lower() == 'obese':
            bmi_category = 'Obese'
        
        synthetic_manifest.append({
            'file_path': row['synthetic_path'],
            'sex': sex,
            'age': age,
            'age_bin': age_bin,
            'bmi_category': bmi_category,
            'weight': 50.0,  # Dummy value for BMI calculation
            'height': 150.0  # Dummy value for BMI calculation
        })
    
    synthetic_df = pd.DataFrame(synthetic_manifest)
    synthetic_df.to_csv(output_path, index=False)
    print(f"Created synthetic manifest with {len(synthetic_df)} samples")
    print(f"Sex distribution: {synthetic_df['sex'].value_counts().to_dict()}")
    print(f"Age bin distribution: {synthetic_df['age_bin'].value_counts().to_dict()}")
    print(f"BMI distribution: {synthetic_df['bmi_category'].value_counts().to_dict()}")
    return output_path


def main():
    print("\n" + "="*80)
    print("DEMOGRAPHIC CLASSIFIER: REAL vs SYNTHETIC EVALUATION")
    print("="*80 + "\n")
    
    # Load config
    config_path = "ef_prediction/config.yaml"
    if not os.path.exists(config_path):
        config_path = "../ef_prediction/config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load trained model - try improved version first
    checkpoint_path = "checkpoints_improved/best.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "../demographic_classifier/checkpoints_improved/best.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "checkpoints/best.pth"
        if not os.path.exists(checkpoint_path):
            checkpoint_path = "../demographic_classifier/checkpoints/best.pth"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model checkpoint not found")
        print("Please train the model first using train_demographic_classifier.py")
        return
    
    # Determine number of classes from training data
    train_manifest = "data/processed_full/train_manifest_filtered_clean.csv"
    if not os.path.exists(train_manifest):
        train_manifest = "../data/processed_full/train_manifest_filtered_clean.csv"
    train_df = pd.read_csv(train_manifest)
    num_age_bins = len(train_df['age_bin'].unique())
    num_bmi_cats = 4
    
    # Check which model architecture to use based on checkpoint
    checkpoint_state = torch.load(checkpoint_path, map_location=device)
    has_backbone = any('backbone' in key for key in checkpoint_state.keys())
    
    if has_backbone:
        print("Using ImprovedDemographicClassifier3D (R3D-18 backbone)")
        model = ImprovedDemographicClassifier3D(num_age_bins=num_age_bins, num_bmi_cats=num_bmi_cats, pretrained=False).to(device)
    else:
        print("Using DemographicClassifier3D (original architecture)")
        model = DemographicClassifier3D(num_age_bins=num_age_bins, num_bmi_cats=num_bmi_cats).to(device)
    
    model.load_state_dict(checkpoint_state)
    print("✓ Loaded trained model\n")
    
    # Evaluate on REAL videos (validation set)
    print("Evaluating on REAL videos...")
    val_manifest = "data/processed_full/val_manifest.csv"
    video_dir = config["data"]["original_video_dir"]
    if not os.path.exists(val_manifest):
        val_manifest = "../data/processed_full/val_manifest.csv"
        video_dir = f"../{video_dir}"
    real_results = evaluate_on_manifest(
        model=model,
        manifest_path=val_manifest,
        video_root_dir=video_dir,
        device=device,
        config=config,
        label_prefix="Real",
        train_manifest_path=train_manifest
    )
    
    print(f"Real Videos - Sex Acc: {real_results['sex_accuracy']:.4f} | "
          f"Age Acc: {real_results['age_accuracy']:.4f} | "
          f"BMI Acc: {real_results['bmi_accuracy']:.4f} | "
          f"Overall: {real_results['overall_accuracy']:.4f}")
    
    # Prepare and evaluate on SYNTHETIC videos
    print("\nPreparing synthetic manifest...")
    variations_manifest = "use_case_2_demographic_variations/demographic_variations/variations_manifest.csv"
    synthetic_manifest_path = "synthetic_manifest.csv"
    synthetic_video_dir = "use_case_2_demographic_variations/demographic_variations"
    if not os.path.exists(variations_manifest):
        variations_manifest = f"../{variations_manifest}"
        synthetic_manifest_path = f"../demographic_classifier/{synthetic_manifest_path}"
        synthetic_video_dir = f"../{synthetic_video_dir}"
    
    if not os.path.exists(variations_manifest):
        print(f"❌ Variations manifest not found: {variations_manifest}")
        return
    
    prepare_synthetic_manifest(variations_manifest, synthetic_manifest_path, train_manifest)
    
    print("Evaluating on SYNTHETIC videos...")
    synthetic_results = evaluate_on_manifest(
        model=model,
        manifest_path=synthetic_manifest_path,
        video_root_dir=synthetic_video_dir,
        device=device,
        config=config,
        label_prefix="Synthetic",
        train_manifest_path=train_manifest
    )
    
    print(f"Synthetic Videos - Sex Acc: {synthetic_results['sex_accuracy']:.4f} | "
          f"Age Acc: {synthetic_results['age_accuracy']:.4f} | "
          f"BMI Acc: {synthetic_results['bmi_accuracy']:.4f} | "
          f"Overall: {synthetic_results['overall_accuracy']:.4f}")
    
    # Calculate differences
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    sex_diff = abs(real_results['sex_accuracy'] - synthetic_results['sex_accuracy'])
    age_diff = abs(real_results['age_accuracy'] - synthetic_results['age_accuracy'])
    bmi_diff = abs(real_results['bmi_accuracy'] - synthetic_results['bmi_accuracy'])
    overall_diff = abs(real_results['overall_accuracy'] - synthetic_results['overall_accuracy'])
    
    print(f"\nSex Accuracy:")
    print(f"  Real:      {real_results['sex_accuracy']:.4f}")
    print(f"  Synthetic: {synthetic_results['sex_accuracy']:.4f}")
    print(f"  Difference: {sex_diff:.4f} ({'✓ Similar' if sex_diff < 0.1 else '⚠️ Different'})")
    
    print(f"\nAge Accuracy:")
    print(f"  Real:      {real_results['age_accuracy']:.4f}")
    print(f"  Synthetic: {synthetic_results['age_accuracy']:.4f}")
    print(f"  Difference: {age_diff:.4f} ({'✓ Similar' if age_diff < 0.1 else '⚠️ Different'})")
    
    print(f"\nBMI Accuracy:")
    print(f"  Real:      {real_results['bmi_accuracy']:.4f}")
    print(f"  Synthetic: {synthetic_results['bmi_accuracy']:.4f}")
    print(f"  Difference: {bmi_diff:.4f} ({'✓ Similar' if bmi_diff < 0.1 else '⚠️ Different'})")
    
    print(f"\nOverall Accuracy:")
    print(f"  Real:      {real_results['overall_accuracy']:.4f}")
    print(f"  Synthetic: {synthetic_results['overall_accuracy']:.4f}")
    print(f"  Difference: {overall_diff:.4f}")
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    if overall_diff < 0.05:
        print("✓ EXCELLENT: Synthetic videos encode demographic signals nearly identically to real videos.")
        print("  This strongly validates that demographic conditioning is working correctly.")
    elif overall_diff < 0.10:
        print("✓ GOOD: Synthetic videos encode demographic signals similarly to real videos.")
        print("  Demographic conditioning is effective, with minor differences.")
    elif overall_diff < 0.15:
        print("⚠️ MODERATE: Some difference in demographic encoding between real and synthetic videos.")
        print("  Demographic conditioning is partially effective but may need improvement.")
    else:
        print("❌ POOR: Significant difference in demographic encoding.")
        print("  Demographic conditioning may not be working as intended.")
    
    # Save results
    results_dir = Path("results")
    if not results_dir.exists():
        results_dir = Path("../demographic_classifier/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_results = {
        'real_videos': {
            'sex_accuracy': float(real_results['sex_accuracy']),
            'age_accuracy': float(real_results['age_accuracy']),
            'bmi_accuracy': float(real_results['bmi_accuracy']),
            'overall_accuracy': float(real_results['overall_accuracy']),
            'num_samples': int(real_results['num_samples'])
        },
        'synthetic_videos': {
            'sex_accuracy': float(synthetic_results['sex_accuracy']),
            'age_accuracy': float(synthetic_results['age_accuracy']),
            'bmi_accuracy': float(synthetic_results['bmi_accuracy']),
            'overall_accuracy': float(synthetic_results['overall_accuracy']),
            'num_samples': int(synthetic_results['num_samples'])
        },
        'differences': {
            'sex_accuracy_diff': float(sex_diff),
            'age_accuracy_diff': float(age_diff),
            'bmi_accuracy_diff': float(bmi_diff),
            'overall_accuracy_diff': float(overall_diff)
        },
        'interpretation': {
            'overall_diff': float(overall_diff),
            'status': 'excellent' if overall_diff < 0.05 else 'good' if overall_diff < 0.10 else 'moderate' if overall_diff < 0.15 else 'poor'
        }
    }
    
    with open(results_dir / "real_vs_synthetic_comparison.json", "w") as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_dir / 'real_vs_synthetic_comparison.json'}")
    print("\n🎉 Evaluation completed!\n")


if __name__ == "__main__":
    main()
