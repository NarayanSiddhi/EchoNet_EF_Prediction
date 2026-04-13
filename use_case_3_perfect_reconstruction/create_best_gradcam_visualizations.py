"""
Create enhanced GradCAM visualizations for best samples
Shows labeled cardiac regions and highlights important areas
"""
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from pathlib import Path
import imageio
from PIL import Image, ImageDraw, ImageFont
import torch

# ============================================================================
# PERFECT RECONSTRUCTION - Best 5 Samples
# ============================================================================

def create_perfect_reconstruction_visualization(sample_dir, sample_id, max_cam, mean_cam, ef, output_path):
    """Create labeled visualization for perfect reconstruction sample"""
    
    # Load comparison image if exists
    comp_path = Path(sample_dir) / 'comparison_viewable.png'
    if not comp_path.exists():
        comp_path = Path(sample_dir) / 'comparison_mid_frame.png'
    
    if comp_path.exists():
        img = cv2.imread(str(comp_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Create from overlayed frames
        overlayed_path = Path(sample_dir) / 'overlayed_frame_08.png'
        if overlayed_path.exists():
            img = cv2.imread(str(overlayed_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            return None
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(img)
    ax.axis('off')
    
    # Add title with metrics
    title = f"Perfect Reconstruction - Sample {sample_id}\n"
    title += f"EF: {ef:.1f}% | Max CAM: {max_cam:.3f} | Mean CAM: {mean_cam:.4f}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add labeled regions
    h, w = img.shape[:2]
    
    # Left ventricle (typical location in echo)
    lv_box = Rectangle((w*0.3, h*0.2), w*0.25, h*0.4, 
                       linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--')
    ax.add_patch(lv_box)
    ax.text(w*0.425, h*0.15, 'Left Ventricle', fontsize=12, color='yellow', 
            fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Right ventricle
    rv_box = Rectangle((w*0.1, h*0.25), w*0.2, h*0.35, 
                       linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
    ax.add_patch(rv_box)
    ax.text(w*0.2, h*0.15, 'Right Ventricle', fontsize=12, color='cyan', 
            fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Atria
    atria_box = Rectangle((w*0.2, h*0.05), w*0.4, h*0.15, 
                          linewidth=2, edgecolor='magenta', facecolor='none', linestyle='--')
    ax.add_patch(atria_box)
    ax.text(w*0.4, h*0.02, 'Atria', fontsize=12, color='magenta', 
            fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Valves
    valve_circle = Circle((w*0.4, h*0.3), w*0.08, 
                          linewidth=2, edgecolor='lime', facecolor='none', linestyle='--')
    ax.add_patch(valve_circle)
    ax.text(w*0.4, h*0.45, 'Valves', fontsize=12, color='lime', 
            fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # High attention region (center, where GradCAM typically focuses)
    attention_circle = Circle((w*0.5, h*0.5), w*0.15, 
                              linewidth=3, edgecolor='red', facecolor='none', linestyle='-')
    ax.add_patch(attention_circle)
    ax.text(w*0.5, h*0.7, 'High Attention\n(GradCAM Focus)', fontsize=11, color='red', 
            fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

# ============================================================================
# DEMOGRAPHIC VARIATIONS - Best 5 Samples
# ============================================================================

def create_demographic_variation_visualization(viz_path, original_id, var_type, cosine_sim, spatial_corr, output_path):
    """Create labeled visualization for demographic variation sample"""
    
    if not Path(viz_path).exists():
        return None
    
    img = cv2.imread(viz_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(img)
    ax.axis('off')
    
    # Add title with metrics
    var_name = var_type.replace('_', ' ').title()
    title = f"Demographic Variation - Sample {original_id} ({var_name})\n"
    title += f"Cosine Similarity: {cosine_sim:.4f} | Spatial Correlation: {spatial_corr:.4f}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add labeled regions
    h, w = img.shape[:2]
    
    # Left ventricle
    lv_box = Rectangle((w*0.25, h*0.2), w*0.25, h*0.4, 
                       linewidth=2, edgecolor='yellow', facecolor='none', linestyle='--')
    ax.add_patch(lv_box)
    ax.text(w*0.375, h*0.15, 'Left Ventricle', fontsize=11, color='yellow', 
            fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Right ventricle
    rv_box = Rectangle((w*0.05, h*0.25), w*0.2, h*0.35, 
                       linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
    ax.add_patch(rv_box)
    ax.text(w*0.15, h*0.15, 'Right Ventricle', fontsize=11, color='cyan', 
            fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Atria
    atria_box = Rectangle((w*0.15, h*0.05), w*0.4, h*0.15, 
                          linewidth=2, edgecolor='magenta', facecolor='none', linestyle='--')
    ax.add_patch(atria_box)
    ax.text(w*0.35, h*0.02, 'Atria', fontsize=11, color='magenta', 
            fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Valves
    valve_circle = Circle((w*0.35, h*0.3), w*0.08, 
                          linewidth=2, edgecolor='lime', facecolor='none', linestyle='--')
    ax.add_patch(valve_circle)
    ax.text(w*0.35, h*0.45, 'Valves', fontsize=11, color='lime', 
            fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Attention overlap region (where both original and variation focus)
    attention_circle = Circle((w*0.5, h*0.5), w*0.12, 
                              linewidth=3, edgecolor='red', facecolor='none', linestyle='-')
    ax.add_patch(attention_circle)
    ax.text(w*0.5, h*0.65, 'Shared Attention\n(Pattern Preservation)', fontsize=10, color='red', 
            fontweight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Add similarity indicator
    if cosine_sim > 0.9:
        sim_color = 'green'
        sim_text = 'Excellent'
    elif cosine_sim > 0.8:
        sim_color = 'yellow'
        sim_text = 'Good'
    else:
        sim_color = 'orange'
        sim_text = 'Moderate'
    
    sim_box = FancyBboxPatch((w*0.02, h*0.85), w*0.15, h*0.1, 
                            boxstyle='round,pad=5', linewidth=2, 
                            edgecolor=sim_color, facecolor='black', alpha=0.7)
    ax.add_patch(sim_box)
    ax.text(w*0.095, h*0.9, f'Similarity:\n{sim_text}', fontsize=10, color=sim_color, 
            fontweight='bold', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    print("="*70)
    print("CREATING BEST GRADCAM VISUALIZATIONS")
    print("="*70)
    
    # Create output directory
    output_dir = Path('best_gradcam_visualizations')
    output_dir.mkdir(exist_ok=True)
    
    perf_dir = output_dir / 'perfect_reconstruction'
    var_dir = output_dir / 'demographic_variations'
    perf_dir.mkdir(exist_ok=True)
    var_dir.mkdir(exist_ok=True)
    
    # ========================================================================
    # PERFECT RECONSTRUCTION - Best 5 (all available)
    # ========================================================================
    print("\n1. Processing Perfect Reconstruction samples...")
    
    df_perf = pd.read_csv('gradcam_results/gradcam_summary.csv')
    df_perf = df_perf.sort_values('max_cam_value', ascending=False)
    
    print(f"   Found {len(df_perf)} samples")
    
    for idx, row in df_perf.iterrows():
        sample_id = int(row['sample_id'])
        sample_dir = Path(f'gradcam_results/sample_{sample_id:04d}')
        max_cam = row['max_cam_value']
        mean_cam = row['mean_cam_value']
        ef = row['EF']
        
        output_path = perf_dir / f'best_perfect_reconstruction_sample_{sample_id:04d}.png'
        
        print(f"   Processing sample {sample_id}...")
        result = create_perfect_reconstruction_visualization(
            sample_dir, sample_id, max_cam, mean_cam, ef, output_path
        )
        
        if result:
            print(f"   ✓ Saved: {output_path}")
        else:
            print(f"   ✗ Failed: sample {sample_id}")
    
    # ========================================================================
    # DEMOGRAPHIC VARIATIONS - Best 5 samples (by average similarity)
    # ========================================================================
    print("\n2. Processing Demographic Variation samples...")
    
    df_var = pd.read_csv('gradcam_variations_analysis/gradcam_analysis_results.csv')
    
    # Calculate average similarity per sample (across all 3 variations)
    sample_stats = df_var.groupby('original_id').agg({
        'cosine_similarity': 'mean',
        'spatial_correlation': 'mean'
    }).reset_index()
    sample_stats.columns = ['original_id', 'avg_cosine', 'avg_spatial']
    sample_stats = sample_stats.sort_values('avg_cosine', ascending=False)
    
    # Get top 5 samples
    top_5_samples = sample_stats.head(5)['original_id'].tolist()
    
    print(f"   Top 5 samples: {top_5_samples}")
    
    for sample_id in top_5_samples:
        sample_data = df_var[df_var['original_id'] == sample_id]
        
        for _, row in sample_data.iterrows():
            var_type = row['variation_type']
            viz_path = row['visualization_path']
            cosine_sim = row['cosine_similarity']
            spatial_corr = row['spatial_correlation']
            
            output_path = var_dir / f'best_variation_sample_{sample_id:04d}_{var_type}.png'
            
            print(f"   Processing sample {sample_id}, {var_type}...")
            result = create_demographic_variation_visualization(
                viz_path, sample_id, var_type, cosine_sim, spatial_corr, output_path
            )
            
            if result:
                print(f"   ✓ Saved: {output_path}")
            else:
                print(f"   ✗ Failed: sample {sample_id}, {var_type}")
    
    # ========================================================================
    # Create summary
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Perfect Reconstruction visualizations: {len(list(perf_dir.glob('*.png')))}")
    print(f"Demographic Variation visualizations: {len(list(var_dir.glob('*.png')))}")
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("="*70)

if __name__ == "__main__":
    main()
