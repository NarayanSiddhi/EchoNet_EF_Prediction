"""
Retraining Guide for UC2 & UC3 at 128×128 Resolution
====================================================

This script provides commands and validation for retraining Use Cases 2 & 3
at 128×128 spatial resolution with 32 frames.

Author: AI Assistant
Date: 2026-04-09
"""

import subprocess
import sys
from pathlib import Path
import torch


def check_gpu_memory():
    """Check available GPU memory"""
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available. Training will be very slow on CPU.")
        return None
    
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"✓ Total GPU memory: {gpu_mem_gb:.1f} GB")
    
    if gpu_mem_gb < 8:
        print("⚠️  WARNING: Less than 8GB GPU memory. Use --batch_size 1 and --base_channels 32")
    elif gpu_mem_gb < 12:
        print("⚠️  Moderate GPU memory. Recommended: --batch_size 2 and --base_channels 64")
    else:
        print("✓ Good GPU memory. Can use --batch_size 4")
    
    return gpu_mem_gb


def verify_paths():
    """Verify required paths exist"""
    required_paths = {
        "Training script": "use_case_3_perfect_reconstruction/train_reconstruction.py",
        "Model definition": "use_case_3_perfect_reconstruction/models.py",
        "Generation script": "use_case_2_demographic_variations/generate_demographic_variations_fixed.py",
        "Manifest (option 1)": "data/processed_full/train_manifest_filtered_clean.csv",
        "Manifest (option 2)": "data/processed/train_manifest.csv",
    }
    
    print("\nVerifying paths:")
    print("-" * 70)
    
    all_ok = True
    manifest_found = None
    
    for name, path in required_paths.items():
        path_obj = Path(path)
        if path_obj.exists():
            print(f"✓ {name:<30} {path}")
            if "Manifest" in name:
                manifest_found = path
        else:
            if "Manifest" in name:
                print(f"⚠  {name:<30} {path} (not found)")
            else:
                print(f"✗ {name:<30} {path} (MISSING)")
                all_ok = False
    
    if not manifest_found:
        print("\n✗ ERROR: No manifest file found. You need a manifest CSV with 'processed_path' column.")
        all_ok = False
    
    return all_ok, manifest_found


def print_training_commands(manifest_path, gpu_mem_gb=None):
    """Print the exact training commands"""
    
    # Determine batch size and base channels based on GPU memory
    if gpu_mem_gb is None:
        batch_size = 2
        base_channels = 64
        note = "CUDA not available"
    elif gpu_mem_gb < 8:
        batch_size = 1
        base_channels = 32
        note = "Low GPU memory"
    elif gpu_mem_gb < 12:
        batch_size = 2
        base_channels = 64
        note = "Moderate GPU memory"
    else:
        batch_size = 4
        base_channels = 64
        note = "Good GPU memory"
    
    print(f"\n{'='*70}")
    print("STEP 1: Train UC3 (Perfect Reconstruction)")
    print(f"{'='*70}")
    print(f"Configuration: {note}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Base channels: {base_channels}")
    print(f"  - Spatial size: 128×128")
    print(f"  - Temporal length: 32 frames")
    print(f"  - Conditioning: FiLM (recommended)")
    print()
    
    uc3_cmd = f"""cd use_case_3_perfect_reconstruction && \\
python train_reconstruction.py \\
  --manifest ../{manifest_path} \\
  --checkpoint_dir ./ckpt_uc3_128x128_T32 \\
  --conditioning film \\
  --epochs 50 \\
  --video_length 32 \\
  --video_size 128 \\
  --batch_size {batch_size} \\
  --base_channels {base_channels} \\
  --lr 1e-4 \\
  --lambda_temp 0.1 \\
  --device cuda
"""
    
    print("Command:")
    print("-" * 70)
    print(uc3_cmd.strip())
    print("-" * 70)
    print()
    print("Expected output:")
    print("  - Checkpoints saved to: use_case_3_perfect_reconstruction/ckpt_uc3_128x128_T32/")
    print("  - Final model: recon_best.pt")
    print("  - Training time: ~2-4 hours (depends on dataset size and GPU)")
    print()
    
    print(f"\n{'='*70}")
    print("STEP 2: Generate UC2 Demographic Variations")
    print(f"{'='*70}")
    print("Using the trained UC3 model to generate demographic variations")
    print()
    
    uc2_cmd = f"""cd use_case_2_demographic_variations && \\
python generate_demographic_variations_fixed.py \\
  --manifest ../{manifest_path} \\
  --checkpoint ../use_case_3_perfect_reconstruction/ckpt_uc3_128x128_T32/recon_best.pt \\
  --output_dir ../demographic_variations_128x128_T32 \\
  --video_length 32 \\
  --video_size 128 \\
  --diversity_weight 0.3 \\
  --use_reference \\
  --max_videos 100
"""
    
    print("Command:")
    print("-" * 70)
    print(uc2_cmd.strip())
    print("-" * 70)
    print()
    print("Expected output:")
    print("  - Variations saved to: demographic_variations_128x128_T32/")
    print("  - 3 variations per input video (different demographics)")
    print("  - Manifest CSV with results")
    print()
    
    print(f"\n{'='*70}")
    print("STEP 3: Evaluate Quality Metrics")
    print(f"{'='*70}")
    print()
    
    eval_note = """
After generation, evaluate quality metrics (SSIM, PSNR, FID) to compare with
previous 64×64 results. You can use:
  - use_case_3_perfect_reconstruction/evaluate_reconstruction.py (if exists)
  - use_case_1_balance_dataset/evaluate_quality_metrics.py (adapted)
  
Update Table 5 in your paper with the new metrics.
"""
    print(eval_note)
    
    print(f"\n{'='*70}")
    print("Memory Management Tips")
    print(f"{'='*70}")
    print("""
If you encounter OOM (Out of Memory) errors:

1. Reduce batch_size:
   --batch_size 1  (minimum)

2. Reduce base_channels:
   --base_channels 32  (halves parameters, keeps architecture)

3. Use gradient checkpointing (requires code modification):
   Add to train_reconstruction.py:
   torch.utils.checkpoint.checkpoint(model, x, c)

4. Mixed precision training (requires code modification):
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   with autocast():
       output = model(x, c)

5. Clear cache between batches:
   torch.cuda.empty_cache()

Note: 64×64×16 → 128×128×32 is 16× more voxels, so memory usage increases ~16×.
""")


def save_commands_to_file(manifest_path, gpu_mem_gb=None):
    """Save commands to a shell script"""
    
    if gpu_mem_gb is None or gpu_mem_gb < 8:
        batch_size = 2
        base_channels = 64
    elif gpu_mem_gb < 12:
        batch_size = 2
        base_channels = 64
    else:
        batch_size = 4
        base_channels = 64
    
    script_content = f"""#!/bin/bash
# Retrain UC2 & UC3 at 128×128 resolution
# Generated: 2026-04-09

set -e  # Exit on error

echo "=========================================="
echo "UC2/UC3 Retraining at 128×128 Resolution"
echo "=========================================="
echo ""

# Step 1: Train UC3 (Perfect Reconstruction)
echo "STEP 1: Training UC3 Perfect Reconstruction Model..."
cd use_case_3_perfect_reconstruction

python train_reconstruction.py \\
  --manifest ../{manifest_path} \\
  --checkpoint_dir ./ckpt_uc3_128x128_T32 \\
  --conditioning film \\
  --epochs 50 \\
  --video_length 32 \\
  --video_size 128 \\
  --batch_size {batch_size} \\
  --base_channels {base_channels} \\
  --lr 1e-4 \\
  --lambda_temp 0.1 \\
  --device cuda

cd ..
echo "✓ UC3 training complete!"
echo ""

# Step 2: Generate UC2 Demographic Variations
echo "STEP 2: Generating UC2 Demographic Variations..."
cd use_case_2_demographic_variations

python generate_demographic_variations_fixed.py \\
  --manifest ../{manifest_path} \\
  --checkpoint ../use_case_3_perfect_reconstruction/ckpt_uc3_128x128_T32/recon_best.pt \\
  --output_dir ../demographic_variations_128x128_T32 \\
  --video_length 32 \\
  --video_size 128 \\
  --diversity_weight 0.3 \\
  --use_reference \\
  --max_videos 100

cd ..
echo "✓ UC2 generation complete!"
echo ""

echo "=========================================="
echo "Retraining Complete!"
echo "=========================================="
echo "Results:"
echo "  - UC3 checkpoints: use_case_3_perfect_reconstruction/ckpt_uc3_128x128_T32/"
echo "  - UC2 variations: demographic_variations_128x128_T32/"
echo ""
echo "Next steps:"
echo "  1. Run cardiac cycle coverage analysis: python analyze_cardiac_cycle_coverage.py"
echo "  2. Evaluate quality metrics (SSIM, PSNR, FID)"
echo "  3. Update paper with new resolution and metrics"
"""
    
    script_path = Path("retrain_uc2_uc3_128x128.sh")
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    
    print(f"\n✓ Commands saved to: {script_path}")
    print(f"  Run with: ./{script_path}")


def main():
    print("=" * 70)
    print("UC2 & UC3 Retraining Guide: 128×128 Resolution")
    print("=" * 70)
    
    # Check GPU
    print("\nChecking GPU availability:")
    print("-" * 70)
    gpu_mem_gb = check_gpu_memory()
    
    # Verify paths
    all_ok, manifest_path = verify_paths()
    
    if not all_ok:
        print("\n✗ ERROR: Required files missing. Please check paths above.")
        sys.exit(1)
    
    # Print commands
    print_training_commands(manifest_path, gpu_mem_gb)
    
    # Save to script
    save_commands_to_file(manifest_path, gpu_mem_gb)
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
1. Review the commands above
2. Run the training script: ./retrain_uc2_uc3_128x128.sh
   OR run commands manually (copy-paste from above)
3. Monitor training progress (should see decreasing L1 loss)
4. After training, run: python analyze_cardiac_cycle_coverage.py
5. Update your paper with:
   - New resolution: 128×128 (instead of 64×64)
   - New frame count: 32 frames (instead of 16)
   - Cardiac cycle coverage justification
   - New SSIM/PSNR metrics from evaluation
""")


if __name__ == "__main__":
    main()
