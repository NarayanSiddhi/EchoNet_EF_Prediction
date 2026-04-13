"""
Master script to run all demographic evaluation metrics.
Runs Option A (Demographic Classification) and Option B (Distribution Metrics).
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}\n")
    print(f"Running: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n❌ Error running: {description}")
        print(f"Command: {cmd}")
        return False
    else:
        print(f"\n✓ Completed: {description}")
        return True


def main():
    print("\n" + "="*80)
    print("DEMOGRAPHIC EVALUATION PIPELINE")
    print("Option A: Demographic Classification Accuracy")
    print("Option B: Distribution Divergence Metrics")
    print("="*80)
    
    # Check if we're in the right directory
    if not Path("demographic_classifier").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Step 1: Train classifier (if not already trained)
    checkpoint_path = Path("demographic_classifier/checkpoints/best.pth")
    if not checkpoint_path.exists():
        print("\n⚠️  Classifier not found. Training classifier first...")
        if not run_command(
            "cd demographic_classifier && python train_demographic_classifier.py",
            "Training Demographic Classifier"
        ):
            print("\n❌ Training failed. Please check errors above.")
            sys.exit(1)
    else:
        print("\n✓ Classifier already trained. Skipping training step.")
    
    # Step 2: Evaluate real vs synthetic
    if not run_command(
        "cd demographic_classifier && python evaluate_real_vs_synthetic.py",
        "Evaluating Real vs Synthetic Videos (Option A)"
    ):
        print("\n⚠️  Real vs Synthetic evaluation had issues. Continuing...")
    
    # Step 3: Calculate distribution metrics
    if not run_command(
        "cd demographic_classifier && python calculate_distribution_metrics.py",
        "Calculating Distribution Metrics (Option B)"
    ):
        print("\n⚠️  Distribution metrics calculation had issues. Continuing...")
    
    # Summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("\nResults saved in:")
    print("  - demographic_classifier/results/real_videos_metrics.json")
    print("  - demographic_classifier/results/real_vs_synthetic_comparison.json")
    print("  - demographic_classifier/results/distribution_metrics.json")
    print("\n🎉 All evaluations completed!\n")


if __name__ == "__main__":
    main()
