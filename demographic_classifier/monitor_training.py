"""
Monitor training progress and check if it's improving.
If not improving, suggest alternative approaches.
"""
import json
import time
import os
from pathlib import Path
import subprocess


def check_training_status():
    """Check if training is running"""
    result = subprocess.run(
        "ps aux | grep 'python.*train_demographic_classifier' | grep -v grep",
        shell=True, capture_output=True, text=True
    )
    return len(result.stdout.strip()) > 0


def get_latest_metrics():
    """Get latest training metrics from history file"""
    history_path = Path("demographic_classifier/checkpoints/training_history.json")
    if not history_path.exists():
        return None
    
    with open(history_path) as f:
        history = json.load(f)
    
    if not history.get('val_loss'):
        return None
    
    epochs = len(history['val_loss'])
    latest = {
        'epoch': epochs,
        'val_loss': history['val_loss'][-1],
        'val_sex_acc': history['val_sex_acc'][-1] if history.get('val_sex_acc') else None,
        'val_age_acc': history['val_age_acc'][-1] if history.get('val_age_acc') else None,
        'val_bmi_acc': history['val_bmi_acc'][-1] if history.get('val_bmi_acc') else None,
    }
    
    # Check if improving
    if epochs >= 3:
        recent_losses = history['val_loss'][-3:]
        improving = recent_losses[-1] < recent_losses[0]
        latest['improving'] = improving
        latest['loss_trend'] = 'decreasing' if improving else 'increasing/stable'
    else:
        latest['improving'] = None
        latest['loss_trend'] = 'too early to tell'
    
    return latest


def check_byobu_output():
    """Check recent output from byobu session"""
    try:
        result = subprocess.run(
            "byobu capture-pane -t demographic_training -p",
            shell=True, capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().split('\n')
        return lines[-20:] if lines else []
    except:
        return []


def suggest_improvements(metrics):
    """Suggest improvements if training not improving"""
    suggestions = []
    
    if metrics and metrics.get('improving') is False:
        suggestions.append("⚠️ Training not improving - consider:")
        suggestions.append("  1. Reduce learning rate (current: 0.001)")
        suggestions.append("  2. Increase model capacity (more channels)")
        suggestions.append("  3. Add data augmentation")
        suggestions.append("  4. Use pretrained 3D CNN backbone")
        suggestions.append("  5. Adjust loss weights for imbalanced classes")
    
    if metrics and metrics.get('val_loss') and metrics['val_loss'] > 2.0:
        suggestions.append("⚠️ High validation loss - model may be underfitting")
        suggestions.append("  - Consider larger model or more training epochs")
    
    if metrics and metrics.get('val_sex_acc') and metrics['val_sex_acc'] < 0.5:
        suggestions.append("⚠️ Sex classification accuracy < 50% - worse than random")
        suggestions.append("  - Check class imbalance")
        suggestions.append("  - Use weighted loss")
    
    return suggestions


def main():
    print("\n" + "="*80)
    print("TRAINING MONITOR")
    print("="*80 + "\n")
    
    if not check_training_status():
        print("❌ Training is not running!")
        return
    
    print("✓ Training is running\n")
    
    # Check byobu output
    output = check_byobu_output()
    if output:
        print("Recent training output:")
        for line in output[-5:]:
            print(f"  {line}")
        print()
    
    # Check metrics
    metrics = get_latest_metrics()
    if metrics:
        print(f"Epoch: {metrics['epoch']}")
        print(f"Validation Loss: {metrics['val_loss']:.4f}")
        if metrics['val_sex_acc']:
            print(f"Val Sex Accuracy: {metrics['val_sex_acc']:.4f}")
        if metrics['val_age_acc']:
            print(f"Val Age Accuracy: {metrics['val_age_acc']:.4f}")
        if metrics['val_bmi_acc']:
            print(f"Val BMI Accuracy: {metrics['val_bmi_acc']:.4f}")
        print(f"Trend: {metrics['loss_trend']}")
        print()
        
        suggestions = suggest_improvements(metrics)
        if suggestions:
            print("\n".join(suggestions))
            print()
    else:
        print("⏳ Training in progress (epoch 1)...")
        print("   Waiting for first validation metrics...\n")
    
    # Check checkpoint
    checkpoint = Path("demographic_classifier/checkpoints/best.pth")
    if checkpoint.exists():
        size_mb = checkpoint.stat().st_size / (1024 * 1024)
        print(f"✓ Best model saved ({size_mb:.1f} MB)")
    else:
        print("⏳ No checkpoint yet (waiting for first improvement)")
    
    print()


if __name__ == "__main__":
    main()
