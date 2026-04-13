"""
Check training progress and switch to improved version if not performing well.
"""
import subprocess
import time
import os
from pathlib import Path


def check_current_training():
    """Check if current training is running and get latest metrics"""
    try:
        output = subprocess.run(
            "byobu capture-pane -t demographic_training -p",
            shell=True, capture_output=True, text=True, timeout=5
        ).stdout
        
        # Look for validation metrics
        lines = output.split('\n')
        metrics = {}
        
        for i, line in enumerate(lines):
            if 'Val' in line and ('Loss' in line or 'Acc' in line):
                # Try to extract metrics
                if 'Loss:' in line:
                    try:
                        loss_str = line.split('Loss:')[1].split()[0]
                        metrics['val_loss'] = float(loss_str)
                    except:
                        pass
                if 'Sex Acc:' in line:
                    try:
                        acc_str = line.split('Sex Acc:')[1].split()[0]
                        metrics['val_sex_acc'] = float(acc_str)
                    except:
                        pass
        
        return metrics, output
    except:
        return {}, ""


def should_switch_to_improved(metrics):
    """Determine if we should switch to improved version"""
    if not metrics:
        return False, "No metrics yet"
    
    # Check if validation loss is very high (> 3.0)
    if metrics.get('val_loss') and metrics['val_loss'] > 3.0:
        return True, f"Validation loss too high: {metrics['val_loss']:.4f}"
    
    # Check if sex accuracy is very low (< 0.4)
    if metrics.get('val_sex_acc') and metrics['val_sex_acc'] < 0.4:
        return True, f"Sex accuracy too low: {metrics['val_sex_acc']:.4f}"
    
    return False, "Metrics acceptable"


def switch_to_improved():
    """Stop current training and start improved version"""
    print("\n" + "="*80)
    print("SWITCHING TO IMPROVED VERSION")
    print("="*80 + "\n")
    
    # Kill current training
    print("Stopping current training...")
    subprocess.run("pkill -f train_demographic_classifier", shell=True)
    time.sleep(2)
    
    # Start improved version in byobu
    print("Starting improved version...")
    cmd = """cd /data/home/sai/Documents/EchoNet-Pediatric-BIGAN-AUGMENTATION && 
             python demographic_classifier/train_demographic_classifier_improved.py"""
    
    subprocess.run(
        f'byobu new-session -d -s demographic_training_improved "{cmd}; echo Training completed! Press any key...; read"',
        shell=True
    )
    
    print("✓ Improved version started in byobu session: demographic_training_improved")
    print("  Attach with: byobu attach -t demographic_training_improved\n")


def main():
    print("\n" + "="*80)
    print("TRAINING QUALITY CHECKER")
    print("="*80 + "\n")
    
    metrics, output = check_current_training()
    
    if metrics:
        print("Current Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        print()
        
        should_switch, reason = should_switch_to_improved(metrics)
        
        if should_switch:
            print(f"⚠️  {reason}")
            print("\nSwitching to improved version...")
            switch_to_improved()
        else:
            print(f"✓ {reason}")
            print("\nCurrent training is performing well. Continue monitoring...")
    else:
        print("⏳ No metrics found yet. Training still in early stages.")
        print("   Will check again after more epochs complete.\n")


if __name__ == "__main__":
    main()
