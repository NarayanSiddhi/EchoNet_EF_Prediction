#!/bin/bash
# Continuous monitoring script for demographic classifier training

cd /data/home/sai/Documents/EchoNet-Pediatric-BIGAN-AUGMENTATION

echo "Starting continuous monitoring..."
echo "Will check every 10 minutes"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    echo "=========================================="
    echo "Training Monitor - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # Check if training is running
    if ps aux | grep -q "train_demographic_classifier_improved" | grep -v grep; then
        echo "✓ Training is running"
        echo ""
        
        # Get latest output
        echo "Latest Training Output:"
        echo "----------------------"
        timeout 5 byobu capture-pane -t demographic_training_improved -p 2>/dev/null | tail -20
        echo ""
        
        # Check for checkpoints
        if [ -f "demographic_classifier/checkpoints_improved/best.pth" ]; then
            echo "✓ Best model checkpoint exists"
            ls -lh demographic_classifier/checkpoints_improved/best.pth
        fi
        
        # Check history if exists
        if [ -f "demographic_classifier/checkpoints_improved/training_history.json" ]; then
            echo ""
            echo "Training History Summary:"
            python3 -c "
import json
try:
    with open('demographic_classifier/checkpoints_improved/training_history.json') as f:
        h = json.load(f)
    epochs = len(h.get('val_loss', []))
    if epochs > 0:
        print(f'  Epochs completed: {epochs}')
        print(f'  Best Val Loss: {min(h[\"val_loss\"]):.4f}')
        print(f'  Latest Val Loss: {h[\"val_loss\"][-1]:.4f}')
        if h.get('val_sex_acc'):
            print(f'  Latest Sex Acc: {h[\"val_sex_acc\"][-1]:.4f}')
        if h.get('val_age_acc'):
            print(f'  Latest Age Acc: {h[\"val_age_acc\"][-1]:.4f}')
        if h.get('val_bmi_acc'):
            print(f'  Latest BMI Acc: {h[\"val_bmi_acc\"][-1]:.4f}')
        
        # Check if improving
        if epochs >= 3:
            recent = h['val_loss'][-3:]
            if recent[-1] < recent[0]:
                print('  Trend: ✓ Improving')
            else:
                print('  Trend: ⚠️ Not improving')
except Exception as e:
    print(f'  Error reading history: {e}')
"
        fi
    else
        echo "❌ Training is not running"
        echo "Checking if it completed..."
        if [ -f "demographic_classifier/checkpoints_improved/final.pth" ]; then
            echo "✓ Training appears to have completed!"
        fi
    fi
    
    echo ""
    echo "Next check in 10 minutes..."
    echo "(Press Ctrl+C to stop monitoring)"
    sleep 600
done
