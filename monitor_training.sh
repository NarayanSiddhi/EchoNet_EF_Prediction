#!/bin/bash
LOG=/data/home/sai/Documents/EchoNet-Pediatric-BIGAN-AUGMENTATION/training_uc3_128x128.log
CKPT=/data/home/sai/Documents/EchoNet-Pediatric-BIGAN-AUGMENTATION/use_case_3_perfect_reconstruction/ckpt_uc3_128x128_T32

clear
while true; do
    tput cup 0 0
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║          UC3 128×128 Training Monitor  —  $(date '+%Y-%m-%d %H:%M:%S')         ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "─── GPU ──────────────────────────────────────────────────────────────"
    nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu,temperature.gpu \
               --format=csv,noheader | \
    awk -F', ' '{printf "  %-22s  Mem: %s used / %s free   GPU: %s   Temp: %s\n",$1,$2,$3,$4,$5}'
    echo ""
    echo "─── Progress ─────────────────────────────────────────────────────────"
    # Extract latest tqdm line (last real progress line)
    LATEST=$(grep -oP 'epoch \d+/50:\s+\d+%.*?it/s\]' "$LOG" 2>/dev/null | tail -1)
    if [ -n "$LATEST" ]; then
        echo "  $LATEST"
    else
        echo "  Starting up..."
    fi
    echo ""
    echo "─── Epoch Losses ─────────────────────────────────────────────────────"
    grep "epoch .* mean L1:" "$LOG" 2>/dev/null | tail -10 || echo "  (none completed yet)"
    echo ""
    echo "─── Checkpoints ──────────────────────────────────────────────────────"
    ls -lh "$CKPT"/*.pt 2>/dev/null | awk '{print "  "$NF"  "$5}' || echo "  none saved yet"
    echo ""
    echo "─── Process ──────────────────────────────────────────────────────────"
    PID=$(pgrep -f "train_reconstruction.py" | head -1)
    if [ -n "$PID" ]; then
        echo "  PID $PID  RUNNING  $(ps -p $PID -o etime= | tr -d ' ') elapsed"
    else
        echo "  ✗ Process NOT running"
    fi
    echo ""
    echo "  Press Ctrl+C to exit monitor. Training continues in window 0."
    sleep 5
done
