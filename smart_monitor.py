"""
Smart training monitor for UC3 128x128 reconstruction.

Watches training_uc3_128x128.log every 60s and:
  - Prints a live loss table
  - Detects plateau / divergence / early convergence
  - Kills training and triggers a fix if loss is bad
  - Writes a report to monitor_report.txt
"""

import subprocess
import time
import re
import os
import signal
from pathlib import Path
from datetime import datetime

LOG    = Path("/data/home/sai/Documents/EchoNet-Pediatric-BIGAN-AUGMENTATION/training_uc3_128x128.log")
REPORT = Path("/data/home/sai/Documents/EchoNet-Pediatric-BIGAN-AUGMENTATION/monitor_report.txt")
CKPT   = Path("/data/home/sai/Documents/EchoNet-Pediatric-BIGAN-AUGMENTATION/use_case_3_perfect_reconstruction/ckpt_uc3_128x128_T32")
SCRIPT = Path("/data/home/sai/Documents/EchoNet-Pediatric-BIGAN-AUGMENTATION/use_case_3_perfect_reconstruction")

# ── Thresholds ────────────────────────────────────────────────────────────────
PLATEAU_MIN_EPOCHS  = 5     # start checking plateau only after this many epochs
PLATEAU_WINDOW      = 3     # consecutive epochs with < min_improvement = plateau
PLATEAU_MIN_IMPROV  = 0.002 # min absolute L1 drop per epoch to count as progress
DIVERGE_THRESHOLD   = 0.05  # if loss rises above this after epoch 3 → bad
EARLY_STOP_L1       = 0.005 # if loss drops below this → good enough, stop early
STALE_TIMEOUT_MIN   = 45    # if no new log output for this many minutes → hung

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_losses():
    if not LOG.exists():
        return []
    text = LOG.read_text(errors="replace")
    return [float(m) for m in re.findall(r"epoch \d+ mean L1: ([0-9.]+)", text)]

def get_current_batch():
    if not LOG.exists():
        return None, None
    text = LOG.read_text(errors="replace")
    matches = re.findall(r"epoch (\d+)/50:\s+\d+%\|[^|]+\|\s*(\d+)/1556", text)
    if matches:
        ep, bat = matches[-1]
        return int(ep), int(bat)
    return None, None

def kill_training():
    os.system("pkill -f 'train_reconstruction.py' 2>/dev/null")
    time.sleep(2)
    os.system("pkill -9 -f 'train_reconstruction.py' 2>/dev/null")

def gpu_info():
    try:
        r = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
             "--format=csv,noheader"], text=True).strip()
        mem, util, temp = [x.strip() for x in r.split(",")]
        return f"GPU {mem} | {util} util | {temp}"
    except Exception:
        return "GPU info unavailable"

def log_report(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(REPORT, "a") as f:
        f.write(line + "\n")

def print_table(losses):
    print("\n" + "─" * 52)
    print(f"  {'Epoch':<8} {'L1 Loss':<14} {'Drop':<12} {'Status'}")
    print("─" * 52)
    for i, loss in enumerate(losses):
        epoch = i + 1
        drop  = f"↓{losses[i-1]-loss:.5f}" if i > 0 else "—"
        color = "✅" if i == 0 or losses[i] < losses[i-1] else "⚠️ "
        print(f"  {epoch:<8} {loss:<14.5f} {drop:<12} {color}")
    print("─" * 52, flush=True)

def launch_fallback():
    """Relaunch with smaller lr and base_channels=32 as fallback."""
    log_report("Launching fallback: lr=5e-5, base_channels=32")
    cmd = (
        "byobu send-keys -t uc3_train:0 "
        "'cd /data/home/sai/Documents/EchoNet-Pediatric-BIGAN-AUGMENTATION/"
        "use_case_3_perfect_reconstruction && "
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        "python train_reconstruction.py "
        "--manifest ../data/processed_full/train_manifest_filtered_clean.csv "
        "--checkpoint_dir ./ckpt_uc3_128x128_T32_fallback "
        "--conditioning film --epochs 50 --video_length 32 --video_size 128 "
        "--batch_size 4 --base_channels 32 --lr 5e-5 --lambda_temp 0.1 "
        "--device cuda 2>&1 | tee ../training_uc3_128x128_fallback.log' Enter"
    )
    os.system(cmd)

# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    log_report("Smart monitor started")
    last_log_size  = 0
    stale_since    = time.time()
    check_interval = 60  # seconds

    while True:
        time.sleep(check_interval)
        losses = get_losses()
        n      = len(losses)
        ep, bt = get_current_batch()

        # ── Track staleness ───────────────────────────────────────────────────
        cur_size = LOG.stat().st_size if LOG.exists() else 0
        if cur_size != last_log_size:
            stale_since    = time.time()
            last_log_size  = cur_size
        stale_min = (time.time() - stale_since) / 60

        # ── Header ────────────────────────────────────────────────────────────
        now = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'═'*52}", flush=True)
        print(f"  UC3 Monitor  {now}  |  {gpu_info()}")
        if ep and bt:
            pct = bt / 1556 * 100
            print(f"  Epoch {ep}/50  batch {bt}/1556  ({pct:.0f}%)")
        print_table(losses)

        if n == 0:
            log_report("Waiting for first epoch to complete...")
            continue

        # ── Check: hung / stale ───────────────────────────────────────────────
        if stale_min > STALE_TIMEOUT_MIN:
            log_report(f"⚠️  LOG STALE for {stale_min:.0f} min — process may be hung. Killing.")
            kill_training()
            launch_fallback()
            break

        # ── Check: early convergence (already good enough) ───────────────────
        if losses[-1] < EARLY_STOP_L1:
            log_report(f"✅ Early stop: L1={losses[-1]:.5f} < {EARLY_STOP_L1} — model converged!")
            log_report("Training is finished early. Proceed to Step 2 (UC2 generation).")
            break

        # ── Check: divergence ─────────────────────────────────────────────────
        if n >= 4 and losses[-1] > DIVERGE_THRESHOLD:
            log_report(f"🚨 DIVERGENCE: L1={losses[-1]:.5f} at epoch {n}. Killing & relaunching with fix.")
            kill_training()
            launch_fallback()
            break

        # ── Check: plateau (only after PLATEAU_MIN_EPOCHS) ───────────────────
        if n >= PLATEAU_MIN_EPOCHS + PLATEAU_WINDOW:
            recent = losses[-PLATEAU_WINDOW:]
            drops  = [recent[i-1] - recent[i] for i in range(1, len(recent))]
            if all(d < PLATEAU_MIN_IMPROV for d in drops):
                log_report(
                    f"⚠️  PLATEAU detected (last {PLATEAU_WINDOW} epochs: "
                    f"{[f'{x:.5f}' for x in recent]}). "
                    f"Drops: {[f'{d:.5f}' for d in drops]}."
                )
                # Only stop if loss is still too high
                if losses[-1] > 0.010:
                    log_report("Loss too high and plateaued. Killing & relaunching with lr=5e-5.")
                    kill_training()
                    launch_fallback()
                    break
                else:
                    log_report(f"Loss={losses[-1]:.5f} — plateau is acceptable (low enough). Continuing.")

        # ── Healthy ───────────────────────────────────────────────────────────
        drop_pct = (losses[-2] - losses[-1]) / losses[-2] * 100 if n >= 2 else 0
        log_report(f"Epoch {n}: L1={losses[-1]:.5f}  drop={drop_pct:.1f}%  — OK")

    log_report("Monitor exiting.")


if __name__ == "__main__":
    main()
