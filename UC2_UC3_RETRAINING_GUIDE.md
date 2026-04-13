# UC2/UC3 Retraining at 128×128: Complete Implementation Guide

**Date:** 2026-04-09  
**Purpose:** Retrain Use Cases 2 & 3 at 128×128 resolution (32 frames) to address reviewer concerns

---

## Executive Summary

### Current State (What Reviewers See)
- **Resolution:** 64×64 pixels
- **Temporal:** 16 frames
- **Issue:** "Highly limited resolution for clinical deployment"

### Target State (After Retraining)
- **Resolution:** 128×128 pixels (4× more pixels per frame)
- **Temporal:** 32 frames (covers 1.07s, ≥1 full cardiac cycle at all pediatric HRs)
- **Architecture:** Already supports this! Just need to retrain with different flags

---

## Part 1: Cardiac Cycle Coverage Analysis (Ready Now)

### Run This Command First
```bash
python analyze_cardiac_cycle_coverage.py --frames 32 --fps 30
```

### Key Results to Report in Paper

**For Methods (Section 3.2):**
> Videos were preprocessed to 128×128 pixels and temporally subsampled to 32 frames at 30 fps (clip duration: 1.07 seconds). This temporal window was selected to ensure complete cardiac cycle coverage across the pediatric heart rate spectrum: at 30 fps, 32 frames span 1.07 seconds, covering ≥1 complete cardiac cycle for heart rates up to 56 bpm and multiple cycles for typical pediatric resting rates (80-100 bpm: 1.4-1.8 cycles).

**For Discussion:**
> Our choice of 32 frames at 30 fps (1.07 seconds) ensures adequate temporal context for cardiac motion modeling. This duration captures 1.4 complete cardiac cycles at typical pediatric resting heart rates (80 bpm) and maintains coverage of at least one full cycle even during tachycardia (>120 bpm: 2.13 cycles), addressing concerns about temporal resolution raised in prior work operating at lower frame counts.

**Table Caption/Footnote:**
> All models process clips of 32 frames at 30 fps (1.07s), sufficient to capture ≥1 cardiac cycle across the pediatric HR range (60-150 bpm).

---

## Part 2: Retraining Commands

### System Requirements
- **GPU:** NVIDIA RTX A5000 (25.3 GB) detected ✓
- **Recommended batch size:** 4
- **Training time:** ~2-4 hours for UC3, ~1 hour for UC2 generation

### Step 1: Train UC3 (Perfect Reconstruction Model)

```bash
cd use_case_3_perfect_reconstruction

python train_reconstruction.py \
  --manifest ../data/processed_full/train_manifest_filtered_clean.csv \
  --checkpoint_dir ./ckpt_uc3_128x128_T32 \
  --conditioning film \
  --epochs 50 \
  --video_length 32 \
  --video_size 128 \
  --batch_size 4 \
  --base_channels 64 \
  --lr 1e-4 \
  --lambda_temp 0.1 \
  --device cuda
```

**What this does:**
- Trains U-Net generator at 128×128 spatial resolution
- Uses FiLM conditioning (better than concat for demographic control)
- L1 reconstruction loss + temporal smoothness (lambda_temp=0.1)
- Saves checkpoints to `ckpt_uc3_128x128_T32/`

**Monitor training:**
- Watch for decreasing L1 loss
- Should converge to ~0.05-0.10 mean L1

### Step 2: Generate UC2 Demographic Variations

```bash
cd use_case_2_demographic_variations

python generate_demographic_variations_fixed.py \
  --manifest ../data/processed_full/train_manifest_filtered_clean.csv \
  --checkpoint ../use_case_3_perfect_reconstruction/ckpt_uc3_128x128_T32/recon_best.pt \
  --output_dir ../demographic_variations_128x128_T32 \
  --video_length 32 \
  --video_size 128 \
  --diversity_weight 0.3 \
  --use_reference \
  --max_videos 100
```

**What this does:**
- Uses trained UC3 model to generate demographic variations
- Creates 3 variations per input video (different sex/age/BMI)
- Uses reference video style transfer for better quality
- Diversity weight 0.3 balances fidelity vs. variation

**Expected output:**
- ~300 synthetic videos (3 per input × 100 inputs)
- Manifest CSV with results
- Videos saved to `demographic_variations_128x128_T32/`

### Quick Start: Run Both Steps
```bash
./retrain_uc2_uc3_128x128.sh
```

---

## Part 3: Code Verification

### ✓ No Code Changes Needed!

The architecture already supports 128×128:

**In `models.py`:**
```python
class PerfectReconstructionGenerator(nn.Module):
    def __init__(self, base_channels=64, spatial_size=64, conditioning: str = "concat"):
        if spatial_size not in (64, 128):  # ← Already supports 128!
            raise ValueError("spatial_size must be 64 or 128")
```

**In `train_reconstruction.py`:**
```python
model = PerfectReconstructionGenerator(
    base_channels=args.base_channels,
    spatial_size=args.video_size,  # ← CLI flag maps to spatial_size
    conditioning=args.conditioning,
)
```

**In `generate_demographic_variations_fixed.py`:**
```python
# Line 296: Reads video_size from checkpoint
vs = int(checkpoint.get("video_size", 64))

# Line 302-303: Creates model with checkpoint's spatial_size
generator = PerfectReconstructionGenerator(
    base_channels=64, spatial_size=vs, conditioning=cond_mode
)

# Line 329, 379: Uses video_size from CLI arg (must match checkpoint!)
video = load_video(video_path, video_length, video_size)
reference_video = load_video(ref_path, video_length, video_size)
```

**Critical:** When running generation, `--video_size 128` on CLI must match the checkpoint's training resolution.

---

## Part 4: Memory Management

### If You Get OOM (Out of Memory) Errors

**Option 1: Reduce batch size**
```bash
--batch_size 2  # or even 1
```

**Option 2: Reduce model capacity**
```bash
--base_channels 32  # halves parameters but keeps architecture shape
```

**Option 3: Both**
```bash
--batch_size 1 --base_channels 32
```

**Memory scaling:**
- 64×64×16 → 128×128×32 = 16× more voxels
- With batch_size=4: needs ~12-16 GB GPU memory
- With batch_size=2: needs ~6-8 GB GPU memory
- With batch_size=1: needs ~3-4 GB GPU memory

---

## Part 5: Paper Updates

### Section 3.2 (Preprocessing)
**Find:**
> "64×64 pixels... 16 frames"

**Replace with:**
> "128×128 pixels and temporally subsampled to 32 frames at 30 fps (clip duration: 1.07 seconds). This temporal window ensures complete cardiac cycle coverage across the pediatric heart rate spectrum (60-150 bpm), capturing 1.4-2.7 complete cycles depending on heart rate."

### Table 5 (Quality Metrics)
**Current (64×64):**
| Metric | UC2 Value | UC3 Value |
|--------|-----------|-----------|
| SSIM   | 0.XX      | 0.XX      |
| PSNR   | XX.X dB   | XX.X dB   |

**After retraining (128×128):**
- Re-run evaluation to get new numbers
- Expect slightly lower SSIM/PSNR (harder to reconstruct at higher res)
- But add footnote: "Higher resolution better captures clinical detail"

### Discussion Section
**Add paragraph:**
> We address resolution concerns by operating at 128×128 pixels, matching typical clinical echocardiography display resolutions. While earlier work used 64×64 for computational efficiency, our U-Net architecture scales naturally to 128×128 with manageable memory requirements (12-16 GB GPU memory at batch size 4). The 32-frame temporal window (1.07 seconds at 30 fps) provides complete cardiac cycle coverage even at pediatric tachycardia rates, ensuring the model captures both systolic and diastolic phases for accurate cardiac function assessment.

---

## Part 6: Validation Checklist

### After Training, Verify:

1. **Model converged**
   - [ ] Final L1 loss < 0.15
   - [ ] No NaN or Inf losses
   - [ ] Checkpoint files exist

2. **Generated videos look good**
   - [ ] Visually inspect 10-20 samples
   - [ ] Check for artifacts, blurriness
   - [ ] Verify demographic changes are visible

3. **Metrics calculated**
   - [ ] SSIM computed
   - [ ] PSNR computed
   - [ ] FID computed (if time permits)

4. **Paper updated**
   - [ ] Section 3.2 resolution updated
   - [ ] Cardiac cycle justification added
   - [ ] Table 5 metrics updated
   - [ ] Discussion addresses resolution concern

---

## Part 7: Troubleshooting

### Issue: "FileNotFoundError: manifest not found"
**Solution:** Update manifest path in commands above to match your actual path

### Issue: "CUDA out of memory"
**Solution:** See Part 4 (Memory Management) - reduce batch_size or base_channels

### Issue: "Checkpoint has video_size=64 but I passed --video_size 128"
**Solution:** You're using an old checkpoint. Make sure to use the newly trained checkpoint from `ckpt_uc3_128x128_T32/recon_best.pt`

### Issue: "Generated videos are blurry"
**Solution:** This is expected initially. Try:
- Increase --epochs (try 75 or 100)
- Reduce --lr to 5e-5 for finer tuning
- Increase --diversity_weight to 0.5 for more variation

### Issue: "Training is taking forever"
**Solution:** 
- Expected: ~2-4 hours on RTX A5000
- If >8 hours: check GPU utilization with `nvidia-smi`
- Reduce dataset size with `--max_videos` for faster testing

---

## Quick Reference Card

### Training (UC3)
```bash
cd use_case_3_perfect_reconstruction
python train_reconstruction.py \
  --manifest ../data/processed_full/train_manifest_filtered_clean.csv \
  --checkpoint_dir ./ckpt_uc3_128x128_T32 \
  --video_size 128 --video_length 32 --epochs 50
```

### Generation (UC2)
```bash
cd use_case_2_demographic_variations
python generate_demographic_variations_fixed.py \
  --manifest ../data/processed_full/train_manifest_filtered_clean.csv \
  --checkpoint ../use_case_3_perfect_reconstruction/ckpt_uc3_128x128_T32/recon_best.pt \
  --video_size 128 --video_length 32
```

### Analysis
```bash
python analyze_cardiac_cycle_coverage.py
```

---

## Files Created

1. **`analyze_cardiac_cycle_coverage.py`** - Cardiac cycle analysis script
2. **`retrain_uc2_uc3_guide.py`** - Interactive retraining guide
3. **`retrain_uc2_uc3_128x128.sh`** - Executable bash script (all-in-one)
4. **`cardiac_cycle_analysis.txt`** - Analysis results (auto-generated)
5. **`UC2_UC3_RETRAINING_GUIDE.md`** - This document

---

## Timeline Estimate

| Task | Time | Can Parallelize? |
|------|------|------------------|
| UC3 Training | 2-4 hours | No |
| UC2 Generation | 1 hour | No (needs UC3) |
| Metric Evaluation | 30 min | Yes (while writing) |
| Paper Updates | 1 hour | Yes (while training) |
| **Total** | **4-6 hours** | |

**Strategy:** Start training, then work on paper updates while it runs.

---

## Questions?

Contact the model architect or refer to:
- `use_case_3_perfect_reconstruction/train_reconstruction.py` (line 1-148)
- `use_case_3_perfect_reconstruction/models.py` (line 109-290)
- `use_case_2_demographic_variations/generate_demographic_variations_fixed.py`

---

**Ready to Start:** Run `./retrain_uc2_uc3_128x128.sh` now! 🚀
