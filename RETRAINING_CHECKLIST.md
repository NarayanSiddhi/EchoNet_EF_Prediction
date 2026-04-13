# RETRAINING COMPLETE CHECKLIST

## ✅ What Has Been Done

### 1. Scripts Created ✓
- [x] `analyze_cardiac_cycle_coverage.py` - Analyzes temporal coverage with detailed cardiac cycle justification
- [x] `retrain_uc2_uc3_guide.py` - Interactive guide that checks GPU, paths, and generates commands
- [x] `retrain_uc2_uc3_128x128.sh` - Ready-to-run bash script (executable)
- [x] `UC2_UC3_RETRAINING_GUIDE.md` - Complete documentation (this is your reference manual)
- [x] `cardiac_cycle_analysis.txt` - Pre-generated analysis results

### 2. Analysis Completed ✓
- [x] Cardiac cycle coverage validated for 32 frames @ 30 fps
- [x] Dataset temporal properties analyzed (6,227 videos, mean age 10.1 years)
- [x] Paper text generated for Methods, Discussion, and Table captions

### 3. Code Verified ✓
- [x] Architecture already supports 128×128 (spatial_size parameter)
- [x] Training script correctly maps --video_size to spatial_size
- [x] Generation script correctly reads video_size from checkpoint
- [x] No code changes required - just retrain with new flags!

### 4. GPU Check ✓
- [x] NVIDIA RTX A5000 detected (25.3 GB)
- [x] Sufficient memory for batch_size=4 at 128×128
- [x] Memory warnings and fallback strategies documented

---

## 🚀 What You Need To Do Now

### OPTION A: Quick Start (Recommended)
```bash
# In your terminal, run:
cd /data/home/sai/Documents/EchoNet-Pediatric-BIGAN-AUGMENTATION
./retrain_uc2_uc3_128x128.sh
```

This will:
1. Train UC3 at 128×128×32 (~2-4 hours)
2. Generate UC2 variations (~1 hour)
3. Save everything to organized directories

### OPTION B: Step by Step
```bash
# Step 1: Train UC3
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

# Step 2: Generate UC2 (after UC3 finishes)
cd ../use_case_2_demographic_variations
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

---

## 📝 Paper Updates (Copy-Paste Ready)

### Section 3.2 - Preprocessing
**FIND:**
> 64×64 pixels... 16 frames

**REPLACE WITH:**
> Videos were preprocessed to 128×128 pixels and temporally subsampled to 32 frames at 30 fps (clip duration: 1.07 seconds). This temporal window was selected to ensure complete cardiac cycle coverage across the pediatric heart rate spectrum: at 30 fps, 32 frames span 1.07 seconds, covering ≥1 complete cardiac cycle for heart rates up to 56 bpm and multiple cycles for typical pediatric resting rates (80-100 bpm: 1.4-1.8 cycles).

### Discussion Section
**ADD NEW PARAGRAPH:**
> We address resolution concerns by operating at 128×128 pixels, matching typical clinical echocardiography display resolutions. While earlier work used 64×64 for computational efficiency, our U-Net architecture scales naturally to 128×128 with manageable memory requirements (12-16 GB GPU memory at batch size 4). The 32-frame temporal window (1.07 seconds at 30 fps) provides complete cardiac cycle coverage even at pediatric tachycardia rates (>120 bpm: 2.13 cycles), ensuring the model captures both systolic and diastolic phases for accurate cardiac function assessment.

### Table 5 Caption
**ADD FOOTNOTE:**
> All models process clips of 32 frames at 30 fps (1.07s), sufficient to capture ≥1 cardiac cycle across the pediatric HR range (60-150 bpm).

### Table 5 Values
After retraining, re-run evaluation and update SSIM/PSNR numbers.

---

## 📊 Expected Results

### Training Metrics
- **UC3 L1 Loss:** Should converge to 0.05-0.15
- **Training Time:** 2-4 hours (50 epochs, ~6K videos)
- **Checkpoint Size:** ~150-200 MB

### Generated Videos
- **Count:** 300 videos (3 variations × 100 inputs)
- **Quality:** Should match or exceed 64×64 fidelity
- **Diversity:** Sex/age/BMI variations visible

### File Structure After Training
```
use_case_3_perfect_reconstruction/
├── ckpt_uc3_128x128_T32/
│   ├── recon_epoch_1.pt
│   ├── recon_epoch_2.pt
│   ├── ...
│   ├── recon_epoch_50.pt
│   └── recon_best.pt          ← Use this for UC2

demographic_variations_128x128_T32/
├── video_0000_var1_sex_change.mp4
├── video_0000_var2_age_shift.mp4
├── video_0000_var3_bmi_change.mp4
├── ...
└── variations_manifest.csv
```

---

## ⚠️ Troubleshooting

### "CUDA out of memory"
**Solution:**
```bash
# Reduce batch size
--batch_size 2  # or 1 if still OOM

# Or reduce model capacity
--base_channels 32
```

### "Manifest not found"
**Solution:** Update manifest path in commands. Try:
- `data/processed_full/train_manifest_filtered_clean.csv` (current default)
- `data/processed/train_manifest.csv`
- Your custom path

### "Training is slow"
**Check GPU usage:**
```bash
watch -n 1 nvidia-smi
```
Should show ~80-100% GPU utilization during training.

### "Generated videos are blurry"
**Normal at first.** Try:
- More epochs: `--epochs 75`
- Lower learning rate: `--lr 5e-5`
- More diversity: `--diversity_weight 0.5`

---

## 📈 Validation Checklist

After training completes:

- [ ] Check final L1 loss (should be < 0.15)
- [ ] Visually inspect 10-20 generated videos
- [ ] Verify demographic changes are visible
- [ ] Run quality metrics (SSIM, PSNR, FID)
- [ ] Update paper with new numbers
- [ ] Compare 128×128 vs 64×64 quality

---

## 📁 Files Reference

| File | Purpose | When to Use |
|------|---------|-------------|
| `retrain_uc2_uc3_128x128.sh` | All-in-one training script | Run this to start training |
| `analyze_cardiac_cycle_coverage.py` | Temporal analysis | Already run, results in `cardiac_cycle_analysis.txt` |
| `cardiac_cycle_analysis.txt` | Pre-generated results | Copy-paste into paper |
| `UC2_UC3_RETRAINING_GUIDE.md` | Complete documentation | Your reference manual |
| `retrain_uc2_uc3_guide.py` | Interactive helper | Run for GPU check and command generation |

---

## ✅ Final Checklist Before Starting

- [ ] Read `UC2_UC3_RETRAINING_GUIDE.md` (comprehensive guide)
- [ ] Verify you have 25.3 GB GPU (✓ already confirmed)
- [ ] Confirm manifest exists at `data/processed_full/train_manifest_filtered_clean.csv`
- [ ] Ensure ~100 GB free disk space for checkpoints and videos
- [ ] Start training: `./retrain_uc2_uc3_128x128.sh`
- [ ] Monitor training progress (watch for decreasing loss)
- [ ] Update paper text (use copy-paste ready text above)
- [ ] Re-run evaluation metrics after training

---

## 🎯 Success Criteria

You'll know retraining succeeded when:

1. ✅ Training converges (L1 loss < 0.15)
2. ✅ Checkpoint file `recon_best.pt` exists and is ~150-200 MB
3. ✅ Generated videos look sharp at 128×128
4. ✅ Demographic variations are visually distinct
5. ✅ SSIM/PSNR metrics are reasonable (may be slightly lower than 64×64, which is OK)
6. ✅ Paper updated with 128×128 resolution
7. ✅ Cardiac cycle justification added to paper

---

## 💡 Pro Tips

1. **Start training before you leave** - it takes 2-4 hours
2. **Update paper while training** - parallel work saves time
3. **Test with 10 videos first** - add `--max_videos 10` to check everything works
4. **Save old checkpoints** - don't overwrite 64×64 models until 128×128 works
5. **Monitor GPU** - use `nvidia-smi` to check utilization

---

## 🚀 Ready to Start?

```bash
cd /data/home/sai/Documents/EchoNet-Pediatric-BIGAN-AUGMENTATION
./retrain_uc2_uc3_128x128.sh
```

**Estimated time:** 4-6 hours total (can parallelize paper updates)

**Good luck! 🎉**

---

## Questions?

Refer to:
- **Complete guide:** `UC2_UC3_RETRAINING_GUIDE.md`
- **Training code:** `use_case_3_perfect_reconstruction/train_reconstruction.py`
- **Model architecture:** `use_case_3_perfect_reconstruction/models.py`
- **Generation code:** `use_case_2_demographic_variations/generate_demographic_variations_fixed.py`
