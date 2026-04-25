# Bundle for LaTeX (128×128 metrics + manifest counts + training facts)

Copy sections below into email or Overleaf notes.

---

## 1. `uc2_128x128_metrics.json` (full file)

```json
{
  "n": 18681,
  "errors": 0,
  "SSIM": 0.07830590410706234,
  "SSIM_std": 0.05853327396999087,
  "PSNR": 15.136410546242233,
  "PSNR_std": 1.828432192183666,
  "MSE": 0.03252391820346932,
  "MSE_std": 0.008740063331940473,
  "by_type": {
    "age_variation": {
      "SSIM": 0.07170073958779762,
      "PSNR": 15.123612703322591,
      "MSE": 0.03164420070707177
    },
    "sex_variation": {
      "SSIM": 0.07204842513280388,
      "PSNR": 15.147100572577076,
      "MSE": 0.031401652551114184
    },
    "bmi_variation": {
      "SSIM": 0.09116854760058549,
      "PSNR": 15.138518362827025,
      "MSE": 0.034525901352221985
    }
  }
}
```

**Path:** `uc2_128x128_metrics.json` (repo root)

---

## 2. `uc3_128x128_metrics.json` (full file)

```json
{
  "n": 6227,
  "errors": 0,
  "SSIM": 0.9906335041185971,
  "SSIM_std": 0.0035174314109471107,
  "PSNR": 45.44507664515271,
  "PSNR_std": 1.1429130357555046,
  "MSE": 3.0094006745331836e-05,
  "MSE_std": 7.487142507103484e-06
}
```

**Path:** `uc3_128x128_metrics.json` (repo root)

---

## 3. `uc2_128x128_fid.json` (full file)

```json
{
  "num_real_videos": 500,
  "num_synthetic_videos": 500,
  "fid": 142.0064681751934,
  "fvd": null,
  "device": "cuda",
  "frames_per_chunk": 16
}
```

**Path:** `uc2_128x128_fid.json` (repo root)

---

## 4. Manifest row counts (verified with `pandas.read_csv`, Feb 2026 workspace)

| Label | Row count | Manifest path |
|-------|------------|---------------|
| **UC2 variations (128×128, matches metrics `n`)** | **18,681** | `use_case_2_demographic_variations/demographic_variations_128x128/variations_manifest.csv` |
| **UC3 perfect copies (128×128, matches metrics `n`)** | **6,227** | `perfect_synthetic_copies_128x128/perfect_copies_manifest.csv` |
| **UC3 perfect copies (64 pipeline / paper-style N)** | **7,791** | `perfect_synthetic_copies/perfect_copies_manifest.csv` |
| UC1 paired synthetic manifest (large run on disk) | 70,119 | `use_case_1_balance_dataset/synthetic_paired_dataset/synthetic_manifest.csv` |
| Root `synthetic_manifest.csv` | 23,373 | `synthetic_manifest.csv` (verify column semantics before calling “UC1”) |

**Paper / planning figure for UC1 additive balancing:** `ALL_EVALUATION_RESULTS_VERIFIED.json` still documents **5,000** synthetic videos to reach per-group targets (total corpus 12,791). That count is **not** the row count of `synthetic_paired_dataset/synthetic_manifest.csv` above—use **5,000** only if you cite the imbalance-analysis pipeline; otherwise cite **70,119** for that CSV.

---

## 5. Loss weights / epochs (what is actually in code—verify against checkpoint README)

### UC1 — Conditional C3D GAN (`use_case_1_balance_dataset/c3dgan/config.yaml`)

| Setting | Value |
|---------|--------|
| `training.n_epochs` | **200** |
| `training.batch_size` | **20** |
| `training.lr_g`, `training.lr_d` | **0.0002** each |
| `training.beta1` | **0.5** |
| `training.n_critic` | **5** |
| `training.lambda_gp` | **10.0** (WGAN-GP) |
| `model.video_length` | **96** |
| `model.video_size` | **128** |
| `augmentation.target_samples_per_group` | **500** |

### UC3 reconstruction trainer (`use_case_3_perfect_reconstruction/train_reconstruction.py`)

This script optimizes **L1(output, input) + λ_temp × L1(Δoutput, Δinput)** only (no SSIM / GAN / demo classifier terms in this file).

| Setting | Default |
|---------|---------|
| `--epochs` | **50** |
| `--batch_size` | **4** |
| `--lr` | **1e-4** |
| `--lambda_temp` | **0.1** |
| `--video_length` | **32** |
| `--video_size` | **64** or **128** (must match checkpoint) |
| `--conditioning` | **`film`** (default) or **`concat`** |
| Optimizer | **Adam** |
| AMP | **on** by default (`--amp`) |

**Checkpoint used in 128×128 augmentation shell:** `use_case_3_perfect_reconstruction/ckpt_uc3_128x128_T32/recon_best.pt` (see `run_augmentation_128x128.sh`).

If the LaTeX must keep the **five-term** loss \((\lambda_{\text{pix}}, \ldots)\) from the narrative draft, add a sentence that those weights describe the **full GAN / pix2pix-style objective** used in a separate training path or earlier design, and point to `ALL_EVALUATION_RESULTS_VERIFIED.json` → `training_hyperparameters.use_cases_2_3_conditional_3d_unet_gan.training.loss_weights` **(100, 5, 10, 1, 5)**—or align text strictly with `train_reconstruction.py` to avoid mismatch.

---

*End bundle.*
