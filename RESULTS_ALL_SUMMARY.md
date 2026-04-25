# All project results (consolidated)

One place for **numbers**, **where they live**, and **when two files disagree**.  
Quantities are as stored in the repo at generation time—re-run scripts to refresh.

---

## 1. Generative augmentation (Use Cases 1–3)

### 1.1 Dataset scale (paper-aligned snapshot)

| Item | Original | Synthetic | Total | Ratio | Primary source |
|------|----------|-----------|-------|-------|------------------|
| Baseline | 7,791 | 0 | 7,791 | 1× | `ALL_EVALUATION_RESULTS_VERIFIED.json` |
| UC1 balancing | 7,791 | 5,000 | 12,791 | ~1.6× | `table_1_dataset_statistics` + `imbalance_analysis` (see JSON) |
| UC2 variations | 7,791 | 23,373 | 31,164 | 4× | `ALL_EVALUATION_RESULTS_VERIFIED.json` |
| UC3 recon pairs | 7,791 | 7,791 | 15,582 | 2× (orig + synth rows) | same |

**Full balancing breakdown (UC1):** `ALL_EVALUATION_RESULTS_VERIFIED.json` → `use_case_1_balancing_details.groups` (per-class counts and “needed/generated”).

---

### 1.2 UC1 — FID (distribution quality)

| Description | num real | num synth | FID ↓ | File |
|-------------|-----------|-------------|-------|------|
| Legacy / paper-style eval | 200 | 200 | **75.085** | `uc1_quality_metrics.json` → `legacy_200_vs_200` |
| Smaller repaired run (Apr 2026) | 20 | 6 | **124.38** | `uc1_quality_metrics.json` → `repaired_evaluator_2026_04` |

**Note:** FID depends on manifest and sample count; cite the row you actually use in the paper.

---

### 1.3 UC2 — Pixel metrics (two different cohorts / resolutions)

**A) 128×128 pipeline** (`demographic_variations_128x128/variations_manifest.csv` style)

| Metric | Value | *n* | File |
|--------|-------|-----|------|
| SSIM | 0.0783 ± 0.0585 | 18,681 | `uc2_128x128_metrics.json`, `recalc_metrics_all_uc.json` |
| PSNR | 15.14 ± 1.83 dB | 18,681 | same |
| MSE | 0.0325 ± 0.00874 | 18,681 | same |

**By variation type (128×128, `uc2_128x128_metrics.json`):**

| Type | SSIM | PSNR (dB) | MSE |
|------|------|-----------|-----|
| age_variation | 0.0717 | 15.12 | 0.0316 |
| sex_variation | 0.0720 | 15.15 | 0.0314 |
| bmi_variation | 0.0912 | 15.14 | 0.0345 |

**B) UC2 FID (128×128 paired subset)**

| num real | num synth | FID | File |
|----------|-----------|-----|------|
| 500 | 500 | **142.01** | `uc2_128x128_fid.json` |

---

### 1.4 UC3 — Reconstruction / perfect copies

**A) Paper-style tensor metrics (N = 7,791)** — `ALL_EVALUATION_RESULTS_VERIFIED.json` → `table_2_reconstruction_fidelity.original_manifest_values.metrics`

| Metric | Mean ± Std | Notes |
|--------|------------|--------|
| SSIM | 0.99469 ± 0.00305 | 93.94% > 0.99 |
| PSNR | 48.98 ± 0.62 dB | mean excludes 8 infinite; 93.97% > 48 dB |
| MSE | 0.8310 ± 0.1243 | 91.62% < 1.0 (table uses raw scale as in JSON) |
| Pixel-perfect | 8 videos | MSE ≈ 0, PSNR → ∞ |

**B) Recalculated from compressed MP4 files (same manifest philosophy, lower numbers)** — same JSON → `recalculated_from_video_files.metrics`

| Metric | Mean ± Std |
|--------|------------|
| SSIM | 0.718 ± 0.109 |
| PSNR | 24.49 ± 5.90 dB |
| MSE | 0.00850 ± 0.00519 |

**C) 128×128 subset on disk** — `perfect_synthetic_copies_128x128/perfect_copies_manifest.csv` (evaluated *n* below)

| Metric | Mean ± Std | *n* | File |
|--------|------------|-----|------|
| SSIM | 0.9906 ± 0.00352 | **6,227** | `uc3_128x128_metrics.json`, `recalc_metrics_all_uc.json` |
| PSNR | 45.45 ± 1.14 dB | 6,227 | same |
| MSE | 3.01×10⁻⁵ ± 7.49×10⁻⁶ | 6,227 | same |

**D) UC1 in `recalc_metrics_all_uc.json`**

| *n* | errors | Note |
|-----|--------|------|
| 0 | 3000 | SSIM/PSNR/MSE NaN — paths/manifest mismatch for UC1 paired video recalc; **do not** use this block for UC1 quality. |

---

### 1.5 UC2 — Demographic redistribution (% of corpus)

From `ALL_EVALUATION_RESULTS_VERIFIED.json` → `table_3_demographic_distribution` (original total 7,791 vs augmented total 31,164).

**Age (%):** 0–1: 8.7 → 6.5 (−2.2) · 1–2: 3.8 → 2.8 (−1.0) · 2–3: 4.4 → 10.1 (+5.7) · 3–5: 7.4 → 5.6 (−1.8) · 5–8: 14.2 → 10.6 (−3.6) · 8–12: 20.4 → 15.3 (−5.1) · 12–15: 21.3 → 21.1 (−0.2) · 15–18: 19.8 → 27.9 (+8.1)

**Sex (%):** Male 57.3 → 53.7 (−3.6) · Female 42.6 → 46.3 (+3.7) · Other 0.1 → 0.0 (−0.1)

**BMI (%):** Underweight 49.3 → 12.2 (−37.1) · Normal 34.6 → 58.7 (+24.1) · Overweight 10.0 → 27.5 (+17.5) · Obese 6.1 → 1.5 (−4.6)

---

### 1.6 Grad-CAM style validation (synthetic vs real attention)

From `ALL_EVALUATION_RESULTS_VERIFIED.json` → `gradcam_validation_results` (claims *n* = 150; sources named in that JSON).

| Metric | Mean ± Std | Above threshold |
|--------|------------|------------------|
| Cosine similarity | 0.8781 ± 0.1139 | 86.7% (> 0.75) |
| Spatial correlation | 0.9204 ± 0.0933 | 95.3% (> 0.70) |

**By variation type (cosine):** age 0.8786 ± 0.1100 · sex 0.8783 ± 0.1195 · BMI 0.8773 ± 0.1145

---

## 2. Downstream EF prediction (`ef_prediction/`)

### 2.1 Overall validation metrics

| Model | MAE | MSE | RMSE | R² | Source file |
|-------|-----|-----|------|-----|----------------|
| Real | 4.520 | 38.236 | 6.184 | 0.586 | `eval_results/real_metrics.json` |
| Fused (run 1, tag `20260419_173306`) | 4.397 | 37.084 | 6.090 | 0.598 | `multi_run_results/fused_run_1_20260419_173306_metrics.json` |
| Fused (5-run aggregate, run_1 line) | 4.384 | 35.966 | 5.997 | 0.610 | `multi_run_results/fused_5run_metrics.json` |

### 2.2 Grouped summary (CSV)

`ef_prediction/group_results/final_results_summary.csv` — key rows:

| Group | MAE | MSE | RMSE | R² | Count |
|-------|-----|-----|------|-----|-------|
| OVERALL_REAL | 4.520 | 38.236 | 6.183 | 0.586 | ALL |
| OVERALL_FUSED | 4.384 | 35.966 | 5.997 | 0.610 | ALL |
| age_0-1 | 5.039 | 49.890 | 7.063 | 0.823 | 960 |
| age_1-2 | 3.943 | 24.366 | 4.936 | 0.860 | 438 |
| … | … | … | … | … | … |

*(Full subgroup table: open the CSV.)*

### 2.3 Other eval JSONs (paths only)

- `ef_prediction/eval_results/fused_metrics.json`
- `ef_prediction/eval_results_train_data/real_vs_fused_metrics.json`
- `ef_prediction/eval_results_val_data/real_vs_fused_metrics.json`
- `ef_prediction/eval_results_temporal/real_vs_concat_metrics.json`
- `ef_prediction/group_results/all_group_metrics.json`
- `ef_prediction/group_results/demographic_ablation_metrics.json`
- `ef_prediction/group_results/demographic_ablation_overall_metrics.json`

Open these for ablation / temporal / split-specific numbers.

---

## 3. Misc / sanity

| File | Content |
|------|---------|
| `fid_sanity_latest.json` | FID 0.0 on 12 vs 12 — **sanity / degenerate test**, not a publication result |
| `demographic_classifier/results/distribution_metrics.json` | Classifier distribution metrics |

---

## 4. How to resolve “which number do I cite?”

| Claim | Recommended source |
|-------|---------------------|
| UC1 FID for paper (200/200) | `uc1_quality_metrics.json` → `legacy_200_vs_200` |
| UC2 quality at 128×128 | `uc2_128x128_metrics.json` + `uc2_128x128_fid.json` |
| UC3 high SSIM / PSNR “tensor” story | `ALL_EVALUATION_RESULTS_VERIFIED.json` → `original_manifest_values` |
| UC3 “after MP4 compression” | same file → `recalculated_from_video_files` |
| UC3 128×128 checkpoint run | `uc3_128x128_metrics.json` (*n* = 6227) |
| EF real vs fused | `real_metrics.json` + chosen fused `*_metrics.json` or `final_results_summary.csv` |

---

*Generated as a repo index. Re-run `recalc_all_metrics.py`, `run_augmentation_128x128.sh`, and `ef_prediction/evaluate_*.py` to refresh numbers.*
