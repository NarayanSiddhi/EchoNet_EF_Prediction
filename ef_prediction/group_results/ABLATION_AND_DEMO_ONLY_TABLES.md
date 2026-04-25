# Ablation and demo-only control — result tables

**Validation cohort:** *n* = **1558** for all rows below (same manifests as overall eval).

**Units:** MAE / RMSE in **EF percentage points**; MSE in **(percentage points)²**; R² dimensionless.

---

## 1. Overall demographic ablation (normal / zero / shuffle)

`demo_vec` is **correct**, **all zeros**, or **shuffled within each mini-batch** before `demo_encoder` → concat → EF head. Video inputs unchanged.

### Fused model (`perfect_copies_val.csv`, checkpoint `fused/run_1_best.pth`)

| Mode | MAE | MSE | RMSE | R² | Δ MAE vs normal | Δ R² vs normal |
|------|-----|-----|------|-----|-----------------|----------------|
| normal | 4.397 | 37.084 | 6.090 | 0.598 | — | — |
| zero | 4.454 | 38.572 | 6.211 | 0.582 | +0.057 | −0.016 |
| shuffle | 4.407 | 37.373 | 6.113 | 0.595 | +0.010 | −0.003 |

*Source:* `demographic_ablation_overall_fused_metrics.json`

### Real model (`val_manifest.csv`, checkpoint `real/best.pth`)

| Mode | MAE | MSE | RMSE | R² | Δ MAE vs normal | Δ R² vs normal |
|------|-----|-----|------|-----|-----------------|----------------|
| normal | 4.520 | 38.236 | 6.184 | 0.586 | — | — |
| zero | 4.655 | 39.757 | 6.305 | 0.569 | +0.135 | −0.017 |
| shuffle | 4.540 | 38.578 | 6.211 | 0.582 | +0.020 | −0.004 |

*Source:* `demographic_ablation_overall_metrics.json` (real)

---

## 2. “Demographics-only through the trained head” (control)

**Procedure:** same checkpoints and val manifests as above, but **`pooled` (video summary) is replaced with zeros** before `concat(pooled, demo_emb)` and the existing EF **`head`**. `demo_emb` from the true `demo_vec` is unchanged.

This is **not** a separately trained demographics-only model; it stress-tests **how the already-trained head behaves when the video block of its input carries no signal**.

| Model | MAE | MSE | RMSE | R² | *n* |
|-------|-----|-----|------|-----|-----|
| Fused | 10.688 | 148.447 | 12.184 | −0.609 | 1558 |
| Real | 7.613 | 102.550 | 10.127 | −0.111 | 1558 |

*Source:* `ef_prediction/eval_results/demo_only_control_fused_metrics.json`, `demo_only_control_real_metrics.json`

---

## 3. Quick comparison to full normal eval (same *n*)

| Model | Full model MAE (normal ablation) | Demo-only control MAE |
|-------|----------------------------------|-------------------------|
| Fused | 4.397 | 10.688 |
| Real | 4.520 | 7.613 |

Removing usable video information at the head input **severely degrades** predictions, as expected.

---

*Generated from on-disk JSON metrics in this repo; re-run `ablate_demographic_effect_overall.py` and `evaluate_demo_only_control.py` after new checkpoints to refresh.*
