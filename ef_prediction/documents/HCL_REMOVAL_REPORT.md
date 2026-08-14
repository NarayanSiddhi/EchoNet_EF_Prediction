# HCL Ablation — Short Summary

**Validation set:** 1,558 clips | MAE / RMSE in EF % | MSE in (EF %)²

---

## Previous results (with HCL)

These were our original trained models using hierarchical contrastive loss during training.

| Model | HCL | Fusion | Data | MAE | MSE | R² |
|-------|-----|--------|------|-----|-----|-----|
| Real | Yes | No | Real video | 4.52 | 38.24 | 0.586 |
| Fused | Yes | Yes (gated) | Real + synthetic | 4.40 | 37.08 | 0.598 |

Checkpoints: `checkpoints/real/best.pth`, `checkpoints/fused/run_1_best.pth`

---

## Ablation results (no HCL)

Same architecture and data, but trained with `--hcl-weight 0`.

| Model | HCL | Fusion | Data | MAE | MSE | R² |
|-------|-----|--------|------|-----|-----|-----|
| Real | No | No | Real video | 4.45 | 36.57 | 0.604 |
| Fused | No | Yes (gated) | Real + synthetic | 4.34 | 35.32 | 0.617 |
| Synthetic-only | No | No | Synthetic video only | 4.80 | 45.14 | 0.516 |

Checkpoints: `checkpoints/real_no_hcl/best.pth`, `checkpoints/fused/run_99_best.pth`, `checkpoints/real_synth_no_fusion_no_hcl/best.pth`

---

## Fused: no HCL vs original (HCL)

| | MAE | MSE | R² |
|---|-----|-----|-----|
| Fused with HCL (original) | 4.40 | 37.08 | 0.598 |
| Fused without HCL | 4.34 | 35.32 | 0.617 |
| **Improvement** | **−0.06** | **−1.76** | **+0.019** |

---

## Why fused no-HCL beats the original fused model

HCL is an extra training loss on the **video-only** embedding (`z`). It tries to organize representations by EF level and demographics, but the actual EF prediction uses **`concat(video features, demographic vector)`** — a different path.

With HCL enabled, the model splits its effort between EF regression and contrastive embedding geometry. That conflict hurts validation MSE. Turning HCL off (`--hcl-weight 0`) lets training focus only on predicting EF, while **demographics are still fed in** through the normal late-fusion path.

Removing HCL does **not** remove demographic inputs — only the auxiliary contrastive objective.

Synthetic-only performs worst, confirming that real video (alone or fused with synthetic) is needed for good EF prediction.

---

*Source: `ef_prediction/ablation_results/ablation_summary.csv`*
