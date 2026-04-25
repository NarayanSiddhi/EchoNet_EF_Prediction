# EF prediction model: implementation summary, results, and conclusions

This document summarizes the **pediatric echo ejection-fraction (EF) regression** stack under `ef_prediction/`: what the models do, how demographics are used, what numbers we obtained, and what the **embedding and hypersphere analyses** imply. It is written to support a methods / results subsection or internal technical memo.

**Structured exports (regenerate after edits):** Word **`.docx`**, print-styled **`.html`**, and **`.pdf`** are built under `ef_prediction/documents/` by running `bash ef_prediction/export_report.sh` (requires **pandoc** and Python package **xhtml2pdf**).

---

## 1. Scope and data

- **Task:** Predict **EF** (as a scalar; internally often scaled to `[0, 1]` with evaluation in percentage units where applicable).
- **Real-only path:** Single processed **original** clip per row (`val_manifest`, train manifest under `config.yaml`).
- **Fused path:** **Real + synthetic** clip pair per row (`train_manifest_fused`, `val_manifest_fused` pointing at `perfect_synthetic_copies/`), same spatial and temporal settings as in `config.yaml` (`video_length`, `video_size`, `original_video_dir`, `synthetic_video_dir`).

**Demographics:** Each row supplies an **11-D** one-hot-style vector (sex × age bin × BMI category), built in `demographics_utils.py` and exposed by `dataset.py` / `dataset_demographics.py` as `demo_vec` plus discrete indices for auxiliary losses.

---

## 2. Model architectures

### 2.1 `PTEFNetReal` (`models/pt_efnet_real.py`)

- **Backbone:** ImageNet-pretrained **ResNet** (default **ResNet34** per `config.yaml`) over grayscale frames expanded to 3 channels.
- **Temporal modeling:** **Bidirectional LSTM** + **temporal attention** → pooled video descriptor.
- **Late fusion:** `demo_encoder`: `Linear(11 → 64) → ReLU → Linear(64 → demo_dim)` with default `demo_dim = 32` (real path uses constructor defaults unless overridden).
- **Head:** MLP on **`concat(pooled_video, demo_emb)`** → sigmoid scalar EF.
- **Auxiliary representation:** `projection_head(pooled)` returns **`z`** (video-only, 64-D after the two-layer projection) used where contrastive / hierarchical objectives apply during training.

### 2.2 `PTEFNetFused` (`models/pt_efnet_fused.py`)

- **Two streams:** Same CNN trunk on **real** and **synthetic** clips; per-frame features are combined by either:
  - **`concat`:** `[real || syn]` → linear down to `frame_dim`, or  
  - **`gated`:** learned **gate** `g ∈ (0,1)^128` so fused frame features are **`g * real + (1-g) * syn`** (current `config.yaml`: **`fused_fusion: gated`**).
- **Temporal stack:** LSTM hidden size **`fused_lstm_hidden: 192`** (→ pooled dim **384**), attention, then the same pattern as real: **`demo_encoder`**, **`concat(pooled, demo_emb)`** into the EF head, and **`projection_head(pooled)`** for **`z`**.
- **Regularization:** `fused_dropout` on the head (e.g. **0.25** in config).

### 2.3 Forward API for analysis (`return_embedding`)

Both fused and real `forward(..., return_embedding=...)` support:

| Value            | Tensor returned | Role |
|------------------|-----------------|------|
| **`contrastive`** (default) | `z = projection_head(pooled)` | **Video-only** embedding; used for **hierarchical contrastive loss (HCL)** on fused/real training. |
| **`head_input`** | `h = concat(pooled, demo_emb)` | **Exact vector fed to the EF MLP** (demographics included). Use this for **t-SNE / UMAP / hypersphere** when asking how the **final regressor** sees samples. |

---

## 3. Training and objectives (high level)

Configured in `ef_prediction/config.yaml` and `train_real.py` / `train_fused.py`:

- **Primary loss:** Hybrid **SmoothL1 + MSE** on EF (`hybrid_mse_weight`), with **cosine LR decay**, **gradient clipping**, **AMP** optional, DataLoader **`num_workers`**, checkpointing on **best validation MSE** (typical setup).
- **Fused auxiliary:** **`fused_hcl_weight`** (often lower than real `hcl_weight`) so EF regression is not overpowered by contrastive terms.
- **Hierarchical contrastive (FairHICON-inspired):** **Level 1** pulls same **EF-derived task label** (e.g. high vs low EF vs threshold); **Level 2** adds **subgroup-aware** structure (sex / age / BMI) with scheduled **`hcl_h2_*`** warm-up, temperatures **`hcl_temp_h1/h2`**, and prototype-style weighting **`hcl_alpha_intra` / `hcl_alpha_inter`**.  
  **Important:** HCL is applied to **`z` (video-only projection)**, not to the concatenated `head_input`. So contrastive geometry is explicitly **not** the same subspace as the scalar that concatenates **`demo_emb`**.

---

## 4. Quantitative results (validation-style aggregates on file)

Numbers below come from saved artifacts in the repo at the time of this report; **re-run** `evaluate_ef_real.py` / `evaluate_ef_fused.py` after any retrain to refresh.

### 4.1 Overall metrics

| Model  | Source file | MAE | MSE | RMSE | R² |
|--------|-------------|-----|-----|------|-----|
| **Real** | `ef_prediction/eval_results/real_metrics.json` | **4.52** | **38.24** | **6.18** | **0.586** |
| **Fused** (run tag `20260419_173306`) | `ef_prediction/multi_run_results/fused_run_1_20260419_173306_metrics.json` | **4.40** | **37.08** | **6.09** | **0.598** |

**`group_results/final_results_summary.csv`** (if used as a dashboard) reports similar overall lines, e.g. **OVERALL_REAL** vs **OVERALL_FUSED** in the same ballpark, with fused **slightly** better on MAE/MSE/R² in that table.

### 4.2 Subgroup snapshot (fused MAE from `final_results_summary.csv`)

Illustrative rows (MAE in EF % units; **lower is better**):

- **Sex:** male **~3.97**, female **~4.24** (both large *n* in that export).
- **Age bins:** best **~3.48** (e.g. `age_3-5`) vs hardest **~5.04** (`age_0-1`, smaller *n*).
- **BMI bins:** spread from **~3.87** (e.g. obese) to **~4.44** (overweight) depending on bin and count.

These tables answer **“does error differ by subgroup?”** They do **not** by themselves prove **causal** demographic effects; they motivate **ablation** and **fairness reporting**.

---

## 5. Analyses performed (demographics, embeddings, sphere)

### 5.1 Why early t-SNE/UMAP looked “mixed” by sex / age / BMI

Initial scripts plotted **`z` = `projection_head(pooled)`**, which is **video-only**. **Demographics never enter `z`**; they enter only via **`demo_emb`** in **`head_input`**. Coloring **`z`** by sex/age/BMI therefore **cannot** show clean demographic blobs **by construction**, even if the head uses demographics strongly.

**Fix applied:** `visualize_embeddings.py` and `visualize_subgroups.py` gained **`--embedding head_input`** (default) so 2D reductions use the **same vector the EF MLP sees**; **`--embedding contrastive`** reproduces the old behavior for comparison. Output filenames include the suffix `_head_input` / `_contrastive`.

### 5.2 Ablation of demographic conditioning

**`ablate_demographic_effect.py`** evaluates **normal** vs **zero** vs **shuffle** on `demo_vec` over grouped manifests:

- **`--model fused`** — fused checkpoint.  
- **`--model real`** — real checkpoint (same protocol on video + demographics).

**Interpretation:** Large **shuffle/zero vs normal** deltas ⇒ the prediction **depends** on supplied demographics; tiny deltas ⇒ the head is **mostly video-driven** for those groups.

### 5.3 Per-group fused metrics

**`evaluate_fused_demographics.py`** (and summarized CSVs under `group_results/`) give **MAE / MSE / R²** per sex, age, BMI bin — aligned with **fairness / equity reporting**, analogous in spirit to gap-focused discussion in fairness-aware contrastive papers (though our backbone is not FairHICON’s multi-encoder genomics design).

### 5.4 3D unit-sphere visualization (FairHICON *style*, not same training)

**`visualize_hypersphere.py`** implements a standard display pipeline:

1. L2-normalize each **`head_input`** (or **`contrastive`**) row.  
2. **PCA → 3** components.  
3. L2-normalize in **ℝ³** so points lie on **S²** with a **wireframe sphere**.  
4. Color by **sex / age / BMI / EF** (`--color-by`, `--ef-mode`).

**Typical outcome on your runs:**

- **Color = EF:** visible **angular** structure (EF varies smoothly or in patches on the sphere) → the **dominant linear variance** in `head_input` aligns with **the supervised target**, as expected for a well-trained regressor.  
- **Color = sex:** **heavy mixing** → **sex is not the leading PCA axis** of `head_input`; demographics may still matter in **higher** PCs or only through **`demo_emb`**’s subvector.

**Explained variance** in the plot title (~**73%** in your EF-colored run) means the **first three PCs** capture most **linear** variance of the **normalized** features; it does not prove optimality or clinical validity.

### 5.5 Relation to *Contrastive Semantic Learning.pdf* (FairHICON)

That preprint targets **transcriptomics** with **separate sex-common and sex-specific encoders** and a **two-level hierarchical contrastive** objective so that **latent “territories”** (common / male / female) are **explicit training targets**.  

Our echo pipeline: **one** visual stream + **late demographic fusion** + **HCL on video-only `z`**. So:

- **Fair comparison:** subgroup metrics, ablations, embedding / sphere **geometry**.  
- **Unfair expectation:** identical **purple / blue / red** manifolds without **matching architecture and losses**.

---

## 6. Conclusions

1. **Modeling:** Pediatric EF is predicted from **video** (ResNet + BiLSTM + attention), with **11-D demographics** encoded and **concatenated** before the final MLP. The **fused** model adds a **gated** blend of real vs synthetic per-frame features, with training stabilized by **dropout**, **hybrid losses**, and **scheduled hierarchical contrastive** auxiliary loss on **`z`**.

2. **Results:** On saved checkpoints, **fused** validation metrics are **modestly better** than **real** (e.g. MAE **~4.40 vs ~4.52**, R² **~0.60 vs ~0.59** on the cited JSON files). **Subgroup tables** show **non-uniform MAE** across age and sex bins — useful for reporting **who** the model serves best.

3. **Demographics and geometry:**  
   - **t-SNE/UMAP of `contrastive` `z`** should **not** be read as “demographics don’t matter” — **`z` omits `demo_emb`**.  
   - **`head_input`** plots and the **hypersphere** view show **EF-aligned** structure and **little large-scale sex separation** in the top 3 PCs — consistent with **video-first** representations and **auxiliary** demographic conditioning.  
   - **Decisive tests** for “do demographics change the scalar EF?” remain **`ablate_demographic_effect.py`** and subgroup **error tables**, not embedding color alone.

4. **Reporting recommendation:** For a paper or thesis, present **(i)** overall + subgroup metrics, **(ii)** ablation deltas, **(iii)** one **head_input** embedding or hypersphere figure with **EF** coloring, and **(iv)** optional **sex** coloring with explicit text that **FairHICON-style disentanglement was not trained** unless you add dedicated branches and losses.

---

## 7. File index (quick reference)

| Artifact / script | Role |
|-------------------|------|
| `config.yaml` | Data paths, backbone, fused architecture, training hyperparameters |
| `train_real.py` / `train_fused.py` | Training loops |
| `evaluate_ef_real.py` / `evaluate_ef_fused.py` | Overall metrics → `eval_results/`, `multi_run_results/` |
| `evaluate_fused_demographics.py` | Per–demographic-bin metrics |
| `ablate_demographic_effect.py` | normal / zero / shuffle `demo_vec` (fused or real) |
| `visualize_embeddings.py`, `visualize_subgroups.py` | t-SNE / UMAP; **`--embedding head_input`** recommended for demo-colored plots |
| `visualize_hypersphere.py` | PCA + unit sphere 3D plots |
| `Contrastive Semantic Learning.pdf` | External FairHICON-style reference (genomics); interpret by analogy only |
| `documents/MODEL_RESULTS_REPORT.{docx,html,pdf}` | Exported report (see `export_report.sh`) |
| `assets/report_print.css` | Stylesheet for HTML/PDF export |

---

## 8. Exported document formats

| Format | Path | Use case |
|--------|------|----------|
| **Markdown (source)** | `ef_prediction/MODEL_RESULTS_AND_ANALYSIS_REPORT.md` | Version control, editing in the IDE |
| **Microsoft Word** | `ef_prediction/documents/MODEL_RESULTS_REPORT.docx` | Track changes, supervisor comments, journal supplementary |
| **HTML** | `ef_prediction/documents/MODEL_RESULTS_REPORT.html` | Browser view, print-to-PDF from Chrome if you prefer native print |
| **PDF** | `ef_prediction/documents/MODEL_RESULTS_REPORT.pdf` | Submission, archiving (generated from HTML via xhtml2pdf) |

Rebuild: `bash ef_prediction/export_report.sh`

---

*Report generated from the current `ef_prediction` codebase and on-disk metric files; refresh numbers after new training runs.*
