# Paper revision checklist: Use Cases 1–3 vs repository

This document maps **required and optional** manuscript updates to the **EchoNet-Pediatric-BIGAN-AUGMENTATION** codebase. Use it alongside your LaTeX source (e.g. HICSS template).

**Authoritative code locations**

| Use case | Primary paths |
|----------|----------------|
| **UC1** | `use_case_1_balance_dataset/c3dgan/` (`models_improved.py`, `train_*.py`, `generate*.py`), `evaluate_quality_metrics.py`, `UC1_QUALITY_EVALUATION_RESULTS.md` |
| **UC2** | `use_case_2_demographic_variations/generate_demographic_variations_fixed.py` |
| **UC3** | `use_case_3_perfect_reconstruction/train_reconstruction.py`, `models.py` |
| **128×128 pipeline** | `run_augmentation_128x128.sh`, metrics: `uc2_128x128_metrics.json`, `uc3_128x128_metrics.json`, `uc2_128x128_fid.json` |
| **Do not confuse** | Repo-root `Data_Augmentation.py` uses `PerfectReconstructionGenerator` (UC2/3 family), **not** the UC1 DCGAN. |

---

## 1. Executive summary (what is wrong today)

1. **UC2/UC3 preprocessing** in the paper (16 frames, 64×64, **14-D** demographics) does **not** match the main inference/training defaults in code (**32** frames, **64 or 128** spatial, **11-D** sex+age+BMI one-hot).
2. **UC2 variation protocol** in the paper (eight age bins cycled, etc.) does **not** match `generate_demographic_variations_fixed.py` (**three** outputs per video: one age nudge in a **5-bin** coarse scheme, sex flip, BMI rotate).
3. **“Four-stage diversity pipeline”** prose should be reconciled with **`generate_diverse_variation`** (reference mixing + noisy multi-candidate selection + blending — not the same four bullets as written).
4. **Figure / framework caption** attributes one U-Net architecture to **all** three use cases; **UC1** in the paper is a **DCGAN**, not the same U-Net as UC2/UC3.
5. If you report the **128×128** experiment, **UC2** pixel metrics are **low** (SSIM ~0.08, PSNR ~15 dB in `uc2_128x128_metrics.json`) — do not blend that story with **UC3** near-reconstruction unless you label each clearly.
6. **UC3 table N and metrics**: perfect-copy manifest used for 128×128 metrics shows **n = 6227** in `uc3_128x128_metrics.json`, not necessarily 7,791; align **N**, resolution, and SSIM/PSNR with the manifest you actually evaluate.

---

## 2. Global changes (anywhere they appear)

### 2.1 Abstract

**Where:** opening `\begin{abstract}` … `\end{abstract}`.

**Changes**

- If you only ship **64×64** UC2/UC3: set preprocessing claims in the abstract to **32 frames** and **11-D** conditioning (or say “video-length T and demographic vector as in §Experimental Setup” without wrong numbers).
- If you add **128×128**: add **one clause** that UC2/UC3 were also run at 128×128 with checkpoint `use_case_3_perfect_reconstruction/ckpt_uc3_128x128_T32/recon_best.pt`, and **do not** claim UC2 has UC3-level SSIM unless you cite the correct UC2 metrics file.
- **UC1 FID 75.08**: keep if still from `UC1_QUALITY_EVALUATION_RESULTS.md` (200 vs 200). Optionally add “after evaluator resampling (see §Experimental Setup).”
- **Reconstruction sentence** (“SSIM … PSNR …”): tie explicitly to **UC3 only**, not to “all synthetic outputs.”

### 2.2 Keywords

**Where:** `\noindent \textbf{Keywords:}` line.

**Changes:** None required unless you add “video-to-video reconstruction” vs “unconditional generation.”

### 2.3 Introduction (`\section{Introduction}`)

**Where:** paragraph that introduces three use cases; any sentence that implies one architecture for all.

**Changes**

- State explicitly: **UC1 = conditional 3D DCGAN from noise**; **UC2 and UC3 = shared conditional 3D U-Net / `PerfectReconstructionGenerator`-style video-to-video path** (use your exact model name from the paper).
- Remove or rephrase any implication that **Grad-CAM** was run on **UC1 noise samples** unless you actually did that experiment (codebase emphasis is UC2/3 + EF models in `ef_prediction/`).

### 2.4 Figure: overall framework (`fig:c3dgan_framework`)

**Where:** `\begin{figure*}[t]` with `C3DGAN.pdf`; `\caption{...}`.

**Changes**

- **Caption:** Split outputs by use case: **(1) DCGAN** for UC1; **(2–3) U-Net / reconstruction generator** for UC2 and UC3. Avoid “single U-Net for all three” unless the figure is redrawn.
- **Optional:** add subpanels or callouts in the figure PDF itself (editorial, outside this MD).

---

## 3. Methodology (`\section{Methodology}`)

### 3.1 Opening paragraph (distributional framing)

**Where:** first paragraph after `\section{Methodology}`.

**Changes:** Optional clarity: “UC2 and UC3 share the same generator class with different conditioning targets (altered vs matched demographics).”

### 3.2 Subsection: Use Case 1 — Conditional 3D DCGAN

**Where:** `\subsection{Use Case~1: ...}`.

**Changes**

- **Output tensor:** Code in `use_case_1_balance_dataset/c3dgan/models_improved.py` documents **`[B, 1, 96, 128, 128]`** (96 temporal, 128 spatial). Keep aligned with that file, not with root `Data_Augmentation.py`.
- **Class labels:** Generator `n_classes=20` appears in code; manuscript says **19** underrepresented subgroups. **Pick one:** (a) explain 20th class (e.g. unused / “other”), or (b) set `n_classes=19` in text only if code truly uses 19 — verify in `c3dgan/config.yaml` / training script.
- **FID:** Keep **75.08** only if evaluation is still `evaluate_quality_metrics.py` with 200/200. Cite internal doc `use_case_1_balance_dataset/UC1_QUALITY_EVALUATION_RESULTS.md`.

### 3.3 Subsection: Use Case 2 — Demographic variation

**Where:** `\subsection{Use Case~2: ...}`.

**Replace / align** the following with `generate_demographic_variations_fixed.py`:

| Topic | Paper risk | Repo behavior |
|-------|------------|----------------|
| Variations per source | “Three counterparts” with eight age cycles | **Exactly three** MP4s per row: `age_variation`, `sex_variation`, `bmi_variation`. |
| Age | Eight dataset age bins | Conditioning uses **`np.digitize(age, [0,5,10,15,18])`** → **5** coarse bins in `encode_demographics`; `get_demographic_variations` uses **`(age_bin_idx + 1) % 5`**, not an 8-bin walk. |
| Sex | “Inverted” | **F↔M** style flip via `1 - sex_idx`; **O** maps to same index as F in `sex_map`. |
| BMI | “Rotated” | **`(bmi_idx + 1) % 4`** over `[underweight, normal, overweight, obese]`. |
| Demographic vector | 14-D | **11-D** = 2 + 5 + 4 in `encode_demographics`. |
| Diversity | Four named stages | Describe **reference-video path** (mix outputs + mix inputs + additive noise) vs **no-reference path** (many noisy-input forwards, score by MSE diversity, blend top candidates). Optionally keep “four ideas” as conceptual grouping **if** you map each bullet to these mechanisms. |

**Add (new paragraph, same subsection, no new `\section` required)**

- Conditioning modes: if you use **FiLM vs concat**, cite `train_reconstruction.py` docstring and checkpoint field `conditioning` in saved `.pt` files.

### 3.4 Subsection: Use Case 3 — Reconstruction validation

**Where:** `\subsection{Use Case~3: ...}`.

**Changes**

- Say inputs are **the same tensor format as UC2** (T, H, W, normalization) with **source demographics** passed to the generator (identity reconstruction task).
- **Loss equation:** Verify each term and **(λ_pix, λ_ssim, λ_temp, λ_gan, λ_demo)** against the **actual** training script / config used for the checkpoint you cite. If they differ, update the equation or add “weights in supplementary table.”

---

## 4. Experimental setup (`\section{Experimental Setup}`)

### 4.1 Dataset subsection

**Where:** `\subsection{Dataset}`.

**Changes**

- If **N = 7,791** and **19** subgroups &lt; 500 are still correct for **your** processed manifest, keep — but add **which manifest** (e.g. `data/processed_full/manifest_full.csv` vs filtered train manifest).
- **New optional sentence (same subsection):** “Augmentation and reconstruction experiments use train-filtered manifests where noted (e.g. `train_manifest_filtered_clean.csv` in `run_augmentation_128x128.sh`).”

### 4.2 Preprocessing subsection — **major rewrite**

**Where:** `\subsection{Preprocessing}` (currently: UC2/3 at 16×64×64, 14-D; UC1 at 96×128×128).

**Replace with structure like this (you can paste and fill numbers from your run):**

```text
Use Cases 2 and 3 load grayscale echocardiogram clips of T frames (default T=32 in
generate_demographic_variations_fixed.py and train_reconstruction.py), spatially
resized to H×H with H∈{64,128} depending on the checkpoint's spatial_size, with
intensities scaled to [-1,1]. Demographics are encoded as an 11-dimensional vector:
2 sex + 5 coarse age bins + 4 BMI categories (see code: encode_demographics).

Use Case 1 (DCGAN) is trained/generated at [B,1,96,128,128] as in models_improved.py,
with class-conditional labels; evaluation for FID may resample temporally/spatially
to match the feature extractor (see UC1_QUALITY_EVALUATION_RESULTS.md).
```

**Remove** incorrect claims: **16 frames** for UC2/UC3 default pipeline, **14-D** vectors.

### 4.3 Training configuration subsection

**Where:** `\subsection{Training Configuration}`.

**Changes**

- **Epochs:** Confirm **200** vs actual (e.g. `train_reconstruction.py` example shows **50** in docstring — your checkpoint may still be 200; verify).
- **Diversity “1.3% → 5.89%”:** Tie to a **logged experiment** or recompute from `diversity_stats` / saved CSVs; if unverifiable, soften to qualitative language or cite a specific log file path and date.

---

## 5. Results (`\section{Results}`)

### 5.1 Dataset expansion table (`tab:augmentation`)

**Where:** table “Dataset Statistics Across Use Cases.”

**Changes**

- Recompute **Synth.** and **Total** from the **actual** `synthetic_manifest.csv`, `variations_manifest.csv`, and UC3 pair manifest you submit with the paper.
- If **128×128** UC2 folder exists (`demographic_variations_128x128`), row counts may differ from 64×64 runs.

### 5.2 UC1 FID table (`tab:fid`)

**Where:** FID table.

**Changes**

- Footnote: evaluation script path, **200/200** samples, any **resize/crop** to 16×128×128 for Inception (per `UC1_QUALITY_EVALUATION_RESULTS.md`).

### 5.3 Demographic redistribution (`tab:distribution`)

**Where:** “Before and After Use Case 2” table.

**Changes**

- Add a table note: **“Percentages computed on [original manifest / augmented manifest path]; age bins are dataset metadata bins (8 rows), which may differ from the 5-bin conditioning used inside the generator.”**  
  This avoids reviewers catching an inconsistency between **8-row table** and **5-bin model**.

### 5.4 Reconstruction fidelity (`tab:recon`)

**Where:** UC3 SSIM/PSNR/MSE table.

**Changes**

- Set **N** to the number of rows in the manifest you evaluate (e.g. `perfect_synthetic_copies_128x128/perfect_copies_manifest.csv`).
- Update **mean ± std / pass rates** from that run, or split into two sub-tables:

**Optional new heading (same `\section{Results}`)**

Add:

```latex
\subsection{High-resolution extension (128$\times$128)}
```

Under it, report from JSON files:

| File | Role |
|------|------|
| `uc2_128x128_metrics.json` | UC2 SSIM/PSNR/MSE (expect low SSIM — intentional variation + domain gap) |
| `uc3_128x128_metrics.json` | UC3 reconstruction quality |
| `uc2_128x128_fid.json` | Paired FID subset (200 pairs in script) |

State clearly these are **additional** to the main 64×64 story **or** replace the main story if 128×128 is the only shipped artifact.

### 5.5 Grad-CAM subsection

**Where:** `\subsection{Grad-CAM ...}`.

**Changes**

- Specify **which model** produced the maps (EF predictor in `ef_prediction/`, which checkpoint, real vs **UC2** or **UC3** synthetic). Do not imply UC1 without evidence.

---

## 6. Discussion (`\section{Discussion}`)

**Where:** paragraphs tying UC3 metrics to UC2 “ceiling.”

**Changes**

- **UC3 as upper bound for UC2:** Keep only if you define “bound” as **same architecture, matched demographics** — not as “pixel similarity of demographic edits.”
- Mention **trade-off:** UC2 diversity objectives **lower** frame-wise similarity to the source by design (`generate_diverse_variation`).

---

## 7. Conclusion (`\section{Conclusion}`)

**Where:** closing paragraphs.

**Changes**

- Same as abstract: **separate** claims for **UC1 (FID)**, **UC2 (redistribution / diversity)**, **UC3 (reconstruction)**.
- Remove blanket “high perceptual realism” for **all** pathways if UC2 metrics contradict.

---

## 8. Bibliography / supplementary material (optional additions)

**Where:** main text near first citation of metrics, or `\appendix` if your venue allows.

**Suggested additions**

1. **Supplementary table S1:** Loss weights, optimizer, batch size, manifest paths, commit hash or date.
2. **Supplementary table S2:** Row counts per output folder and manifest filenames.

---

## 9. Quick “diff checklist” (copy for PR / co-authors)

- [ ] Abstract: UC2/UC3 numbers vs UC1 FID disambiguated; reconstruction tied to UC3 only  
- [ ] Fig. 1 caption: UC1 = DCGAN; UC2/3 = U-Net / shared generator  
- [ ] §Method UC2: 3 variations; 11-D; 5-bin age in model; diversity = `generate_diverse_variation`  
- [ ] §Method UC3: λ values verified against training code  
- [ ] §Setup Preprocessing: **32** frames; **64 or 128**; **11-D**; remove **14-D** and default **16** for UC2/3  
- [ ] §Results `tab:recon`: N and metrics match evaluated manifest  
- [ ] §Results `tab:distribution`: footnote on 8-bin table vs 5-bin conditioning  
- [ ] Optional new `\subsection{High-resolution extension (128$\times$128)}` with `uc2_*` / `uc3_*` JSON  
- [ ] Discussion: UC3 “ceiling” wording qualified  
- [ ] Root `Data_Augmentation.py` not cited as UC1 unless you redefine UC1 in the paper  

---

## 10. One-line “where to add headings” summary

| Location in LaTeX | Action |
|-------------------|--------|
| `\section{Results}` | **Optional** `\subsection{High-resolution extension (128$\times$128)}` |
| `\appendix` or supplementary | **Optional** tables: hyperparameters + manifest row counts |
| `\subsection{Preprocessing}` | **Mandatory** rewrite (frames, dims, 11-D) |
| `\subsection{Use Case~2}` | **Mandatory** protocol + diversity description alignment |
| `Figure~\ref{fig:c3dgan_framework}` caption | **Mandatory** architecture clarification |

---

*End of checklist. Regenerate all reported numbers from your final manifests before submission.*
