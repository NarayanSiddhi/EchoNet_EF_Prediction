# Complete List of Grad-CAM PNG Files

## USE CASE 2: DEMOGRAPHIC VARIATIONS

### 1. CARDIAC-SPECIFIC GRAD-CAM (EF Prediction Model)
**Purpose**: Highlights cardiac structures (LV, RV, myocardium) important for EF estimation  
**Model**: PTEFNet (EF Prediction Model)  
**Target**: Ejection Fraction prediction

#### Sample 0000:
- `paper_gradcam_collection/cardiac_specific/sample_0000/cardiac_age_variation.png`
- `paper_gradcam_collection/cardiac_specific/sample_0000/cardiac_bmi_variation.png`
- `paper_gradcam_collection/cardiac_specific/sample_0000/cardiac_sex_variation.png`

#### Sample 0001:
- `paper_gradcam_collection/cardiac_specific/sample_0001/cardiac_age_variation.png`
- `paper_gradcam_collection/cardiac_specific/sample_0001/cardiac_bmi_variation.png`
- `paper_gradcam_collection/cardiac_specific/sample_0001/cardiac_sex_variation.png`

#### Sample 0002:
- `paper_gradcam_collection/cardiac_specific/sample_0002/cardiac_age_variation.png`
- `paper_gradcam_collection/cardiac_specific/sample_0002/cardiac_bmi_variation.png`
- `paper_gradcam_collection/cardiac_specific/sample_0002/cardiac_sex_variation.png`

**Total**: 9 files (3 samples × 3 variations)

---

### 2. DEMOGRAPHIC-SPECIFIC GRAD-CAM (Generator - Demographic Changes)
**Purpose**: Highlights regions that change when demographics change  
**Model**: Perfect Reconstruction C3D-GAN Generator  
**Target**: Difference between original and variation outputs

#### Sample 0000:
- `paper_gradcam_collection/demographic_specific/sample_0000/demographic_age_variation.png`
- `paper_gradcam_collection/demographic_specific/sample_0000/demographic_bmi_variation.png`
- `paper_gradcam_collection/demographic_specific/sample_0000/demographic_sex_variation.png`

#### Sample 0001:
- `paper_gradcam_collection/demographic_specific/sample_0001/demographic_age_variation.png`
- `paper_gradcam_collection/demographic_specific/sample_0001/demographic_bmi_variation.png`
- `paper_gradcam_collection/demographic_specific/sample_0001/demographic_sex_variation.png`

#### Sample 0002:
- `paper_gradcam_collection/demographic_specific/sample_0002/demographic_age_variation.png`
- `paper_gradcam_collection/demographic_specific/sample_0002/demographic_bmi_variation.png`
- `paper_gradcam_collection/demographic_specific/sample_0002/demographic_sex_variation.png`

**Total**: 9 files (3 samples × 3 variations)

---

### 3. GENERATOR-BASED GRAD-CAM (Fixed Variations Correct)
**Purpose**: Shows generator's internal attention during generation  
**Model**: Generator encoder layers  
**Target**: Reconstruction loss

#### Sample 0000:
- `paper_gradcam_collection/fixed_variations_correct/sample_0000/correct_age_variation.png`
- `paper_gradcam_collection/fixed_variations_correct/sample_0000/correct_bmi_variation.png`
- `paper_gradcam_collection/fixed_variations_correct/sample_0000/correct_sex_variation.png`

#### Sample 0001:
- `paper_gradcam_collection/fixed_variations_correct/sample_0001/correct_age_variation.png`
- `paper_gradcam_collection/fixed_variations_correct/sample_0001/correct_bmi_variation.png`
- `paper_gradcam_collection/fixed_variations_correct/sample_0001/correct_sex_variation.png`

#### Sample 0002:
- `paper_gradcam_collection/fixed_variations_correct/sample_0002/correct_age_variation.png`
- `paper_gradcam_collection/fixed_variations_correct/sample_0002/correct_bmi_variation.png`
- `paper_gradcam_collection/fixed_variations_correct/sample_0002/correct_sex_variation.png`

**Total**: 9 files (3 samples × 3 variations)

---

## USE CASE 3: PERFECT RECONSTRUCTION

### 4. PERFECT RECONSTRUCTION GRAD-CAM
**Purpose**: Shows preservation of cardiac attention in perfect copies  
**Model**: EF Prediction Model or Generator  
**Target**: Perfect reconstruction validation

- `use_case_3_perfect_reconstruction/best_gradcam_visualizations/perfect_reconstruction/best_perfect_reconstruction_sample_0000.png`
- `use_case_3_perfect_reconstruction/best_gradcam_visualizations/perfect_reconstruction/best_perfect_reconstruction_sample_1947.png`
- `use_case_3_perfect_reconstruction/best_gradcam_visualizations/perfect_reconstruction/best_perfect_reconstruction_sample_3895.png`
- `use_case_3_perfect_reconstruction/best_gradcam_visualizations/perfect_reconstruction/best_perfect_reconstruction_sample_5842.png`
- `use_case_3_perfect_reconstruction/best_gradcam_visualizations/perfect_reconstruction/best_perfect_reconstruction_sample_7790.png`

**Total**: 5 files

---

## ADDITIONAL COLLECTIONS

### 5. DEMOGRAPHIC VARIATIONS (Original Collection - Overlays)
**Purpose**: Original overlay visualizations from initial collection

#### Sample 0000:
- `paper_gradcam_collection/demographic_variations/overlays/sample_0000_real_overlay.png`
- `paper_gradcam_collection/demographic_variations/overlays/sample_0000_age_variation_overlay.png`
- `paper_gradcam_collection/demographic_variations/overlays/sample_0000_sex_variation_overlay.png`
- `paper_gradcam_collection/demographic_variations/overlays/sample_0000_bmi_variation_overlay.png`

#### Sample 0001:
- `paper_gradcam_collection/demographic_variations/overlays/sample_0001_real_overlay.png`
- `paper_gradcam_collection/demographic_variations/overlays/sample_0001_age_variation_overlay.png`
- `paper_gradcam_collection/demographic_variations/overlays/sample_0001_sex_variation_overlay.png`
- `paper_gradcam_collection/demographic_variations/overlays/sample_0001_bmi_variation_overlay.png`

#### Sample 0002:
- `paper_gradcam_collection/demographic_variations/overlays/sample_0002_real_overlay.png`
- `paper_gradcam_collection/demographic_variations/overlays/sample_0002_age_variation_overlay.png`
- `paper_gradcam_collection/demographic_variations/overlays/sample_0002_sex_variation_overlay.png`
- `paper_gradcam_collection/demographic_variations/overlays/sample_0002_bmi_variation_overlay.png`

#### Sample 0003:
- `paper_gradcam_collection/demographic_variations/overlays/sample_0003_real_overlay.png`
- `paper_gradcam_collection/demographic_variations/overlays/sample_0003_age_variation_overlay.png`
- `paper_gradcam_collection/demographic_variations/overlays/sample_0003_sex_variation_overlay.png`
- `paper_gradcam_collection/demographic_variations/overlays/sample_0003_bmi_variation_overlay.png`

#### Sample 0004:
- `paper_gradcam_collection/demographic_variations/overlays/sample_0004_real_overlay.png`
- `paper_gradcam_collection/demographic_variations/overlays/sample_0004_age_variation_overlay.png`
- `paper_gradcam_collection/demographic_variations/overlays/sample_0004_sex_variation_overlay.png`
- `paper_gradcam_collection/demographic_variations/overlays/sample_0004_bmi_variation_overlay.png`

**Total**: 20 files (5 samples × 4 overlays each)

---

### 6. DEMOGRAPHIC VARIATIONS (Combined Visualizations)
**Purpose**: Combined multi-panel visualizations

- `paper_gradcam_collection/demographic_variations/sample_0000_variations.png`
- `paper_gradcam_collection/demographic_variations/sample_0001_variations.png`
- `paper_gradcam_collection/demographic_variations/sample_0002_variations.png`
- `paper_gradcam_collection/demographic_variations/sample_0003_variations.png`
- `paper_gradcam_collection/demographic_variations/sample_0004_variations.png`

**Total**: 5 files

---

### 7. FIXED VARIATIONS (Comparison Visualizations)
**Purpose**: Comparison visualizations for fixed variations

- `paper_gradcam_collection/fixed_variations/gradcam_comparison_fixed_age_variation.png`
- `paper_gradcam_collection/fixed_variations/gradcam_comparison_fixed_bmi_variation.png`
- `paper_gradcam_collection/fixed_variations/gradcam_comparison_fixed_sex_variation.png`

**Total**: 3 files

---

### 8. DEMOGRAPHIC CATEGORIES
**Purpose**: Grad-CAM for different demographic categories

- `paper_gradcam_collection/demographic_categories/overlays/early_frame.png`
- `paper_gradcam_collection/demographic_categories/overlays/early_heatmap.png`
- `paper_gradcam_collection/demographic_categories/overlays/early.png`
- `paper_gradcam_collection/demographic_categories/overlays/middle_frame.png`
- `paper_gradcam_collection/demographic_categories/overlays/middle_heatmap.png`
- `paper_gradcam_collection/demographic_categories/overlays/middle.png`
- `paper_gradcam_collection/demographic_categories/overlays/normal_frame.png`
- `paper_gradcam_collection/demographic_categories/overlays/normal_heatmap.png`
- `paper_gradcam_collection/demographic_categories/overlays/normal.png`
- `paper_gradcam_collection/demographic_categories/overlays/overweight_frame.png`

**Total**: 10+ files (partial list)

---

### 9. SUBGROUP ANALYSIS
**Purpose**: Subgroup analysis and MICCAI-style visualizations

- `paper_gradcam_collection/subgroup_analysis/demographic_gradcam_grid.png`
- `paper_gradcam_collection/subgroup_analysis/miccai_8_conditions_exact.png`
- `paper_gradcam_collection/subgroup_analysis/miccai_demographic_gradcam_extended.png`
- `paper_gradcam_collection/subgroup_analysis/miccai_demographic_gradcam.png`
- `paper_gradcam_collection/subgroup_analysis/miccai_exact_style_gradcam.png`
- `paper_gradcam_collection/subgroup_analysis/miccai_representative_samples.png`
- `paper_gradcam_collection/subgroup_analysis/sex_subgroup_gradcam.png`

**Total**: 7 files

---

## SUMMARY BY USE CASE

### Use Case 2 (Demographic Variations):
- **Cardiac-Specific**: 9 files
- **Demographic-Specific**: 9 files
- **Generator-Based**: 9 files
- **Original Overlays**: 20 files
- **Combined Visualizations**: 5 files
- **Fixed Comparisons**: 3 files
- **Subtotal**: 55 files

### Use Case 3 (Perfect Reconstruction):
- **Perfect Reconstruction**: 5 files
- **Subtotal**: 5 files

### Additional Collections:
- **Demographic Categories**: 10+ files
- **Subgroup Analysis**: 7 files
- **Subtotal**: 17+ files

**GRAND TOTAL**: 77+ PNG files

---

## RECOMMENDED FILES FOR PAPER

### For Main Results Table (Sample 0002):

**Top Row (Real Video):**
- Use any cardiac-specific file (shows original in top row):
  - `paper_gradcam_collection/cardiac_specific/sample_0002/cardiac_sex_variation.png`

**Bottom Rows (Synthetic Variations):**

**Age Variation:**
- Cardiac: `paper_gradcam_collection/cardiac_specific/sample_0002/cardiac_age_variation.png`
- Demographic: `paper_gradcam_collection/demographic_specific/sample_0002/demographic_age_variation.png`

**Sex Variation:**
- Cardiac: `paper_gradcam_collection/cardiac_specific/sample_0002/cardiac_sex_variation.png`
- Demographic: `paper_gradcam_collection/demographic_specific/sample_0002/demographic_sex_variation.png`

**BMI Variation:**
- Cardiac: `paper_gradcam_collection/cardiac_specific/sample_0002/cardiac_bmi_variation.png`
- Demographic: `paper_gradcam_collection/demographic_specific/sample_0002/demographic_bmi_variation.png`

**Perfect Reconstruction:**
- `use_case_3_perfect_reconstruction/best_gradcam_visualizations/perfect_reconstruction/best_perfect_reconstruction_sample_0000.png`

---

## FILE ORGANIZATION STRUCTURE

```
paper_gradcam_collection/
├── cardiac_specific/              # 9 files (EF model Grad-CAM)
│   ├── sample_0000/
│   ├── sample_0001/
│   └── sample_0002/
│
├── demographic_specific/          # 9 files (Generator demographic Grad-CAM)
│   ├── sample_0000/
│   ├── sample_0001/
│   └── sample_0002/
│
├── fixed_variations_correct/      # 9 files (Generator encoder Grad-CAM)
│   ├── sample_0000/
│   ├── sample_0001/
│   └── sample_0002/
│
├── demographic_variations/        # 25 files (original collection)
│   ├── overlays/                 # 20 files
│   └── sample_XXXX_variations.png # 5 files
│
├── fixed_variations/             # 3 files
│
├── demographic_categories/        # 10+ files
│
└── subgroup_analysis/            # 7 files

use_case_3_perfect_reconstruction/
└── best_gradcam_visualizations/
    └── perfect_reconstruction/    # 5 files
```
