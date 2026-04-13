import pandas as pd
import os

# =========================
# PATHS
# =========================
ORIGINAL_CSV = "data/processed_full/train_manifest_filtered_clean.csv"
FUSED_CSV = "perfect_synthetic_copies/perfect_copies_train.csv"
OUTPUT_DIR = "ef_prediction/grouped_manifests"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
original_df = pd.read_csv(ORIGINAL_CSV)
fused_df = pd.read_csv(FUSED_CSV)

print("Original size:", len(original_df))
print("Fused size:", len(fused_df))

# =========================
# 🔥 FIX MERGE USING BASE ID
# =========================
def extract_id(path):
    name = os.path.basename(path)
    parts = name.split("-")
    return "-".join(parts[:2])

original_df["base_id"] = original_df["file_path"].apply(extract_id)
fused_df["base_id"] = fused_df["original_path"].apply(extract_id)

merged = pd.merge(fused_df, original_df, on="base_id", how="inner")

print("Merged size:", len(merged))

if len(merged) == 0:
    print("❌ Merge failed")
    exit()

# =========================
# BMI CALCULATION
# =========================
merged["height_m"] = merged["height"] / 100
merged["bmi"] = merged["weight"] / (merged["height_m"] ** 2)

# =========================
# 🔥 AGE BIN FUNCTION (CORRECT FIX)
# =========================
def get_age_bin(age):
    if age <= 1:
        return "0-1"
    elif age <= 2:
        return "1-2"
    elif age <= 3:
        return "2-3"
    elif age <= 5:
        return "3-5"
    elif age <= 8:
        return "5-8"
    elif age <= 12:
        return "8-12"
    elif age <= 15:
        return "12-15"
    else:
        return "15-18"

merged["age_bin"] = merged["age"].apply(get_age_bin)

# =========================
# SEX GROUPS
# =========================
male = merged[merged["sex"] == "M"]
female = merged[merged["sex"] == "F"]

male.to_csv(f"{OUTPUT_DIR}/sex_male.csv", index=False)
female.to_csv(f"{OUTPUT_DIR}/sex_female.csv", index=False)

print("\nSEX:")
print("Male:", len(male))
print("Female:", len(female))

# =========================
# AGE GROUPS
# =========================
age_groups = [
    "0-1","1-2","2-3","3-5",
    "5-8","8-12","12-15","15-18"
]

for group in age_groups:
    df_group = merged[merged["age_bin"] == group]
    df_group.to_csv(f"{OUTPUT_DIR}/age_{group}.csv", index=False)
    print(f"Age {group}:", len(df_group))

# =========================
# BMI GROUPS
# =========================
bmi_under = merged[merged["bmi"] < 18.5]
bmi_normal = merged[(merged["bmi"] >= 18.5) & (merged["bmi"] < 25)]
bmi_over = merged[(merged["bmi"] >= 25) & (merged["bmi"] < 30)]
bmi_obese = merged[merged["bmi"] >= 30]

bmi_under.to_csv(f"{OUTPUT_DIR}/bmi_underweight.csv", index=False)
bmi_normal.to_csv(f"{OUTPUT_DIR}/bmi_normal.csv", index=False)
bmi_over.to_csv(f"{OUTPUT_DIR}/bmi_overweight.csv", index=False)
bmi_obese.to_csv(f"{OUTPUT_DIR}/bmi_obese.csv", index=False)

print("\nBMI:")
print("Underweight:", len(bmi_under))
print("Normal:", len(bmi_normal))
print("Overweight:", len(bmi_over))
print("Obese:", len(bmi_obese))

print("\n✅ ALL GROUPED FILES CREATED CORRECTLY!")