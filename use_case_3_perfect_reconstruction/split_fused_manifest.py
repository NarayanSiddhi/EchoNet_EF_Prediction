import pandas as pd
from pathlib import Path

# ==============================
# Paths (modify only if needed)
# ==============================

real_train_path = "data/processed_full/train_manifest_filtered_clean.csv"
real_val_path = "data/processed_full/val_manifest.csv"

fused_manifest_path = "perfect_synthetic_copies/perfect_copies_manifest.csv"

output_train_path = "perfect_synthetic_copies/perfect_copies_train.csv"
output_val_path = "perfect_synthetic_copies/perfect_copies_val.csv"


def main():

    print("Loading real train/val manifests...")
    real_train = pd.read_csv(real_train_path)
    real_val = pd.read_csv(real_val_path)

    print("Loading fused manifest...")
    fused = pd.read_csv(fused_manifest_path)

    # Ensure consistent column naming
    if "processed_path" in real_train.columns:
        real_train_paths = set(real_train["processed_path"])
    else:
        raise ValueError("processed_path not found in real train manifest")

    if "processed_path" in real_val.columns:
        real_val_paths = set(real_val["processed_path"])
    else:
        raise ValueError("processed_path not found in real val manifest")

    # Fused uses 'original_path'
    if "original_path" not in fused.columns:
        raise ValueError("original_path not found in fused manifest")

    print("Splitting fused manifest based on original train/val IDs...")

    fused_train = fused[fused["original_path"].isin(real_train_paths)]
    fused_val = fused[fused["original_path"].isin(real_val_paths)]

    print(f"Fused Train size: {len(fused_train)}")
    print(f"Fused Val size  : {len(fused_val)}")

    # Save
    Path("perfect_synthetic_copies").mkdir(parents=True, exist_ok=True)

    fused_train.to_csv(output_train_path, index=False)
    fused_val.to_csv(output_val_path, index=False)

    print("\n✅ Saved:")
    print(output_train_path)
    print(output_val_path)


if __name__ == "__main__":
    main()