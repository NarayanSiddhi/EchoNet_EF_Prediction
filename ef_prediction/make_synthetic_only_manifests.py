"""
Create synthetic-only manifests for training a real-only model on generated videos.

Input: perfect_synthetic_copies/perfect_copies_{train,val}.csv
Output: ef_prediction/synthetic_only_manifests/{train,val}_synthetic_only.csv

The output CSVs intentionally do NOT include an `original_path` column, so that
DualVideoEFDataset (fused=False) will use `processed_path` as the video path.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def convert(split: str) -> Path:
    src = ROOT / "perfect_synthetic_copies" / f"perfect_copies_{split}.csv"
    if not src.is_file():
        raise FileNotFoundError(f"Missing source manifest: {src}")

    df = pd.read_csv(src)
    required = {"synthetic_path", "EF", "demographics"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{src} missing columns: {sorted(missing)}")

    out_dir = ROOT / "ef_prediction" / "synthetic_only_manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    out = pd.DataFrame(
        {
            # DualVideoEFDataset(fused=False) reads `processed_path` if `original_path` is absent.
            "processed_path": df["synthetic_path"].astype(str),
            "EF": df["EF"],
            "demographics": df["demographics"].astype(str),
        }
    )

    out_path = out_dir / f"{split}_synthetic_only.csv"
    out.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    train_path = convert("train")
    val_path = convert("val")
    print(f"✓ Wrote: {train_path}")
    print(f"✓ Wrote: {val_path}")


if __name__ == "__main__":
    main()

