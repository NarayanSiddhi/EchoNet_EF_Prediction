"""
Summarize subgroup EF metrics on the validation split only.

Reads existing per-clip prediction CSVs (no model inference) and joins them
with demographic fields from the validation manifest. Writes outputs under
ef_prediction/val_group_results/ without modifying group_results/.

Fused predictions default to fused_run_1_20260419_173306.csv (matches Table 3).
Real predictions default to eval_results/real_results.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ef_prediction.demographics_utils import compute_bmi_category

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "ef_prediction" / "val_group_results"

SEX_LABELS = {"M": "sex_male", "F": "sex_female", "O": "sex_other"}
AGE_BINS = ["0-1", "2-5", "6-10", "11-15", "16-18"]
BMI_BINS = ["underweight", "normal", "overweight", "obese"]


def compute_metrics(errors: np.ndarray, labels: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    denom = float(np.sum((labels - labels.mean()) ** 2))
    # errors = predicted - true
    r2 = 0.0 if denom == 0 else float(1 - np.sum(errors ** 2) / denom)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "count": int(len(errors))}


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "bmi_category" not in df.columns:
        df["bmi_category"] = df.apply(
            lambda r: compute_bmi_category(r.get("weight"), r.get("height")),
            axis=1,
        )
    return df


def attach_predictions(
    manifest: pd.DataFrame,
    pred_path: Path,
    true_col: str,
    pred_col: str,
) -> pd.DataFrame:
    preds = pd.read_csv(pred_path)
    if len(manifest) != len(preds):
        raise ValueError(
            f"Row count mismatch: manifest {len(manifest)} vs predictions {len(preds)} "
            f"({pred_path})"
        )
    out = manifest.copy().reset_index(drop=True)
    out["True_EF"] = preds[true_col].astype(float).values
    out["Predicted_EF"] = preds[pred_col].astype(float).values
    out["Error"] = out["Predicted_EF"] - out["True_EF"]
    return out


def subgroup_key_sex(row: pd.Series) -> str | None:
    sex = str(row.get("sex", "")).strip().upper()
    return SEX_LABELS.get(sex)


def subgroup_key_age(row: pd.Series) -> str | None:
    age_bin = str(row.get("age_bin", "")).strip()
    if age_bin in AGE_BINS:
        return f"age_{age_bin}"
    return None


def subgroup_key_bmi(row: pd.Series) -> str | None:
    bmi = str(row.get("bmi_category", "")).strip().lower()
    if bmi in BMI_BINS:
        return f"bmi_{bmi}"
    return None


def evaluate_model(
    df: pd.DataFrame,
    model_label: str,
    output_dir: Path,
) -> dict[str, dict]:
    labels = df["True_EF"].to_numpy(dtype=float)
    errors = df["Error"].to_numpy(dtype=float)

    overall = compute_metrics(errors, labels)
    overall_row = {
        "Group": f"OVERALL_{model_label.upper()}",
        **overall,
    }

    group_metrics: dict[str, dict] = {}
    summary_rows = [overall_row]

    key_fns = [
        ("sex", subgroup_key_sex),
        ("age", subgroup_key_age),
        ("bmi", subgroup_key_bmi),
    ]

    for _, key_fn in key_fns:
        for key, part in df.groupby(df.apply(key_fn, axis=1)):
            if key is None or pd.isna(key):
                continue
            part_labels = part["True_EF"].to_numpy(dtype=float)
            part_errors = part["Error"].to_numpy(dtype=float)
            metrics = compute_metrics(part_errors, part_labels)
            group_metrics[str(key)] = metrics
            summary_rows.append({"Group": str(key), **metrics})

            part[["processed_path", "file_name", "sex", "age_bin", "bmi_category",
                  "True_EF", "Predicted_EF", "Error"]].to_csv(
                output_dir / f"{str(key)}_{model_label}.csv",
                index=False,
            )

    summary_df = pd.DataFrame(summary_rows)
    base = summary_df[summary_df["Group"].str.startswith("OVERALL_")]
    groups = summary_df[~summary_df["Group"].str.startswith("OVERALL_")].sort_values("MAE")
    summary_df = pd.concat([base, groups], ignore_index=True)
    summary_df.to_csv(output_dir / f"val_subgroup_summary_{model_label}.csv", index=False)

    payload = {
        "model": model_label,
        "manifest": str(df.attrs.get("manifest_path", "")),
        "predictions": str(df.attrs.get("predictions_path", "")),
        "n_clips": int(len(df)),
        "overall": {k: overall[k] for k in ("MAE", "MSE", "RMSE", "R2", "count")},
        "groups": group_metrics,
    }
    with open(output_dir / f"val_subgroup_metrics_{model_label}.json", "w") as f:
        json.dump(payload, f, indent=2)

    return group_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only subgroup EF summary.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Directory for new validation subgroup artifacts.",
    )
    parser.add_argument(
        "--fused-predictions",
        type=Path,
        default=ROOT / "ef_prediction/multi_run_results/fused_run_1_20260419_173306.csv",
    )
    parser.add_argument(
        "--real-predictions",
        type=Path,
        default=ROOT / "ef_prediction/eval_results/real_results.csv",
    )
    parser.add_argument("--fused-only", action="store_true")
    parser.add_argument("--real-only", action="store_true")
    args = parser.parse_args()

    with open(ROOT / "ef_prediction/config.yaml") as f:
        cfg = yaml.safe_load(f)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = ROOT / cfg["data"]["val_manifest"]
    manifest = load_manifest(manifest_path)
    manifest.attrs["manifest_path"] = str(manifest_path)

    run_fused = not args.real_only
    run_real = not args.fused_only

    if run_fused:
        fused_preds = args.fused_predictions
        fused_df = attach_predictions(
            manifest,
            fused_preds,
            true_col="True_EF",
            pred_col="Predicted_EF_Fused",
        )
        fused_df.attrs["manifest_path"] = str(manifest_path)
        fused_df.attrs["predictions_path"] = str(fused_preds)
        fused_df.to_csv(out_dir / "val_predictions_merged_fused.csv", index=False)
        evaluate_model(fused_df, "fused", out_dir)
        print(f"✓ Fused validation subgroups → {out_dir}")

    if run_real:
        real_preds = args.real_predictions
        real_df = attach_predictions(
            manifest,
            real_preds,
            true_col="True_EF",
            pred_col="Predicted_EF",
        )
        real_df.attrs["manifest_path"] = str(manifest_path)
        real_df.attrs["predictions_path"] = str(real_preds)
        real_df.to_csv(out_dir / "val_predictions_merged_real.csv", index=False)
        evaluate_model(real_df, "real", out_dir)
        print(f"✓ Real validation subgroups → {out_dir}")


if __name__ == "__main__":
    main()
