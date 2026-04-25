import json
from pathlib import Path

import pandas as pd

# =========================
# PATHS
# =========================
GROUP_JSON = Path("ef_prediction/group_results/all_group_metrics.json")
MULTI_DIR = Path("ef_prediction/multi_run_results")
FUSED_LEGACY_JSON = MULTI_DIR / "fused_5run_metrics.json"
REAL_JSON = Path("ef_prediction/eval_results/real_metrics.json")
OUTPUT_DIR = Path("ef_prediction/group_results")


def load_latest_fused_overall() -> tuple[dict, str]:
    """Return dict with MAE, MSE, RMSE, R2 and the path used."""
    # e.g. fused_run_1_20260419_173306_metrics.json (mtime = newest wins)
    candidates = sorted(
        MULTI_DIR.glob("fused_run_*_metrics.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        with open(path, "r") as f:
            data = json.load(f)
        if all(k in data for k in ("MAE", "MSE", "RMSE", "R2")):
            return {
                "MAE": data["MAE"],
                "MSE": data["MSE"],
                "RMSE": data["RMSE"],
                "R2": data["R2"],
            }, str(path.resolve())

    if FUSED_LEGACY_JSON.is_file():
        with open(FUSED_LEGACY_JSON, "r") as f:
            legacy = json.load(f)
        fused = legacy.get("run_1") or legacy
        mse = fused.get("MSE")
        if mse is None and "RMSE" in fused:
            mse = fused["RMSE"] ** 2
        return {
            "MAE": fused["MAE"],
            "MSE": mse,
            "RMSE": fused["RMSE"],
            "R2": fused["R2"],
        }, str(FUSED_LEGACY_JSON.resolve())

    raise FileNotFoundError(
        "No fused overall metrics: add fused_run_*_*_metrics.json under "
        f"{MULTI_DIR} or restore {FUSED_LEGACY_JSON}"
    )


def main():
    with open(GROUP_JSON, "r") as f:
        group_data = json.load(f)

    fused_overall, fused_src = load_latest_fused_overall()

    with open(REAL_JSON, "r") as f:
        real_data = json.load(f)

    rows = []

    rows.append(
        {
            "Group": "OVERALL_FUSED",
            "MAE": fused_overall["MAE"],
            "MSE": fused_overall["MSE"],
            "RMSE": fused_overall["RMSE"],
            "R2": fused_overall["R2"],
            "Count": "ALL",
        }
    )

    rows.append(
        {
            "Group": "OVERALL_REAL",
            "MAE": real_data["MAE"],
            "MSE": real_data.get("MSE", real_data["RMSE"] ** 2),
            "RMSE": real_data["RMSE"],
            "R2": real_data["R2"],
            "Count": "ALL",
        }
    )

    for group, metrics in group_data.items():
        rows.append(
            {
                "Group": group,
                "MAE": metrics["MAE"],
                "MSE": metrics["RMSE"] ** 2,
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "Count": metrics["count"],
            }
        )

    df = pd.DataFrame(rows)

    base_rows = df[df["Group"].isin(["OVERALL_FUSED", "OVERALL_REAL"])]
    group_rows = df[~df["Group"].isin(["OVERALL_FUSED", "OVERALL_REAL"])]
    group_rows = group_rows.sort_values(by="MAE")
    final_df = pd.concat([base_rows, group_rows])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "final_results_summary.csv"
    final_df.to_csv(out_path, index=False)

    print(f"OVERALL_FUSED source: {fused_src}")
    print(f"OVERALL_REAL source:  {REAL_JSON.resolve()}")
    print("\n===== FINAL RESULTS =====")
    print(final_df.to_string(index=False))
    print(f"\nWrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
