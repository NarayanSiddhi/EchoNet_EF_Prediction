import json
import pandas as pd
from pathlib import Path

# =========================
# PATHS
# =========================
GROUP_JSON = "ef_prediction/group_results/all_group_metrics.json"
FUSED_JSON = "ef_prediction/multi_run_results/fused_5run_metrics.json"
REAL_JSON = "ef_prediction/eval_results/real_metrics.json"
OUTPUT_DIR = Path("ef_prediction/group_results")

# =========================
# LOAD FILES
# =========================
with open(GROUP_JSON, "r") as f:
    group_data = json.load(f)

with open(FUSED_JSON, "r") as f:
    fused_data = json.load(f)

with open(REAL_JSON, "r") as f:
    real_data = json.load(f)

fused = fused_data["run_1"]
real = real_data

# =========================
# BUILD TABLE
# =========================
rows = []

# 🔥 OVERALL FUSED
rows.append({
    "Group": "OVERALL_FUSED",
    "MAE": fused["MAE"],
    "MSE": fused["RMSE"] ** 2,
    "RMSE": fused["RMSE"],
    "R2": fused["R2"],
    "Count": "ALL"
})

# 🔥 OVERALL REAL
rows.append({
    "Group": "OVERALL_REAL",
    "MAE": real["MAE"],
    "MSE": real["RMSE"] ** 2,
    "RMSE": real["RMSE"],
    "R2": real["R2"],
    "Count": "ALL"
})

# =========================
# GROUP RESULTS
# =========================
for group, metrics in group_data.items():
    rows.append({
        "Group": group,
        "MAE": metrics["MAE"],
        "MSE": metrics["RMSE"] ** 2,
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
        "Count": metrics["count"]
    })

df = pd.DataFrame(rows)

# =========================
# SORT GROUPS ONLY
# =========================
base_rows = df[df["Group"].isin(["OVERALL_FUSED", "OVERALL_REAL"])]
group_rows = df[~df["Group"].isin(["OVERALL_FUSED", "OVERALL_REAL"])]

group_rows = group_rows.sort_values(by="MAE")

final_df = pd.concat([base_rows, group_rows])

# =========================
# SAVE
# =========================
final_df.to_csv(OUTPUT_DIR / "final_results_summary.csv", index=False)

print("\n===== FINAL RESULTS =====")
print(final_df.to_string(index=False))

print("\n✅ FINAL SUMMARY WITH REAL + FUSED CREATED!")