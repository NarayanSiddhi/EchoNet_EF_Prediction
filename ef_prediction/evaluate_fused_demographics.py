import yaml
import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from torch.utils.data import DataLoader

from ef_prediction.dataset_demographics import DualVideoEFDataset
from ef_prediction.models.pt_efnet_fused import PTEFNetFused


def compute_metrics(preds, labels):
    mae = np.mean(np.abs(preds - labels))
    mse = np.mean((preds - labels) ** 2)
    rmse = np.sqrt(mse)
    denom = np.sum((labels - labels.mean()) ** 2)
    r2 = 0.0 if denom == 0 else 1 - np.sum((labels - preds) ** 2) / denom
    return mae, rmse, r2


GROUP_DIR = Path("ef_prediction/grouped_manifests")
OUTPUT_DIR = Path("ef_prediction/group_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = "ef_prediction/checkpoints/fused/run_1_best.pth"

with open("ef_prediction/config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone = cfg["model"].get("backbone", "resnet34")
model = PTEFNetFused(backbone=backbone).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

results = {}

csv_files = sorted(GROUP_DIR.glob("*.csv"))

print(f"Found {len(csv_files)} files\n")

for csv_file in csv_files:

    name = csv_file.stem
    print(f"Processing: {name}")

    dataset = DualVideoEFDataset(
        manifest_path=str(csv_file),
        video_root_dir=cfg["data"]["original_video_dir"],
        synthetic_root_dir=cfg["data"]["synthetic_video_dir"],
        video_length=cfg["model"]["video_length"],
        video_size=cfg["model"]["video_size"],
        fused=True
    )

    if len(dataset) == 0:
        print("Skipped\n")
        continue

    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    preds, labels = [], []

    with torch.no_grad():
        for real_video, syn_video, ef, _, _, _, demo_vec in loader:

            real_video = real_video.to(device)
            syn_video = syn_video.to(device)
            ef = ef.to(device)
            demo_vec = demo_vec.to(device).float()

            pred, _ = model(real_video, syn_video, demo_vec)

            preds.extend(pred.cpu().numpy())
            labels.extend(ef.cpu().numpy())

    preds = np.array(preds) * 100
    labels = np.array(labels) * 100

    mae, rmse, r2 = compute_metrics(preds, labels)

    df = pd.DataFrame({
        "True_EF": labels,
        "Predicted_EF": preds,
        "Error": preds - labels
    })

    df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

    results[name] = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "count": int(len(labels))
    }

    print(f"MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}\n")

with open(OUTPUT_DIR / "all_group_metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("✅ DONE — All groups evaluated!")