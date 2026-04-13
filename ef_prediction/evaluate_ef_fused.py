import yaml
import torch
import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

from .dataset import DualVideoEFDataset
from .models.pt_efnet_fused import PTEFNetFused


def compute_metrics(preds, labels):
    mae = np.mean(np.abs(preds - labels))
    mse = np.mean((preds - labels) ** 2)
    rmse = np.sqrt(mse)
    denom = np.sum((labels - labels.mean()) ** 2)
    r2 = 0.0 if denom == 0 else 1 - np.sum((labels - preds) ** 2) / denom
    return mae, mse, rmse, r2


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, required=True)
    args = parser.parse_args()

    run_id = args.run

    with open("ef_prediction/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DualVideoEFDataset(
        manifest_path=cfg["data"]["val_manifest_fused"],
        video_root_dir=cfg["data"]["original_video_dir"],
        synthetic_root_dir=cfg["data"]["synthetic_video_dir"],
        video_length=cfg["model"]["video_length"],
        video_size=cfg["model"]["video_size"],
        fused=True
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    backbone = cfg["model"].get("backbone", "resnet34")
    model = PTEFNetFused(backbone=backbone).to(device)
    model.load_state_dict(
        torch.load(f"ef_prediction/checkpoints/fused/run_{run_id}_best.pth",
                   map_location=device)
    )
    model.eval()

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

    mae, mse, rmse, r2 = compute_metrics(preds, labels)

    out_dir = Path("ef_prediction/multi_run_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV per run
    df = pd.DataFrame({
        "True_EF": labels,
        "Predicted_EF_Fused": preds,
        "Error": preds - labels
    })

    df.to_csv(out_dir / f"fused_run_{run_id}.csv", index=False)

    # Append metrics to single JSON
    metrics = {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

    json_path = out_dir / "fused_5run_metrics.json"

    if json_path.exists():
        with open(json_path, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    all_metrics[f"run_{run_id}"] = metrics

    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

    print(f"\nRun {run_id} Results")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")
    print("✓ Results saved and JSON updated\n")


if __name__ == "__main__":
    main()