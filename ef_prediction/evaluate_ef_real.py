import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from .dataset import DualVideoEFDataset
from .models.pt_efnet_real import PTEFNetReal


def compute_metrics(preds, labels):
    mae = np.mean(np.abs(preds - labels))
    mse = np.mean((preds - labels) ** 2)
    rmse = np.sqrt(mse)
    denom = np.sum((labels - labels.mean()) ** 2)
    r2 = 0.0 if denom == 0 else 1 - np.sum((labels - preds) ** 2) / denom
    return mae, mse, rmse, r2


def main():

    with open("ef_prediction/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataset
    val_ds = DualVideoEFDataset(
        manifest_path=cfg["data"]["val_manifest"],
        video_root_dir=cfg["data"]["original_video_dir"],
        video_length=cfg["model"]["video_length"],
        video_size=cfg["model"]["video_size"],
        fused=False
    )

    loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    backbone = cfg["model"].get("backbone", "resnet34")
    model = PTEFNetReal(backbone=backbone).to(device)
    model.load_state_dict(
        torch.load("ef_prediction/checkpoints/real/best.pth", map_location=device)
    )
    model.eval()

    preds, labels = [], []

    with torch.no_grad():
        for video, ef, _, _, _, demo_vec in tqdm(loader, desc="Evaluating Real Model"):
            video = video.to(device)
            ef = ef.to(device)
            demo_vec = demo_vec.to(device).float()

            pred, _ = model(video, demo_vec)

            preds.extend(pred.cpu().numpy())
            labels.extend(ef.cpu().numpy())

    preds = np.array(preds) * 100
    labels = np.array(labels) * 100

    mae, mse, rmse, r2 = compute_metrics(preds, labels)

    print("\nREAL MODEL RESULTS")
    print(f"MAE  : {mae:.2f}")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")

    # Save results
    out_dir = Path("ef_prediction/eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "True_EF": labels,
        "Predicted_EF": preds,
        "Error": preds - labels
    })

    df.to_csv(out_dir / "real_results.csv", index=False)

    metrics = {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

    pd.Series(metrics).to_json(out_dir / "real_metrics.json")

    print("\nSaved real_results.csv and real_metrics.json")


if __name__ == "__main__":
    main()