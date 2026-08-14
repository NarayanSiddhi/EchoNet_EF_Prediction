"""
Evaluate ablation checkpoints on the validation split and write a combined CSV
alongside the original real and fused models.

Does NOT modify existing results. Outputs go to ef_prediction/ablation_results/:
  - ablation_summary.csv          (one row per model: MAE/MSE/RMSE/R2/n)
  - predictions_<name>.csv        (per-clip True/Pred/Error for each model)

Models evaluated:
  real_original                 checkpoints/real/best.pth                       (real,  val_manifest)
  fused_original                checkpoints/fused/run_1_best.pth                (fused, val_manifest_fused)
  fused_no_hcl                  checkpoints/fused/run_99_best.pth               (fused, val_manifest_fused)
  real_no_hcl                   checkpoints/real_no_hcl/best.pth                (real,  val_manifest)
  synth_only_no_fusion_no_hcl   checkpoints/real_synth_no_fusion_no_hcl/best.pth(real,  synthetic_only val)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import DualVideoEFDataset
from .models.pt_efnet_fused import PTEFNetFused
from .models.pt_efnet_real import PTEFNetReal

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "ef_prediction" / "ablation_results"


def compute_metrics(preds, labels):
    mae = float(np.mean(np.abs(preds - labels)))
    mse = float(np.mean((preds - labels) ** 2))
    rmse = float(np.sqrt(mse))
    denom = float(np.sum((labels - labels.mean()) ** 2))
    r2 = 0.0 if denom == 0 else float(1 - np.sum((preds - labels) ** 2) / denom)
    return mae, mse, rmse, r2


def evaluate_one(spec: dict, cfg: dict, device: torch.device, batch_size: int):
    ckpt = ROOT / spec["checkpoint"]
    if not ckpt.is_file():
        print(f"⚠️  Skipping {spec['name']}: checkpoint not found ({ckpt})")
        return None

    manifest = ROOT / spec["manifest"]
    fused = spec["model_type"] == "fused"

    ds_kwargs = dict(
        manifest_path=str(manifest),
        video_root_dir=str(ROOT / spec["video_root"]),
        video_length=cfg["model"]["video_length"],
        video_size=cfg["model"]["video_size"],
        fused=fused,
    )
    if fused:
        ds_kwargs["synthetic_root_dir"] = str(ROOT / cfg["data"]["synthetic_video_dir"])

    dataset = DualVideoEFDataset(**ds_kwargs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if fused:
        model = PTEFNetFused(**PTEFNetFused.kwargs_from_cfg(cfg)).to(device)
    else:
        model = PTEFNetReal(backbone=cfg["model"].get("backbone", "resnet34")).to(device)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            if fused:
                real_video, syn_video, ef, _, _, _, demo_vec = batch
                real_video = real_video.to(device)
                syn_video = syn_video.to(device)
                demo_vec = demo_vec.to(device).float()
                pred, _ = model(real_video, syn_video, demo_vec)
            else:
                video, ef, _, _, _, demo_vec = batch
                video = video.to(device)
                demo_vec = demo_vec.to(device).float()
                pred, _ = model(video, demo_vec)
            preds.extend(pred.cpu().numpy())
            labels.extend(ef.cpu().numpy())

    preds = np.array(preds) * 100
    labels = np.array(labels) * 100
    mae, mse, rmse, r2 = compute_metrics(preds, labels)

    return {
        "preds": preds,
        "labels": labels,
        "metrics": {
            "Model": spec["name"],
            "HCL": spec["hcl"],
            "Fusion": spec["fusion"],
            "Data": spec["data"],
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "n": int(len(labels)),
            "checkpoint": spec["checkpoint"],
        },
    }


def build_specs(cfg: dict) -> list[dict]:
    val_real = cfg["data"]["val_manifest"]
    val_fused = cfg["data"]["val_manifest_fused"]
    orig_videos = cfg["data"]["original_video_dir"]
    synth_videos = cfg["data"]["synthetic_video_dir"]

    return [
        {
            "name": "real_original", "model_type": "real",
            "checkpoint": "ef_prediction/checkpoints/real/best.pth",
            "manifest": val_real, "video_root": orig_videos,
            "hcl": "yes", "fusion": "no", "data": "real",
        },
        {
            "name": "fused_original", "model_type": "fused",
            "checkpoint": "ef_prediction/checkpoints/fused/run_1_best.pth",
            "manifest": val_fused, "video_root": orig_videos,
            "hcl": "yes", "fusion": "yes", "data": "real+synthetic",
        },
        {
            "name": "fused_no_hcl", "model_type": "fused",
            "checkpoint": "ef_prediction/checkpoints/fused/run_99_best.pth",
            "manifest": val_fused, "video_root": orig_videos,
            "hcl": "no", "fusion": "yes", "data": "real+synthetic",
        },
        {
            "name": "real_no_hcl", "model_type": "real",
            "checkpoint": "ef_prediction/checkpoints/real_no_hcl/best.pth",
            "manifest": val_real, "video_root": orig_videos,
            "hcl": "no", "fusion": "no", "data": "real",
        },
        {
            "name": "synth_only_no_fusion_no_hcl", "model_type": "real",
            "checkpoint": "ef_prediction/checkpoints/real_synth_no_fusion_no_hcl/best.pth",
            "manifest": "ef_prediction/synthetic_only_manifests/val_synthetic_only.csv",
            "video_root": synth_videos,
            "hcl": "no", "fusion": "no", "data": "synthetic",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ablation + original models on validation.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    with open(ROOT / "ef_prediction/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in build_specs(cfg):
        print(f"\n▶ Evaluating {spec['name']} ...")
        result = evaluate_one(spec, cfg, device, args.batch_size)
        if result is None:
            continue
        m = result["metrics"]
        print(f"   MAE={m['MAE']} | MSE={m['MSE']} | RMSE={m['RMSE']} | R2={m['R2']} | n={m['n']}")
        rows.append(m)

        pd.DataFrame({
            "True_EF": result["labels"],
            "Predicted_EF": result["preds"],
            "Error": result["preds"] - result["labels"],
        }).to_csv(out_dir / f"predictions_{spec['name']}.csv", index=False)

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n✓ Wrote summary: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
