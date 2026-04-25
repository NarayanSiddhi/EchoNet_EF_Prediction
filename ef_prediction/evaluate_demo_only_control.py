"""
Control experiment: run the *trained* EF head with the **video summary zeroed**
so `head` input is `concat(zeros_like(pooled), demo_emb)`.

This answers: "If the video vector carried no signal (zeros), what would EF
predictions look like using only the demographic path through the existing
head?" It is **not** the same as retraining a demographics-only model.

Run from repo root:
  python ef_prediction/evaluate_demo_only_control.py --model fused
  python ef_prediction/evaluate_demo_only_control.py --model real
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ef_prediction.dataset import DualVideoEFDataset as RealVideoEFDataset
from ef_prediction.dataset_demographics import DualVideoEFDataset
from ef_prediction.models.pt_efnet_fused import PTEFNetFused
from ef_prediction.models.pt_efnet_real import PTEFNetReal


def compute_metrics(preds, labels):
    mae = np.mean(np.abs(preds - labels))
    mse = np.mean((preds - labels) ** 2)
    rmse = np.sqrt(mse)
    denom = np.sum((labels - labels.mean()) ** 2)
    r2 = 0.0 if denom == 0 else 1 - np.sum((labels - preds) ** 2) / denom
    return mae, mse, rmse, r2


def predict_fused_zero_pooled(model, real_video, syn_video, demo_vec):
    real_feats = model.extract_features(real_video)
    syn_feats = model.extract_features(syn_video)
    cat = torch.cat([real_feats, syn_feats], dim=2)
    if model.fusion_mode == "gated":
        g = model.gate_net(cat)
        fused_frames = g * real_feats + (1.0 - g) * syn_feats
    else:
        fused_frames = model.fusion_proj(cat)
    lstm_out, _ = model.lstm(fused_frames)
    pooled = model.attn(lstm_out)
    pooled = torch.zeros_like(pooled)
    demo_emb = model.demo_encoder(demo_vec.float())
    h = torch.cat([pooled, demo_emb], dim=1)
    return model.head(h).squeeze(1)


def predict_real_zero_pooled(model, video, demo_vec):
    b, c, t, h, w = video.shape
    x = video
    if c == 1:
        x = x.repeat(1, 3, 1, 1, 1)
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, 3, h, w)
    feats = model.cnn(x).squeeze(-1).squeeze(-1).view(b, t, 512)
    lstm_out, _ = model.lstm(feats)
    pooled = model.attn(lstm_out)
    pooled = torch.zeros_like(pooled)
    demo_emb = model.demo_encoder(demo_vec.float())
    fused = torch.cat([pooled, demo_emb], dim=1)
    return model.head(fused).squeeze(1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["fused", "real"], default="fused")
    p.add_argument("--checkpoint", default=None)
    p.add_argument(
        "--manifest",
        default=None,
        help="CSV manifest (default: val fused or val real from config).",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--output-dir",
        default="ef_prediction/eval_results",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.checkpoint is None:
        args.checkpoint = (
            "ef_prediction/checkpoints/fused/run_1_best.pth"
            if args.model == "fused"
            else "ef_prediction/checkpoints/real/best.pth"
        )

    with open("ef_prediction/config.yaml") as f:
        cfg = yaml.safe_load(f)

    if args.manifest:
        manifest = Path(args.manifest)
    elif args.model == "fused":
        manifest = ROOT / cfg["data"]["val_manifest_fused"]
    else:
        manifest = ROOT / cfg["data"]["val_manifest"]
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "fused":
        model = PTEFNetFused(**PTEFNetFused.kwargs_from_cfg(cfg)).to(device)
        ds = DualVideoEFDataset(
            manifest_path=str(manifest),
            video_root_dir=cfg["data"]["original_video_dir"],
            synthetic_root_dir=cfg["data"]["synthetic_video_dir"],
            video_length=cfg["model"]["video_length"],
            video_size=cfg["model"]["video_size"],
            fused=True,
        )
        forward_fn = predict_fused_zero_pooled
    else:
        model = PTEFNetReal(
            backbone=cfg["model"].get("backbone", "resnet34")
        ).to(device)
        ds = RealVideoEFDataset(
            manifest_path=str(manifest),
            video_root_dir=cfg["data"]["original_video_dir"],
            video_length=cfg["model"]["video_length"],
            video_size=cfg["model"]["video_size"],
            fused=False,
        )
        forward_fn = predict_real_zero_pooled

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    preds, labels = [], []

    with torch.no_grad():
        if args.model == "fused":
            for batch in tqdm(loader, desc="zero-pooled fused"):
                rv, sv, ef, _, _, _, dv = batch
                rv = rv.to(device)
                sv = sv.to(device)
                ef = ef.to(device)
                dv = dv.to(device).float()
                pred = forward_fn(model, rv, sv, dv)
                preds.extend(pred.cpu().numpy())
                labels.extend(ef.cpu().numpy())
        else:
            for batch in tqdm(loader, desc="zero-pooled real"):
                vid, ef, _, _, _, dv = batch
                vid = vid.to(device)
                ef = ef.to(device)
                dv = dv.to(device).float()
                pred = forward_fn(model, vid, dv)
                preds.extend(pred.cpu().numpy())
                labels.extend(ef.cpu().numpy())

    preds = np.array(preds) * 100.0
    labels = np.array(labels) * 100.0
    mae, mse, rmse, r2 = compute_metrics(preds, labels)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"demo_only_control_{args.model}"
    metrics = {
        "description": "Video pooled vector zeroed before head; demo_emb unchanged.",
        "model": args.model,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(manifest.resolve()),
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "n": int(len(labels)),
    }
    with open(out_dir / f"{tag}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({"True_EF": labels, "Predicted_EF": preds}).to_csv(
        out_dir / f"{tag}_predictions.csv", index=False
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
