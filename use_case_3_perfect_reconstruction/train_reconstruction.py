"""
Train PerfectReconstructionGenerator with L1 reconstruction (identity: output ≈ input).

Use ``--conditioning film`` for FiLM-based demographic modulation (Step 3).
Use ``--conditioning concat`` for legacy concat+1x1 fusion (matches older checkpoints).

Example:
  python train_reconstruction.py --manifest path/to.csv --checkpoint_dir ./ckpt_film \\
      --conditioning film --epochs 50 --video_length 32 --video_size 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import cv2

from models import PerfectReconstructionGenerator


class ManifestVideoDataset(Dataset):
    def __init__(self, manifest_csv: str, video_length: int, video_size: int):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df["processed_path"].notna()].reset_index(drop=True)
        self.video_length = video_length
        self.video_size = video_size
        self.sex_map = {"F": 0, "M": 1, "O": 0}
        self.age_map = {"0-1": 0, "2-5": 1, "6-10": 2, "11-15": 3, "16-18": 4}
        self.bmi_map = {"underweight": 0, "normal": 1, "overweight": 2, "obese": 3}
        if "bmi_category" not in self.df.columns:
            self.df["bmi_category"] = "normal"

    def __len__(self):
        return len(self.df)

    def _load(self, path: str) -> np.ndarray:
        p = Path(path)
        if not p.is_absolute():
            c = Path.cwd() / p
            p = c if c.exists() else Path(__file__).resolve().parent.parent / p
        cap = cv2.VideoCapture(str(p))
        frames = []
        while len(frames) < self.video_length:
            ok, fr = cap.read()
            if not ok:
                break
            if len(fr.shape) == 3:
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            if fr.shape[0] != self.video_size or fr.shape[1] != self.video_size:
                fr = cv2.resize(fr, (self.video_size, self.video_size))
            frames.append(fr)
        cap.release()
        if not frames:
            return np.zeros((self.video_length, self.video_size, self.video_size), dtype=np.uint8)
        v = np.array(frames, dtype=np.float32)
        if len(v) < self.video_length:
            pad = np.zeros((self.video_length - len(v), self.video_size, self.video_size), dtype=v.dtype)
            v = np.concatenate([v, pad], axis=0)
        else:
            v = v[: self.video_length]
        return v / 127.5 - 1.0

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = self._load(row["processed_path"])
        x = torch.from_numpy(x).unsqueeze(0).float()
        sex_col = "Sex" if "Sex" in row else "sex"
        s = self.sex_map.get(row[sex_col], 0)
        a = self.age_map.get(row["age_bin"], 0)
        b = self.bmi_map.get(row.get("bmi_category", "normal"), 1)
        cond = torch.zeros(11)
        cond[s] = 1.0
        cond[2 + a] = 1.0
        cond[7 + b] = 1.0
        return x, cond


def main():
    p = argparse.ArgumentParser(description="Train reconstruction U-Net (L1)")
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints_reconstruction")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--video_length", type=int, default=32)
    p.add_argument("--video_size", type=int, default=64)
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--conditioning", type=str, default="film", choices=["concat", "film"])
    p.add_argument("--lambda_temp", type=float, default=0.1, help="Temporal L1 between pred and input")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true", default=True, help="Use mixed-precision (FP16) training")
    p.add_argument("--no_amp", dest="amp", action="store_false", help="Disable mixed-precision training")
    args = p.parse_args()

    if args.video_size not in (64, 128):
        raise SystemExit("video_size must be 64 or 128 (matches PerfectReconstructionGenerator spatial_size)")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ds = ManifestVideoDataset(args.manifest, args.video_length, args.video_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    num_workers=8, pin_memory=True, persistent_workers=True)

    model = PerfectReconstructionGenerator(
        base_channels=args.base_channels,
        spatial_size=args.video_size,
        conditioning=args.conditioning,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    l1 = nn.L1Loss()
    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    print(f"Mixed-precision (AMP): {'ON' if use_amp else 'OFF'}")

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x, c in tqdm(dl, desc=f"epoch {epoch+1}/{args.epochs}"):
            x = x.to(device)
            c = c.to(device)
            opt.zero_grad()
            with autocast("cuda", enabled=use_amp):
                out = model(x, c)
                loss = l1(out, x)
                if args.lambda_temp > 0 and out.size(2) > 1:
                    dt_out = out[:, :, 1:] - out[:, :, :-1]
                    dt_x = x[:, :, 1:] - x[:, :, :-1]
                    loss = loss + args.lambda_temp * l1(dt_out, dt_x)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        print(f"epoch {epoch+1} mean L1: {float(np.mean(losses)):.5f}")
        torch.save(
            {"generator": model.state_dict(), "conditioning": args.conditioning, "video_size": args.video_size},
            Path(args.checkpoint_dir) / f"recon_epoch_{epoch+1}.pt",
        )
    torch.save(
        {"generator": model.state_dict(), "conditioning": args.conditioning, "video_size": args.video_size},
        Path(args.checkpoint_dir) / "recon_best.pt",
    )
    print("Saved:", args.checkpoint_dir)


if __name__ == "__main__":
    main()
