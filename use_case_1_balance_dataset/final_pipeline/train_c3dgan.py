"""
Conditional 3D DCGAN for pediatric echo clips (sex, age bin, BMI).

Training resolution (--size) must match preprocessed frame size (preprocessing width/height).
See preprocessing/config.yaml and final_pipeline/README.md.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import cv2

from c3dgan_arch import Discriminator, Generator, weights_init

# Generator/Discriminator use a fixed temporal depth of 32 frames (see c3dgan_arch.py).
DCGAN_VIDEO_LENGTH = 32


class EchoDataset(Dataset):
    """Loads preprocessed MP4 or .npy tensors; clips/pads to video_length frames."""

    def __init__(self, manifest_csv: str, video_length: int = 32, video_size: int = 128):
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df["processed_path"].notna()].reset_index(drop=True)
        self.video_length = video_length
        self.video_size = video_size

        self.sex_map = {"F": 0, "M": 1, "O": 0}
        self.age_map = {"0-1": 0, "2-5": 1, "6-10": 2, "11-15": 3, "16-18": 4}
        self.bmi_map = {"underweight": 0, "normal": 1, "overweight": 2, "obese": 3}

        if "bmi_category" not in self.df.columns:
            self._compute_bmi_categories()

        print(f"Dataset: {len(self.df)} videos loaded (T={video_length}, H=W={video_size})")

    def _compute_bmi_categories(self) -> None:
        def categorize_bmi(row):
            if pd.isna(row.get("weight")) or pd.isna(row.get("height")) or row["height"] <= 0:
                return "normal"
            bmi = row["weight"] / ((row["height"] / 100.0) ** 2)
            if bmi < 18.5:
                return "underweight"
            if bmi < 25:
                return "normal"
            if bmi < 30:
                return "overweight"
            return "obese"

        self.df["bmi_category"] = self.df.apply(categorize_bmi, axis=1)

    def _load_video(self, video_path):
        video_path = Path(video_path)
        if not video_path.is_absolute():
            cand = Path.cwd() / video_path
            if cand.exists():
                video_path = cand
            else:
                alt = Path(__file__).resolve().parent.parent / video_path
                video_path = alt if alt.exists() else cand

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if video_path.suffix == ".npy":
            video = np.load(video_path)
        else:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            frames = []
            while len(frames) < self.video_length:
                ret, frame = cap.read()
                if not ret:
                    break
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if frame.shape[0] != self.video_size or frame.shape[1] != self.video_size:
                    frame = cv2.resize(frame, (self.video_size, self.video_size))
                frames.append(frame)
            cap.release()

            if len(frames) == 0:
                video = np.zeros(
                    (self.video_length, self.video_size, self.video_size), dtype=np.uint8
                )
            else:
                video = np.array(frames)
                if len(video) < self.video_length:
                    pad = np.zeros(
                        (self.video_length - len(video), self.video_size, self.video_size),
                        dtype=video.dtype,
                    )
                    video = np.concatenate([video, pad], axis=0)
                elif len(video) > self.video_length:
                    video = video[: self.video_length]

        # If numpy loaded with wrong shape, best-effort fix
        if video.ndim == 3 and video.shape[0] != self.video_length:
            if len(video) < self.video_length:
                pad = np.zeros(
                    (self.video_length - len(video), self.video_size, self.video_size),
                    dtype=video.dtype,
                )
                video = np.concatenate([video, pad], axis=0)
            else:
                video = video[: self.video_length]
        if video.shape[-2] != self.video_size or video.shape[-1] != self.video_size:
            raise ValueError(
                f"Frame size {video.shape[-2:]} != ({self.video_size},{self.video_size}); "
                "re-run preprocessing or pass matching --size."
            )

        return video

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            video = self._load_video(row["processed_path"])
        except Exception as e:
            print(f"Warning: Failed to load {row['processed_path']}: {e}")
            video = np.zeros(
                (self.video_length, self.video_size, self.video_size), dtype=np.uint8
            )

        video = video.astype(np.float32) / 127.5 - 1.0
        x = torch.from_numpy(video).unsqueeze(0)

        sex_col = "Sex" if "Sex" in row else "sex"
        sex = self.sex_map.get(row[sex_col], 0)
        age = self.age_map.get(row["age_bin"], 0)
        bmi = self.bmi_map.get(row.get("bmi_category", "normal"), 1)

        sex_onehot = torch.zeros(2)
        sex_onehot[sex] = 1
        age_onehot = torch.zeros(5)
        age_onehot[age] = 1
        bmi_onehot = torch.zeros(4)
        bmi_onehot[bmi] = 1

        cond = torch.cat([sex_onehot, age_onehot, bmi_onehot])
        return x, cond


def train(cfg: dict) -> None:
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tlen = int(cfg.get("video_length", DCGAN_VIDEO_LENGTH))
    if tlen != DCGAN_VIDEO_LENGTH:
        raise ValueError(
            f"video_length must be {DCGAN_VIDEO_LENGTH} (fixed by C3D DCGAN architecture); got {tlen}"
        )
    dataset = EchoDataset(cfg["manifest"], video_length=tlen, video_size=cfg["size"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=(device == "cuda"),
    )
    print(f"Batches per epoch: {len(loader)}")

    netG = Generator(cfg["z_dim"], cfg["cond_dim"], cfg["size"]).to(device)
    netD = Discriminator(cfg["cond_dim"], cfg["size"]).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    print(f"Generator parameters: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in netD.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizerD = optim.Adam(netD.parameters(), lr=cfg["lr_d"], betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=cfg["lr_g"], betas=(0.5, 0.999))

    real_label = 1.0
    fake_label = 0.0

    lt = float(cfg.get("lambda_temp", 0.0))
    print(
        f"\nEpochs: {cfg['epochs']} | batch_size: {cfg['batch_size']} | "
        f"resolution: {cfg['size']}x{cfg['size']} x T={DCGAN_VIDEO_LENGTH} | "
        f"lr_g: {cfg['lr_g']} | lr_d: {cfg['lr_d']} | lambda_temp: {lt}\n"
    )

    G_losses: list[float] = []
    D_losses: list[float] = []

    for epoch in range(cfg["epochs"]):
        netG.train()
        netD.train()
        epoch_G: list[float] = []
        epoch_D: list[float] = []

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg['epochs']}")
        for _, (real_videos, conditions) in enumerate(pbar):
            real_videos = real_videos.to(device)
            conditions = conditions.to(device)
            b_size = real_videos.size(0)

            netD.zero_grad()
            label = torch.full((b_size, 1), real_label, dtype=torch.float, device=device)
            output = netD(real_videos, conditions)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = torch.sigmoid(output).mean().item()

            noise = torch.randn(b_size, cfg["z_dim"], device=device)
            fake = netG(noise, conditions)
            label.fill_(fake_label)
            output = netD(fake.detach(), conditions)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = torch.sigmoid(output).mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake, conditions)
            errG = criterion(output, label)
            lambda_temp = float(cfg.get("lambda_temp", 0.0))
            if lambda_temp > 0.0:
                # Temporal smoothness on generated clips (reduces frame-to-frame flicker)
                if fake.size(2) > 1:
                    df = fake[:, :, 1:, :, :] - fake[:, :, :-1, :, :]
                    errG = errG + lambda_temp * df.abs().mean()
            errG.backward()
            D_G_z2 = torch.sigmoid(output).mean().item()
            optimizerG.step()

            epoch_D.append(errD.item())
            epoch_G.append(errG.item())
            pbar.set_postfix(
                D_loss=f"{errD.item():.3f}",
                G_loss=f"{errG.item():.3f}",
                D_x=f"{D_x:.3f}",
                D_G_z=f"{D_G_z2:.3f}",
            )

        avg_D = float(np.mean(epoch_D))
        avg_G = float(np.mean(epoch_G))
        D_losses.append(avg_D)
        G_losses.append(avg_G)
        print(f"[Epoch {epoch + 1}] D_loss: {avg_D:.4f} | G_loss: {avg_G:.4f}")

        os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
        torch.save(netG.state_dict(), f"{cfg['checkpoint_dir']}/generator_epoch{epoch}.pt")
        torch.save(netD.state_dict(), f"{cfg['checkpoint_dir']}/discriminator_epoch{epoch}.pt")

        prev = G_losses[:-1]
        if epoch == 0 or not prev or avg_G < min(prev):
            torch.save(netG.state_dict(), f"{cfg['checkpoint_dir']}/generator_best.pt")
            print(f"Saved best generator (epoch {epoch + 1})")

    print("Training complete.")
    print(f"Best G_loss: {min(G_losses):.4f} | checkpoints: {cfg['checkpoint_dir']}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train conditional C3D DCGAN")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/processed_full/manifest_full.csv",
        help="CSV with processed_path, sex/Sex, age_bin, weight, height",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        choices=[32, 64, 128, 256],
        help="Frame H=W; must match preprocessed videos (see preprocessing/config.yaml).",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr_g", type=float, default=0.0002)
    parser.add_argument("--lr_d", type=float, default=0.0002)
    parser.add_argument("--z_dim", type=int, default=128)
    parser.add_argument("--cond_dim", type=int, default=11)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_c3dgan")
    parser.add_argument(
        "--video_length",
        type=int,
        default=DCGAN_VIDEO_LENGTH,
        help=f"Must stay {DCGAN_VIDEO_LENGTH} (architecture-fixed temporal depth).",
    )
    parser.add_argument(
        "--lambda_temp",
        type=float,
        default=0.0,
        help="If >0, add temporal smoothness penalty on generator (frame-difference L1).",
    )
    args = parser.parse_args()
    train(vars(args))
