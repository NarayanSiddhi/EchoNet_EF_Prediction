"""
Train the real-only EF model on synthetic videos only (no fusion).

This does not modify config.yaml or existing checkpoints. It uses manifests
created by `make_synthetic_only_manifests.py` and writes to a user-specified
checkpoint directory (default: ef_prediction/checkpoints/real_synthetic_only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import DualVideoEFDataset
from .losses import hierarchical_multidemographic_loss
from .models.pt_efnet_real import PTEFNetReal


def compute_metrics(preds, labels):
    mae = np.mean(np.abs(preds - labels))
    mse = np.mean((preds - labels) ** 2)
    rmse = np.sqrt(mse)
    denom = np.sum((labels - labels.mean()) ** 2)
    r2 = 0 if denom == 0 else 1 - np.sum((labels - preds) ** 2) / denom
    return mae, mse, rmse, r2


def _dataloader_kwargs(training_cfg: dict, device: torch.device) -> dict:
    nw = int(training_cfg.get("num_workers", 0))
    pin = bool(training_cfg.get("pin_memory", True)) and device.type == "cuda"
    kw: dict = {"num_workers": nw, "pin_memory": pin}
    if nw > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = int(training_cfg.get("prefetch_factor", 2))
    return kw


def _h2_lambda(epoch, start_epoch, warmup_epochs, lambda_max):
    if epoch < start_epoch:
        return 0.0
    if warmup_epochs <= 0:
        return lambda_max
    progress = (epoch - start_epoch + 1) / float(warmup_epochs)
    return float(lambda_max) * min(1.0, max(0.0, progress))


def main() -> None:
    p = argparse.ArgumentParser(description="Train PTEFNetReal on synthetic-only manifests.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--val_every", type=int, default=None)
    p.add_argument("--hcl-weight", type=float, default=0.0, help="0 = no-HCL (recommended for this ablation).")
    p.add_argument(
        "--train-manifest",
        type=str,
        default="ef_prediction/synthetic_only_manifests/train_synthetic_only.csv",
    )
    p.add_argument(
        "--val-manifest",
        type=str,
        default="ef_prediction/synthetic_only_manifests/val_synthetic_only.csv",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default="ef_prediction/checkpoints/real_synthetic_only",
    )
    args = p.parse_args()

    with open("ef_prediction/config.yaml") as f:
        cfg = yaml.safe_load(f)

    tr = cfg["training"]
    num_epochs = int(args.epochs if args.epochs is not None else tr["num_epochs"])
    val_every = int(args.val_every if args.val_every is not None else tr.get("val_every_n_epochs", 1))
    val_every = max(1, val_every)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(tr.get("use_amp", True)) and device.type == "cuda"

    print("\n🚀 REAL EF training on SYNTHETIC videos only\n")
    print(f"device={device} | AMP={use_amp} | hcl_weight={args.hcl_weight} | epochs={num_epochs} | val_every={val_every}")

    dl_kw = _dataloader_kwargs(tr, device)

    train_ds = DualVideoEFDataset(
        manifest_path=args.train_manifest,
        video_root_dir=cfg["data"]["synthetic_video_dir"],
        video_length=cfg["model"]["video_length"],
        video_size=cfg["model"]["video_size"],
        fused=False,
    )
    val_ds = DualVideoEFDataset(
        manifest_path=args.val_manifest,
        video_root_dir=cfg["data"]["synthetic_video_dir"],
        video_length=cfg["model"]["video_length"],
        video_size=cfg["model"]["video_size"],
        fused=False,
    )

    train_loader = DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=True, **dl_kw)
    val_loader = DataLoader(val_ds, batch_size=tr["batch_size"], shuffle=False, **dl_kw)

    backbone = cfg["model"].get("backbone", "resnet34")
    model = PTEFNetReal(backbone=backbone).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["learning_rate"], weight_decay=tr["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=float(tr.get("eta_min", 1e-6)))
    scaler = GradScaler("cuda", enabled=use_amp)
    huber = torch.nn.SmoothL1Loss()
    mse_loss = torch.nn.MSELoss()

    # HCL parameters (only used if hcl_weight > 0)
    h2_start_epoch = int(tr.get("hcl_h2_start_epoch", 20))
    h2_warmup_epochs = int(tr.get("hcl_h2_warmup_epochs", 20))
    h2_lambda_max = float(tr.get("hcl_h2_lambda_max", 1.0))
    hcl_temp_h1 = float(tr.get("hcl_temp_h1", 0.2))
    hcl_temp_h2 = float(tr.get("hcl_temp_h2", 0.2))
    hcl_alpha_intra = float(tr.get("hcl_alpha_intra", 0.5))
    hcl_alpha_inter = float(tr.get("hcl_alpha_inter", 1.0))
    ef_task_threshold = float(tr.get("hcl_ef_task_threshold", 0.5))
    mix_mse = float(tr.get("hybrid_mse_weight", 0.5))
    mix_mse = max(0.0, min(1.0, mix_mse))

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pth"

    best_mse = float("inf")
    best_mae = float("inf")

    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch} ===== lr={scheduler.get_last_lr()[0]:.2e}")
        lambda_h2_epoch = _h2_lambda(epoch, h2_start_epoch, h2_warmup_epochs, h2_lambda_max)

        model.train()
        train_losses = []

        for video, ef, sex, age, bmi, demo_vec in tqdm(train_loader, desc="Training"):
            video = video.to(device, non_blocking=True)
            ef = ef.to(device, non_blocking=True).float()
            sex = sex.to(device, non_blocking=True)
            age = age.to(device, non_blocking=True)
            bmi = bmi.to(device, non_blocking=True)
            demo_vec = demo_vec.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                ef_pred, z = model(video, demo_vec)
                loss_ef = (1.0 - mix_mse) * huber(ef_pred, ef) + mix_mse * mse_loss(ef_pred, ef)

                if args.hcl_weight > 0:
                    task_labels = (ef >= ef_task_threshold).long()
                    loss_hcl, _, _ = hierarchical_multidemographic_loss(
                        z=z,
                        task_labels=task_labels,
                        sex=sex,
                        age=age,
                        bmi=bmi,
                        lambda_h2=lambda_h2_epoch,
                        temp_h1=hcl_temp_h1,
                        temp_h2=hcl_temp_h2,
                        alpha_intra=hcl_alpha_intra,
                        alpha_inter=hcl_alpha_inter,
                    )
                    loss = loss_ef + float(args.hcl_weight) * loss_hcl
                else:
                    loss = loss_ef

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_losses.append(float(loss.detach().cpu().item()))

        print(f"Train loss: {np.mean(train_losses):.4f}")

        run_val = (epoch % val_every == 0) or (epoch == num_epochs - 1)
        if not run_val:
            scheduler.step()
            continue

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for video, ef, _, _, _, demo_vec in tqdm(val_loader, desc="Validation"):
                video = video.to(device, non_blocking=True)
                ef = ef.to(device, non_blocking=True).float()
                demo_vec = demo_vec.to(device, non_blocking=True).float()
                with autocast("cuda", enabled=use_amp):
                    pred, _ = model(video, demo_vec)
                preds.append(pred.float().cpu().numpy())
                labels.append(ef.cpu().numpy())

        preds = np.concatenate(preds) * 100
        labels = np.concatenate(labels) * 100
        mae, mse, rmse, r2 = compute_metrics(preds, labels)
        print(f"Val MAE={mae:.2f} | MSE={mse:.2f} | RMSE={rmse:.2f} | R2={r2:.4f}")

        if mse < best_mse:
            best_mse = float(mse)
            best_mae = float(mae)
            torch.save(model.state_dict(), best_path)
            print(f"✓ Saved best checkpoint: {best_path} (val MSE={best_mse:.2f}, MAE={best_mae:.2f})")

        scheduler.step()

    print(f"\nDone. Best val MSE={best_mse:.2f} (MAE={best_mae:.2f}) | ckpt={best_path}\n")


if __name__ == "__main__":
    main()

