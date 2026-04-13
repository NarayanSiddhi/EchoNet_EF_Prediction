import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from .dataset import DualVideoEFDataset
from .models.pt_efnet_real import PTEFNetReal
from .losses import hierarchical_loss


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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override config training.num_epochs when set.",
    )
    parser.add_argument(
        "--val_every",
        type=int,
        default=None,
        help="Override training.val_every_n_epochs (validate less often = faster).",
    )
    args = parser.parse_args()

    print("\n🚀 REAL EF training (late fusion + ResNet + cosine LR + hybrid loss)\n")

    with open("ef_prediction/config.yaml") as f:
        cfg = yaml.safe_load(f)

    num_epochs = (
        args.epochs
        if args.epochs is not None
        else cfg["training"]["num_epochs"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tr = cfg["training"]
    backbone = cfg["model"].get("backbone", "resnet34")
    grad_clip = float(tr.get("grad_clip_norm", 1.0))
    eta_min = float(tr.get("eta_min", 1e-6))
    hcl_w = float(tr.get("hcl_weight", 0.04))
    mix_mse = float(tr.get("hybrid_mse_weight", 0.5))
    mix_mse = max(0.0, min(1.0, mix_mse))
    val_every = int(args.val_every if args.val_every is not None else tr.get("val_every_n_epochs", 1))
    val_every = max(1, val_every)

    use_amp = bool(tr.get("use_amp", True)) and device.type == "cuda"
    print(f"AMP: {use_amp} | val every {val_every} epoch(s) | num_workers={tr.get('num_workers', 0)}")

    dl_kw = _dataloader_kwargs(tr, device)

    train_ds = DualVideoEFDataset(
        manifest_path=cfg["data"]["train_manifest"],
        video_root_dir=cfg["data"]["original_video_dir"],
        video_length=cfg["model"]["video_length"],
        video_size=cfg["model"]["video_size"],
        fused=False
    )

    val_ds = DualVideoEFDataset(
        manifest_path=cfg["data"]["val_manifest"],
        video_root_dir=cfg["data"]["original_video_dir"],
        video_length=cfg["model"]["video_length"],
        video_size=cfg["model"]["video_size"],
        fused=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=tr["batch_size"],
        shuffle=True,
        **dl_kw,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=tr["batch_size"],
        shuffle=False,
        **dl_kw,
    )

    model = PTEFNetReal(backbone=backbone).to(device)

    if bool(tr.get("torch_compile", False)) and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            print("torch.compile enabled")
        except Exception as e:
            print("torch.compile skipped:", e)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tr["learning_rate"],
        weight_decay=tr["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=eta_min
    )

    scaler = GradScaler("cuda", enabled=use_amp)

    huber = torch.nn.SmoothL1Loss()
    mse_loss = torch.nn.MSELoss()

    ckpt_dir = Path("ef_prediction/checkpoints/real")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_mse = float("inf")
    best_mae_at_best_mse = float("inf")

    for epoch in range(num_epochs):

        print(f"\n===== Epoch {epoch} =====  lr={scheduler.get_last_lr()[0]:.2e}")

        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc="Training"):

            video, ef, sex, age, bmi, demo_vec = batch

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
                loss_hcl = hierarchical_loss(z, sex, age, bmi)
                loss = loss_ef + hcl_w * loss_hcl

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            train_losses.append(loss.item())

        print(f"Train Loss: {np.mean(train_losses):.4f}")

        run_val = (epoch % val_every == 0) or (epoch == num_epochs - 1)
        if not run_val:
            print("  (validation skipped this epoch)")
            scheduler.step()
            continue

        model.eval()
        preds, labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):

                video, ef, _, _, _, demo_vec = batch

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

        print(f"Val  MAE: {mae:.2f} | MSE: {mse:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")

        if mse < best_mse:
            best_mse = mse
            best_mae_at_best_mse = mae
            m_save = getattr(model, "_orig_mod", model)
            torch.save(m_save.state_dict(), ckpt_dir / "best.pth")
            print(f"✓ Saved best checkpoint (val MSE={mse:.2f}, MAE={mae:.2f})")

        scheduler.step()

    print(f"\n🎉 Done. Best val MSE={best_mse:.2f} (MAE at that epoch: {best_mae_at_best_mse:.2f})\n")


if __name__ == "__main__":
    main()
