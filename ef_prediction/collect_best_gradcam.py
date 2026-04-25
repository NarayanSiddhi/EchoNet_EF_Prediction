"""
Copy the K best GradCAM overlays per demographic leaf folder into gradcam_results/best/.

"Best" = lowest absolute EF prediction error on the val manifest (same splits as training).

Run from repo root:
  python -m ef_prediction.collect_best_gradcam --top 5
  python -m ef_prediction.collect_best_gradcam --top 5 --source-dir ef_prediction/gradcam_results/patients
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset_demographics import DualVideoEFDataset
from .generate_patient_gradcam import _patient_stem_from_row
from .models.pt_efnet_fused import PTEFNetFused
from .models.pt_efnet_real import PTEFNetReal


def _infer_fused_errors(
    cfg: dict,
    device: torch.device,
    checkpoint: Path,
    manifest: Path,
    batch_size: int,
    num_workers: int,
) -> dict[str, float]:
    ds = DualVideoEFDataset(
        manifest_path=str(manifest),
        video_root_dir=cfg["data"]["original_video_dir"],
        synthetic_root_dir=cfg["data"]["synthetic_video_dir"],
        video_length=cfg["model"]["video_length"],
        video_size=cfg["model"]["video_size"],
        fused=True,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(cfg.get("training", {}).get("pin_memory", True)),
    )
    model = PTEFNetFused(**PTEFNetFused.kwargs_from_cfg(cfg)).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    stem_to_err: dict[str, float] = {}
    idx = 0
    with torch.no_grad():
        for real_v, syn_v, ef, _, _, _, demo in tqdm(loader, desc="Score fused (|pred-EF|)"):
            real_v = real_v.to(device)
            syn_v = syn_v.to(device)
            ef = ef.to(device)
            demo = demo.to(device).float()
            pred, _ = model(real_v, syn_v, demo)
            err = (pred.squeeze() - ef).abs().cpu()
            b = real_v.size(0)
            for j in range(b):
                row = ds.df.iloc[idx + j]
                stem = _patient_stem_from_row(row)
                stem_to_err[stem] = float(err[j].item())
            idx += b
    return stem_to_err


def _infer_real_errors(
    cfg: dict,
    device: torch.device,
    checkpoint: Path,
    manifest: Path,
    batch_size: int,
    num_workers: int,
) -> dict[str, float]:
    ds = DualVideoEFDataset(
        manifest_path=str(manifest),
        video_root_dir=cfg["data"]["original_video_dir"],
        synthetic_root_dir=None,
        video_length=cfg["model"]["video_length"],
        video_size=cfg["model"]["video_size"],
        fused=False,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(cfg.get("training", {}).get("pin_memory", True)),
    )
    backbone = cfg["model"].get("backbone", "resnet34")
    model = PTEFNetReal(backbone=backbone).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    stem_to_err: dict[str, float] = {}
    idx = 0
    with torch.no_grad():
        for vid, ef, _, _, _, demo in tqdm(loader, desc="Score real (|pred-EF|)"):
            vid = vid.to(device)
            ef = ef.to(device)
            demo = demo.to(device).float()
            pred, _ = model(vid, demo)
            err = (pred.squeeze() - ef).abs().cpu()
            b = vid.size(0)
            for j in range(b):
                row = ds.df.iloc[idx + j]
                stem = _patient_stem_from_row(row)
                stem_to_err[stem] = float(err[j].item())
            idx += b
    return stem_to_err


def _collect_leaf_groups(mode_root: Path) -> list[tuple[Path, str, str]]:
    """List (leaf_dir, tree_name, group_name) e.g. (.../sex/female, sex, female)."""
    out = []
    for tree in ("sex", "age", "bmi"):
        tdir = mode_root / tree
        if not tdir.is_dir():
            continue
        for sub in sorted(tdir.iterdir()):
            if sub.is_dir():
                out.append((sub, tree, sub.name))
    return out


def _copy_top_k_per_leaf(
    source_mode_root: Path,
    dest_mode_root: Path,
    stem_to_err: dict[str, float],
    top_k: int,
) -> int:
    copied = 0
    for leaf, tree, group in _collect_leaf_groups(source_mode_root):
        stems = sorted({p.stem for p in leaf.glob("*.png")})
        if not stems:
            continue
        ranked = sorted(stems, key=lambda s: stem_to_err.get(s, 1e9))[:top_k]
        for stem in ranked:
            src = leaf / f"{stem}.png"
            if not src.is_file():
                continue
            dst = dest_mode_root / tree / group / f"{stem}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    return copied


def main() -> None:
    p = argparse.ArgumentParser(description="Copy top-K GradCAM PNGs per group by lowest |pred-EF|.")
    p.add_argument("--config", type=str, default="ef_prediction/config.yaml")
    p.add_argument(
        "--source-dir",
        type=str,
        default="ef_prediction/gradcam_results/patients",
        help="Folder containing fused/ and real/ with sex/, age/, bmi/ trees.",
    )
    p.add_argument(
        "--dest-dir",
        type=str,
        default="ef_prediction/gradcam_results/best",
        help="Output root: best/fused/... and best/real/...",
    )
    p.add_argument("--top", type=int, default=5, help="Number of images to keep per leaf folder.")
    p.add_argument("--mode", choices=("fused", "real", "both"), default="both")
    p.add_argument("--run", type=int, default=1, help="Fused checkpoint run id.")
    p.add_argument("--manifest-fused", type=str, default=None)
    p.add_argument("--manifest-real", type=str, default=None)
    p.add_argument("--checkpoint-fused", type=str, default=None)
    p.add_argument("--checkpoint-real", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None, help="Default: cfg training.batch_size")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    bs = args.batch_size or int(cfg.get("training", {}).get("batch_size", 16))
    nw = int(cfg.get("training", {}).get("num_workers", 4))

    source = Path(args.source_dir)
    dest = Path(args.dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    total_copied = 0
    if args.mode in ("fused", "both"):
        fused_dest = dest / "fused"
        if fused_dest.exists():
            shutil.rmtree(fused_dest)
        manifest = Path(args.manifest_fused or cfg["data"]["val_manifest_fused"])
        ckpt = Path(args.checkpoint_fused or f"ef_prediction/checkpoints/fused/run_{args.run}_best.pth")
        err = _infer_fused_errors(cfg, device, ckpt, manifest, bs, nw)
        n = _copy_top_k_per_leaf(source / "fused", fused_dest, err, args.top)
        total_copied += n
        print(f"Fused: copied {n} files into {dest / 'fused'}")

    if args.mode in ("real", "both"):
        real_dest = dest / "real"
        if real_dest.exists():
            shutil.rmtree(real_dest)
        manifest = Path(args.manifest_real or cfg["data"]["val_manifest"])
        ckpt = Path(args.checkpoint_real or "ef_prediction/checkpoints/real/best.pth")
        err = _infer_real_errors(cfg, device, ckpt, manifest, bs, nw)
        n = _copy_top_k_per_leaf(source / "real", real_dest, err, args.top)
        total_copied += n
        print(f"Real: copied {n} files into {dest / 'real'}")

    print(f"Done. Total copies: {total_copied}. Destination: {dest.resolve()}")


if __name__ == "__main__":
    main()
