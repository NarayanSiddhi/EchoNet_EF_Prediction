"""
Generate GradCAM overlay PNGs for each row in a manifest (per study / clip).

Each GradCAM is saved under three trees (same filename, same heatmap; caption lists full Sex / Age / BMI):
  fused/sex/{male,female,unknown}/, fused/age/{0-1,2-5,...}/, fused/bmi/{underweight,normal,...}/
  PNGs are video_size wide with a text strip below (taller than square). For 14-D demographics only,
  age folders use age_bin_0 ... age_bin_7 instead of named year ranges.

Run from repository root:
  python -m ef_prediction.generate_patient_gradcam --mode fused --limit 20
  python -m ef_prediction.generate_patient_gradcam --mode real
  python -m ef_prediction.generate_patient_gradcam --mode both --run 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from .dataset_demographics import DualVideoEFDataset
from .demographics_utils import indices_from_demo_vector, row_to_demo_vector
from .gradcam_fused import GradCAM as GradCAMFused
from .gradcam_real import GradCAM as GradCAMReal
from .models.pt_efnet_fused import PTEFNetFused
from .models.pt_efnet_real import PTEFNetReal


def _patient_stem_from_row(row: pd.Series) -> str:
    if "original_path" in row.index and pd.notna(row.get("original_path", np.nan)):
        p = row["original_path"]
    else:
        p = row.get("processed_path", "")
    if pd.notna(p) and str(p).strip():
        return Path(str(p)).stem
    if "original_id" in row.index and pd.notna(row.get("original_id", np.nan)):
        return str(int(row["original_id"]))
    return "unknown"


# 11-D layout: sex(2) + age(5) + bmi(4), same as row_to_demo_vector / training.
_AGE_FOLDER_11 = {0: "0-1", 1: "2-5", 2: "6-10", 3: "11-15", 4: "16-18"}
_BMI_FOLDER = {0: "underweight", 1: "normal", 2: "overweight", 3: "obese"}


def _demographic_subfolders_from_row(row: pd.Series) -> tuple[str, str, str]:
    """Return (sex_dir, age_dir, bmi_dir) for filesystem paths under sex/, age/, bmi/."""
    demo = row_to_demo_vector(row)
    s_i, a_i, b_i = indices_from_demo_vector(demo)
    d = np.asarray(demo, dtype=float).ravel()

    if s_i == 1:
        sex_dir = "male"
    elif s_i == 0:
        sex_dir = "female"
    else:
        sex_dir = "unknown"

    if d.size >= 14:
        age_dir = f"age_bin_{a_i}"
    else:
        age_dir = _AGE_FOLDER_11.get(a_i, "unknown")

    bmi_dir = _BMI_FOLDER.get(b_i, "unknown")
    return sex_dir, age_dir, bmi_dir


def _output_paths_for_stem(out_sub: Path, stem: str, sex_d: str, age_d: str, bmi_d: str) -> list[Path]:
    name = f"{stem}.png"
    return [
        out_sub / "sex" / sex_d / name,
        out_sub / "age" / age_d / name,
        out_sub / "bmi" / bmi_d / name,
    ]


def _pretty_demo_labels(sex_d: str, age_d: str, bmi_d: str) -> tuple[str, str, str]:
    """Human-readable strings for on-image caption (matches folder semantics)."""
    sex_show = {"male": "Male", "female": "Female", "unknown": "Unknown"}.get(sex_d, sex_d.replace("_", " ").title())
    if age_d.startswith("age_bin_"):
        idx = age_d.replace("age_bin_", "")
        age_show = f"14-D bin {idx}"
    else:
        age_show = age_d
    bmi_show = bmi_d.replace("_", " ").title()
    return sex_show, age_show, bmi_show


def _draw_caption_strip(
    overlay_bgr: np.ndarray,
    sex_d: str,
    age_d: str,
    bmi_d: str,
) -> np.ndarray:
    """Stack overlay (H,W,3) above a light strip with Sex / Age / BMI lines."""
    h, w = overlay_bgr.shape[:2]
    sex_show, age_show, bmi_show = _pretty_demo_labels(sex_d, age_d, bmi_d)
    lines = [f"Sex: {sex_show}", f"Age: {age_show}", f"BMI: {bmi_show}"]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = float(max(0.38, min(0.85, w / 220.0)))
    thickness = max(1, int(round(font_scale)))
    line_height = int(22 * font_scale + 10)
    margin_x = max(6, w // 64)
    strip_h = int(len(lines) * line_height + margin_x * 2)

    canvas = np.full((h + strip_h, w, 3), 245, dtype=np.uint8)
    canvas[:h, :w] = overlay_bgr

    y = h + margin_x + int(18 * font_scale)
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        fs = font_scale
        while tw > w - 2 * margin_x and fs > 0.28:
            fs *= 0.92
            (tw, th), _ = cv2.getTextSize(line, font, fs, thickness)
        for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cv2.putText(
                canvas,
                line,
                (margin_x + ox, y + oy),
                font,
                fs,
                (255, 255, 255),
                thickness + 1,
                cv2.LINE_AA,
            )
        cv2.putText(canvas, line, (margin_x, y), font, fs, (30, 30, 30), thickness, cv2.LINE_AA)
        y += line_height
    return canvas


def _save_overlay(
    cam: np.ndarray,
    frame_gray: np.ndarray,
    out_path: Path,
    video_size: int,
    sex_d: str,
    age_d: str,
    bmi_d: str,
) -> None:
    cam = cv2.resize(cam, (video_size, video_size))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    frame_rgb = np.stack([frame_gray] * 3, axis=-1)
    frame_rgb = (frame_rgb * 255).astype(np.uint8)
    overlay = cv2.addWeighted(frame_rgb, 0.5, heatmap, 0.5, 0)
    out = _draw_caption_strip(overlay, sex_d, age_d, bmi_d)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)


def _save_overlay_to_all_demo_groups(
    out_sub: Path,
    stem: str,
    sex_d: str,
    age_d: str,
    bmi_d: str,
    cam: np.ndarray,
    frame_gray: np.ndarray,
    video_size: int,
) -> None:
    for path in _output_paths_for_stem(out_sub, stem, sex_d, age_d, bmi_d):
        _save_overlay(cam, frame_gray, path, video_size, sex_d, age_d, bmi_d)


def run_fused(
    cfg: dict,
    device: torch.device,
    checkpoint: Path,
    manifest: Path,
    output_dir: Path,
    limit: int | None,
    skip_existing: bool,
) -> None:
    dataset = DualVideoEFDataset(
        manifest_path=str(manifest),
        video_root_dir=cfg["data"]["original_video_dir"],
        synthetic_root_dir=cfg["data"]["synthetic_video_dir"],
        video_length=cfg["model"]["video_length"],
        video_size=cfg["model"]["video_size"],
        fused=True,
    )
    video_size = int(cfg["model"]["video_size"])

    model = PTEFNetFused(**PTEFNetFused.kwargs_from_cfg(cfg)).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.train()

    target_layer = model.cnn[7]
    gradcam = GradCAMFused(model, target_layer)

    n = len(dataset) if limit is None else min(len(dataset), limit)
    out_sub = output_dir / "fused"

    for i in tqdm(range(n), desc="GradCAM fused"):
        row = dataset.df.iloc[i]
        stem = _patient_stem_from_row(row)
        sex_d, age_d, bmi_d = _demographic_subfolders_from_row(row)
        dest_paths = _output_paths_for_stem(out_sub, stem, sex_d, age_d, bmi_d)
        if skip_existing and all(p.is_file() for p in dest_paths):
            continue

        real_video, syn_video, _, _, _, _, demo_vec = dataset[i]
        real_video = real_video.unsqueeze(0).to(device)
        syn_video = syn_video.unsqueeze(0).to(device)
        demo_vec = demo_vec.unsqueeze(0).to(device).float()

        cam = gradcam.generate(real_video, syn_video, demo_vec)
        t = real_video.size(2) // 2
        frame = real_video[0, 0, t].detach().cpu().numpy()
        _save_overlay_to_all_demo_groups(out_sub, stem, sex_d, age_d, bmi_d, cam, frame, video_size)


def run_real(
    cfg: dict,
    device: torch.device,
    checkpoint: Path,
    manifest: Path,
    output_dir: Path,
    limit: int | None,
    skip_existing: bool,
) -> None:
    dataset = DualVideoEFDataset(
        manifest_path=str(manifest),
        video_root_dir=cfg["data"]["original_video_dir"],
        synthetic_root_dir=None,
        video_length=cfg["model"]["video_length"],
        video_size=cfg["model"]["video_size"],
        fused=False,
    )
    video_size = int(cfg["model"]["video_size"])

    backbone = cfg["model"].get("backbone", "resnet34")
    model = PTEFNetReal(backbone=backbone).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.train()

    target_layer = model.cnn[7]
    gradcam = GradCAMReal(model, target_layer)

    n = len(dataset) if limit is None else min(len(dataset), limit)
    out_sub = output_dir / "real"

    for i in tqdm(range(n), desc="GradCAM real"):
        row = dataset.df.iloc[i]
        stem = _patient_stem_from_row(row)
        sex_d, age_d, bmi_d = _demographic_subfolders_from_row(row)
        dest_paths = _output_paths_for_stem(out_sub, stem, sex_d, age_d, bmi_d)
        if skip_existing and all(p.is_file() for p in dest_paths):
            continue

        video, _, _, _, _, demo_vec = dataset[i]
        video = video.unsqueeze(0).to(device)
        demo_vec = demo_vec.unsqueeze(0).to(device).float()

        cam = gradcam.generate(video, demo_vec)
        t = video.size(2) // 2
        frame = video[0, 0, t].detach().cpu().numpy()
        _save_overlay_to_all_demo_groups(out_sub, stem, sex_d, age_d, bmi_d, cam, frame, video_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-patient GradCAM overlays from manifest rows.")
    parser.add_argument(
        "--config",
        type=str,
        default="ef_prediction/config.yaml",
        help="Path to config YAML (cwd: repo root).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("fused", "real", "both"),
        default="fused",
        help="Which model(s) to run.",
    )
    parser.add_argument(
        "--manifest-fused",
        type=str,
        default=None,
        help="Override fused manifest (default: cfg data.val_manifest_fused).",
    )
    parser.add_argument(
        "--manifest-real",
        type=str,
        default=None,
        help="Override real-only manifest (default: cfg data.val_manifest).",
    )
    parser.add_argument(
        "--checkpoint-fused",
        type=str,
        default=None,
        help="Override fused weights (default: ef_prediction/checkpoints/fused/run_{run}_best.pth).",
    )
    parser.add_argument(
        "--checkpoint-real",
        type=str,
        default=None,
        help="Override real weights (default: ef_prediction/checkpoints/real/best.pth).",
    )
    parser.add_argument("--run", type=int, default=1, help="Fused checkpoint run id when using default path.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ef_prediction/gradcam_results/patients",
        help="Root; fused|real each get sex/, age/, bmi/ group subfolders.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max number of manifest rows (for testing).")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip PNGs that already exist.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda | cpu (default: cuda if available).",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)

    if args.mode in ("fused", "both"):
        manifest = Path(args.manifest_fused or cfg["data"]["val_manifest_fused"])
        ckpt = Path(
            args.checkpoint_fused
            or f"ef_prediction/checkpoints/fused/run_{args.run}_best.pth"
        )
        run_fused(cfg, device, ckpt, manifest, output_dir, args.limit, args.skip_existing)

    if args.mode in ("real", "both"):
        manifest = Path(args.manifest_real or cfg["data"]["val_manifest"])
        ckpt = Path(args.checkpoint_real or "ef_prediction/checkpoints/real/best.pth")
        run_real(cfg, device, ckpt, manifest, output_dir, args.limit, args.skip_existing)

    print(f"Done. Outputs under {output_dir.resolve()}")


if __name__ == "__main__":
    main()
