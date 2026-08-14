"""
Create one fused-only counterfactual Grad-CAM summary image:
  - 3 sex comparisons
  - 3 age comparisons
  - 3 BMI comparisons

Each comparison shows Original vs Changed with full condition text below each image.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from .dataset_demographics import DualVideoEFDataset
from .gradcam_fused import GradCAM as GradCAMFused
from .models.pt_efnet_fused import PTEFNetFused

SEX_LABEL = {0: "female", 1: "male"}
AGE_LABEL = {0: "0-1", 1: "2-5", 2: "6-10", 3: "11-15", 4: "16-18"}
BMI_LABEL = {0: "underweight", 1: "normal", 2: "overweight", 3: "obese"}


def _load_times_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    if bold:
        candidates = [
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf",
            "/usr/share/fonts/truetype/microsoft/Times New Roman Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
            "/usr/share/fonts/truetype/microsoft/Times New Roman.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
        ]
    for fp in candidates:
        if Path(fp).is_file():
            try:
                return ImageFont.truetype(fp, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _draw_text(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    size: int,
    bold: bool = False,
    color: tuple[int, int, int] = (20, 20, 20),
) -> np.ndarray:
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    draw.text((x, y), text, fill=color, font=_load_times_font(size, bold=bold))
    return np.array(pil)


def _decode_demo(demo: torch.Tensor) -> tuple[int, int, int]:
    d = demo.detach().cpu().numpy().ravel()
    return int(np.argmax(d[0:2])), int(np.argmax(d[2:7])), int(np.argmax(d[7:11]))


def _encode_demo(sex_i: int, age_i: int, bmi_i: int, device: torch.device) -> torch.Tensor:
    v = np.zeros(11, dtype=np.float32)
    v[sex_i] = 1.0
    v[2 + age_i] = 1.0
    v[7 + bmi_i] = 1.0
    return torch.from_numpy(v).to(device=device).unsqueeze(0)


def _overlay(cam: np.ndarray, frame_gray: np.ndarray, size: int) -> np.ndarray:
    cam = cv2.resize(cam, (size, size))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    frame_gray = cv2.resize(frame_gray, (size, size))
    frame_rgb = np.stack([frame_gray] * 3, axis=-1)
    frame_rgb = (frame_rgb * 255).astype(np.uint8)
    return cv2.addWeighted(frame_rgb, 0.5, heatmap, 0.5, 0)


def _cond_text(sex_i: int, age_years: float, bmi_i: int, ef_pred_pct: float) -> list[str]:
    return [
        f"Sex: {SEX_LABEL[sex_i].title()}",
        f"Age: {age_years:.1f} years",
        f"BMI: {BMI_LABEL[bmi_i].title()}",
        f"Pred EF: {ef_pred_pct:.1f}",
    ]


def _draw_text_block(
    canvas: np.ndarray,
    lines: list[str],
    x: int,
    y: int,
    size: int = 18,
) -> np.ndarray:
    dy = 20
    for i, line in enumerate(lines):
        yy = y + i * dy
        canvas = _draw_text(canvas, line, x, yy - 14, size=size, bold=False, color=(20, 20, 20))
    return canvas


def _cam_shift_score(cam_a: np.ndarray, cam_b: np.ndarray) -> float:
    a = cv2.resize(cam_a, (128, 128)).astype(np.float32)
    b = cv2.resize(cam_b, (128, 128)).astype(np.float32)
    # Mean absolute heatmap difference as a simple saliency-shift proxy.
    return float(np.mean(np.abs(a - b)))


def _cam_strength(cam: np.ndarray) -> float:
    c = cv2.resize(cam, (128, 128)).astype(np.float32)
    # Prefer maps with a clear hotspot (high upper-tail response).
    return float(np.percentile(c, 99))


def _row_sex_from_manifest(row) -> int:
    # fused manifest stores encoded demographic vector in 'demographics'
    if "demographics" in row and isinstance(row["demographics"], str):
        try:
            v = ast.literal_eval(row["demographics"])
            if isinstance(v, list) and len(v) >= 2:
                return int(np.argmax(np.asarray(v[:2], dtype=float)))
        except (ValueError, SyntaxError):
            pass
    if "sex" in row and str(row["sex"]).strip().upper() == "M":
        return 1
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Build fused-only counterfactual Grad-CAM grid.")
    p.add_argument("--config", type=str, default="ef_prediction/config.yaml")
    p.add_argument("--run", type=int, default=1)
    p.add_argument("--per-feature", type=int, default=3, help="Comparisons for age/bmi rows.")
    p.add_argument("--sex-pairs", type=int, default=2, help="Comparisons for sex row.")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument(
        "--output",
        type=str,
        default="ef_prediction/gradcam_results/figures/fused_counterfactual_all_features.png",
    )
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = int(cfg["model"]["video_size"])

    ds = DualVideoEFDataset(
        manifest_path=cfg["data"]["val_manifest_fused"],
        video_root_dir=cfg["data"]["original_video_dir"],
        synthetic_root_dir=cfg["data"]["synthetic_video_dir"],
        video_length=cfg["model"]["video_length"],
        video_size=size,
        fused=True,
    )
    # Map patient stem -> exact age from real validation manifest.
    age_lookup: dict[str, float] = {}
    real_val_df = pd.read_csv(cfg["data"]["val_manifest"])
    for _, r in real_val_df.iterrows():
        if "processed_path" in r and pd.notna(r["processed_path"]) and "age" in r and pd.notna(r["age"]):
            age_lookup[Path(str(r["processed_path"])).stem] = float(r["age"])

    model = PTEFNetFused(**PTEFNetFused.kwargs_from_cfg(cfg)).to(device)
    ckpt = f"ef_prediction/checkpoints/fused/run_{args.run}_best.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.train()
    gradcam = GradCAMFused(model, model.cnn[7])

    k = int(args.per_feature)
    sex_k = int(args.sex_pairs)
    # Explicit target ages (years) for age counterfactual display + bin mapping.
    target_ages_years = [3.0, 9.0, 17.0]
    target_bmis = [0, 2, 3]

    # Pick samples that show larger Grad-CAM differences for each feature change.
    search_n = min(len(ds), 220)
    # sex_* candidate tuple:
    # (rank_score, min_strength, strength_orig, strength_changed, shift, idx)
    candidates = {"sex_m2f": [], "sex_f2m": [], "age": [], "bmi": []}
    # Build balanced index pools to guarantee both sex directions are searched.
    male_pool: list[int] = []
    female_pool: list[int] = []
    for ridx, row in ds.df.iterrows():
        sx = _row_sex_from_manifest(row)
        if sx == 1:
            male_pool.append(int(ridx))
        else:
            female_pool.append(int(ridx))

    # Sex scans from dedicated pools.
    sex_scan = min(search_n, len(male_pool), len(female_pool))
    for ridx in male_pool[:sex_scan]:
        real_v, syn_v, _, _, _, _, demo_vec = ds[ridx]
        real_v = real_v.unsqueeze(0).to(device)
        syn_v = syn_v.unsqueeze(0).to(device)
        demo = demo_vec.unsqueeze(0).to(device).float()
        s0, a0, b0 = _decode_demo(demo[0])
        cam0 = gradcam.generate(real_v, syn_v, demo)
        d_sex = _encode_demo(0, a0, b0, device)  # male -> female
        c_sex = gradcam.generate(real_v, syn_v, d_sex)
        shift = _cam_shift_score(cam0, c_sex)
        s0v = _cam_strength(cam0)
        s1v = _cam_strength(c_sex)
        smin = min(s0v, s1v)
        strength = 0.5 * (s0v + s1v)
        candidates["sex_m2f"].append((smin * 1.8 + strength * 0.6 + shift, smin, s0v, s1v, shift, ridx))

    for ridx in female_pool[:sex_scan]:
        real_v, syn_v, _, _, _, _, demo_vec = ds[ridx]
        real_v = real_v.unsqueeze(0).to(device)
        syn_v = syn_v.unsqueeze(0).to(device)
        demo = demo_vec.unsqueeze(0).to(device).float()
        s0, a0, b0 = _decode_demo(demo[0])
        cam0 = gradcam.generate(real_v, syn_v, demo)
        d_sex = _encode_demo(1, a0, b0, device)  # female -> male
        c_sex = gradcam.generate(real_v, syn_v, d_sex)
        shift = _cam_shift_score(cam0, c_sex)
        s0v = _cam_strength(cam0)
        s1v = _cam_strength(c_sex)
        smin = min(s0v, s1v)
        strength = 0.5 * (s0v + s1v)
        candidates["sex_f2m"].append((smin * 1.8 + strength * 0.6 + shift, smin, s0v, s1v, shift, ridx))

    # Age/BMI scans can use contiguous window.
    for idx in range(args.start_index, args.start_index + search_n):
        ridx = idx % len(ds)
        real_v, syn_v, _, _, _, _, demo_vec = ds[ridx]
        real_v = real_v.unsqueeze(0).to(device)
        syn_v = syn_v.unsqueeze(0).to(device)
        demo = demo_vec.unsqueeze(0).to(device).float()
        s0, a0, b0 = _decode_demo(demo[0])
        cam0 = gradcam.generate(real_v, syn_v, demo)

        # Age change (toward far bin when possible)
        a1 = 4 if a0 != 4 else 0
        d_age = _encode_demo(s0, a1, b0, device)
        c_age = gradcam.generate(real_v, syn_v, d_age)
        candidates["age"].append((_cam_shift_score(cam0, c_age), ridx))

        # BMI change (cyclic next class)
        b1 = (b0 + 1) % 4
        d_bmi = _encode_demo(s0, a0, b1, device)
        c_bmi = gradcam.generate(real_v, syn_v, d_bmi)
        candidates["bmi"].append((_cam_shift_score(cam0, c_bmi), ridx))

    # If one direction is still empty, fail fast with actionable guidance.
    if not candidates["sex_m2f"] or not candidates["sex_f2m"]:
        raise RuntimeError("Could not find both male and female pools for sex counterfactual selection.")

    used_idxs: set[int] = set()
    chosen: dict[str, list[int]] = {"age": [], "bmi": []}

    # Force requested sex row: one male->female pair and one female->male pair.
    sex_pairs: list[tuple[int, str]] = []
    for feat, direction in [("sex_m2f", "m2f"), ("sex_f2m", "f2m")]:
        ranked = sorted(candidates[feat], key=lambda x: x[0], reverse=True)
        # Visibility guardrail: require non-flat maps if possible.
        visible = [r for r in ranked if r[1] >= 0.50]
        pool = visible if visible else ranked
        for rec in pool:
            ridx = rec[-1]
            if ridx in used_idxs:
                continue
            sex_pairs.append((ridx, direction))
            used_idxs.add(ridx)
            break

    for feat in ["age", "bmi"]:
        ranked = sorted(candidates[feat], key=lambda x: x[0], reverse=True)
        for _, ridx in ranked:
            if ridx in used_idxs:
                continue
            chosen[feat].append(ridx)
            used_idxs.add(ridx)
            if len(chosen[feat]) >= k:
                break

    rows = [("Sex Change", sex_pairs), ("Age Change", chosen["age"]), ("BMI Change", chosen["bmi"])]

    # Layout (strict block model to prevent overlap)
    tile = 186
    text_h = 104
    pair_w = tile * 2 + 24
    heading_h = 26         # "Sex/Age/BMI Change"
    pair_label_h = 20      # "Original/Changed"
    pair_h = heading_h + pair_label_h + tile + text_h
    col_gap = 26
    row_gap = 18
    pad = 28
    title_h = 78
    max_pairs = max(len(sex_pairs), len(chosen["age"]), len(chosen["bmi"]), 1)
    canvas_w = pad * 2 + max_pairs * pair_w + (max_pairs - 1) * col_gap
    canvas_h = pad + title_h + len(rows) * pair_h + (len(rows) - 1) * row_gap + pad
    canvas = np.full((canvas_h, canvas_w, 3), 246, dtype=np.uint8)

    canvas = _draw_text(
        canvas,
        "Fused Counterfactual Grad-CAM: Demographic Feature Changes",
        pad,
        pad + 2,
        size=32,
        bold=True,
        color=(15, 15, 15),
    )
    canvas = _draw_text(
        canvas,
        "Each comparison keeps the same video and changes only one demographic feature.",
        pad,
        pad + 40,
        size=20,
        bold=False,
        color=(60, 60, 60),
    )

    y = pad + title_h
    for section_i, (section_name, idxs) in enumerate(rows):
        y_heading = y
        y_labels = y_heading + heading_h
        y_pair = y_labels + pair_label_h
        canvas = _draw_text(canvas, section_name, pad, y_heading - 2, size=26, bold=True, color=(20, 20, 20))

        for j, item in enumerate(idxs):
            x_pair = pad + j * (pair_w + col_gap)
            if section_i == 0:
                idx, sex_dir = item
            else:
                idx = item
                sex_dir = ""

            real_v, syn_v, _, _, _, _, demo_vec = ds[idx]
            real_v = real_v.unsqueeze(0).to(device)
            syn_v = syn_v.unsqueeze(0).to(device)
            demo = demo_vec.unsqueeze(0).to(device).float()
            s0, a0, b0 = _decode_demo(demo[0])
            stem = Path(str(ds.df.iloc[idx]["original_path"])).stem if "original_path" in ds.df.columns else None
            age0_years = age_lookup.get(stem, float([0.5, 3.5, 8.0, 13.0, 17.0][a0]))

            if section_i == 0:
                # Enforce requested directions.
                if sex_dir == "m2f":
                    s0 = 1
                    s1 = 0
                else:
                    s0 = 0
                    s1 = 1
                a1, b1 = a0, b0
                age1_years = age0_years
                demo = _encode_demo(s0, a0, b0, device)
            elif section_i == 1:
                age1_years = target_ages_years[j % len(target_ages_years)]
                # Map explicit age (years) to model age bin.
                if age1_years <= 1:
                    a1 = 0
                elif age1_years <= 5:
                    a1 = 1
                elif age1_years <= 10:
                    a1 = 2
                elif age1_years <= 15:
                    a1 = 3
                else:
                    a1 = 4
                if a1 == a0:
                    a1 = (a1 + 1) % 5
                    age1_years = [0.5, 3.0, 9.0, 13.0, 17.0][a1]
                s1, b1 = s0, b0
            else:
                tgt = target_bmis[j % len(target_bmis)]
                if tgt == b0:
                    tgt = (tgt + 1) % 4
                s1, a1, b1 = s0, a0, tgt
                age1_years = age0_years

            demo_cf = _encode_demo(s1, a1, b1, device)
            frame = real_v[0, 0, real_v.size(2) // 2].detach().cpu().numpy()

            cam0 = gradcam.generate(real_v, syn_v, demo)
            cam1 = gradcam.generate(real_v, syn_v, demo_cf)
            with torch.no_grad():
                p0, _ = model(real_v, syn_v, demo)
                p1, _ = model(real_v, syn_v, demo_cf)
            ef0 = float(p0.item() * 100.0)
            ef1 = float(p1.item() * 100.0)

            img0 = _overlay(cam0, frame, tile)
            img1 = _overlay(cam1, frame, tile)

            x0 = x_pair
            x1 = x_pair + tile + 24
            canvas[y_pair:y_pair + tile, x0:x0 + tile] = img0
            canvas[y_pair:y_pair + tile, x1:x1 + tile] = img1

            canvas = _draw_text(canvas, "Original", x0, y_labels - 1, size=19, bold=False, color=(25, 25, 25))
            canvas = _draw_text(canvas, "Changed", x1, y_labels - 1, size=19, bold=False, color=(25, 25, 25))
            canvas = _draw_text_block(canvas, _cond_text(s0, age0_years, b0, ef0), x0, y_pair + tile + 18, size=18)
            canvas = _draw_text_block(canvas, _cond_text(s1, age1_years, b1, ef1), x1, y_pair + tile + 18, size=18)

        # Draw separator lines after each pair in this section.
        y1 = y_labels - 2
        y2 = y_pair + tile + text_h - 8
        for j in range(max(0, len(idxs) - 1)):
            x_pair = pad + j * (pair_w + col_gap)
            x_sep = x_pair + pair_w + (col_gap // 2)
            cv2.line(canvas, (x_sep, y1), (x_sep, y2), (90, 90, 90), 2, cv2.LINE_AA)

        y = y + pair_h + row_gap

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print(f"Saved: {out.resolve()}")
    print(f"Comparisons: sex={len(sex_pairs)}, age={len(chosen['age'])}, bmi={len(chosen['bmi'])} (fused only)")


if __name__ == "__main__":
    main()

