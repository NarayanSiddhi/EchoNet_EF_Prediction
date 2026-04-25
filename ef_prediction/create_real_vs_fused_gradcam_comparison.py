"""
Create per-subgroup Real vs Fused Grad-CAM comparisons.

Layout:
  one subgroup per row, with two tiles side-by-side: Real | Fused

Example:
  python -m ef_prediction.create_real_vs_fused_gradcam_comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ef_prediction.build_demographic_gradcam_panel import (
    GROUPS,
    _bmi_category_from_value,
    _build_demo_lookup,
    _format_age,
    _format_sex,
    _main_group_line,
    _panel_crop,
    _rank_candidates,
)
from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
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


def _draw_tile(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    img_path: Path,
    group_type: str,
    group_name: str,
    demo_lookup: dict,
    x0: int,
    y0: int,
    tile_w: int,
    font_main: ImageFont.ImageFont,
    font_meta: ImageFont.ImageFont,
    demo_override: dict | None = None,
) -> None:
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return
    cropped = _panel_crop(bgr)
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    tile = Image.fromarray(rgb).resize((tile_w, tile_w), Image.Resampling.BILINEAR)
    canvas.paste(tile, (x0, y0))

    stem = img_path.stem
    demo = demo_override if demo_override is not None else demo_lookup.get(stem, {})
    sex_text = _format_sex(demo.get("sex")) or "Unknown"
    age_fmt = _format_age(demo.get("age"))
    age_text = f"{age_fmt} years" if age_fmt is not None else "Unknown"
    bmi_cat = demo.get("bmi_category") or _bmi_category_from_value(demo.get("bmi"))
    bmi_text = bmi_cat or "Unknown"

    fake_selection = type("S", (), {"group_type": group_type, "group_name": group_name, "exact_age_years": demo.get("age")})
    main_line = _main_group_line(fake_selection)

    y1 = y0 + tile_w + 8
    y2 = y1 + 40
    y3 = y2 + 32
    draw.text((x0, y1), main_line, fill=(15, 15, 15), font=font_main)
    if group_type == "sex":
        draw.text((x0, y2), f"Age: {age_text}", fill=(35, 35, 35), font=font_meta)
        draw.text((x0, y3), f"BMI: {bmi_text}", fill=(35, 35, 35), font=font_meta)
    elif group_type == "age":
        draw.text((x0, y2), f"Sex: {sex_text}", fill=(35, 35, 35), font=font_meta)
        draw.text((x0, y3), f"BMI: {bmi_text}", fill=(35, 35, 35), font=font_meta)
    else:
        draw.text((x0, y2), f"Sex: {sex_text}", fill=(35, 35, 35), font=font_meta)
        draw.text((x0, y3), f"Age: {age_text}", fill=(35, 35, 35), font=font_meta)


def _pick_matched_pair(
    real_dir: Path,
    fused_dir: Path,
    exclude_stems: set[str] | None = None,
) -> tuple[Path, Path] | None:
    """
    Pick one subgroup example using the same patient stem in real and fused.
    Prefers the strongest combined saliency among stems not in exclude_stems
    (so different subgroup rows do not reuse the same patient when avoidable).
    """
    real_ranked = _rank_candidates(real_dir)
    fused_ranked = _rank_candidates(fused_dir)
    if not real_ranked or not fused_ranked:
        return None

    real_map = {c.path.stem: c for c in real_ranked}
    fused_map = {c.path.stem: c for c in fused_ranked}
    common = set(real_map.keys()) & set(fused_map.keys())
    if not common:
        return None

    ex = exclude_stems or set()
    ranked_stems = sorted(
        common,
        key=lambda s: float(real_map[s].strength) + float(fused_map[s].strength),
        reverse=True,
    )
    for stem in ranked_stems:
        if stem not in ex:
            return real_map[stem].path, fused_map[stem].path
    # Only one viable stem or all excluded: reuse best-ranked to keep row populated
    stem = ranked_stems[0]
    return real_map[stem].path, fused_map[stem].path


def create_comparison(real_root: Path, fused_root: Path, real_manifest: Path, fused_manifest: Path, output_path: Path) -> None:
    real_demo = _build_demo_lookup(real_manifest)
    fused_demo = _build_demo_lookup(fused_manifest)

    rows: list[tuple[str, str, Path, Path]] = []
    used_stems: set[str] = set()
    for group_type, group_name in GROUPS:
        real_dir = real_root / group_type / group_name
        fused_dir = fused_root / group_type / group_name
        if not real_dir.is_dir() or not fused_dir.is_dir():
            continue
        pair = _pick_matched_pair(real_dir, fused_dir, exclude_stems=used_stems)
        if pair is None:
            continue
        real_path, fused_path = pair
        used_stems.add(real_path.stem)
        rows.append((group_type, group_name, real_path, fused_path))

    if not rows:
        raise RuntimeError("No common subgroup images found between real and fused sources.")

    pad = 20
    # Tight header: title → subtitle → Real/Fused → subgroup rows (minimal dead space).
    title_y = 8
    subtitle_y = 52  # below 38pt title line
    top_h = 82  # y for "Real" / "Fused" column headings (below subtitle)
    # Clear gap below Real/Fused before subgroup labels (30pt headings need breathing room).
    model_label_h = 54
    row_gap = 20
    tile_w = 220
    tile_h = tile_w + 142
    col_gap = 28
    block_gap = 36

    # Requested arrangement: 4 rows in left block, 3 rows in right block.
    left_rows = rows[:4]
    right_rows = rows[4:]
    max_rows = max(len(left_rows), len(right_rows))

    block_w = tile_w + col_gap + tile_w
    canvas_w = pad + block_w + block_gap + block_w + pad
    y_base = top_h + model_label_h
    canvas_h = y_base + max_rows * tile_h + max(0, max_rows - 1) * row_gap + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)

    title_font = _load_font(38, bold=True)
    subtitle_font = _load_font(22, bold=False)
    model_font = _load_font(30, bold=True)
    row_font = _load_font(24, bold=True)
    main_font = _load_font(24, bold=True)
    meta_font = _load_font(20, bold=False)

    draw.text((pad, title_y), "Real vs Fused Grad-CAM (Per Subgroup)", fill=(20, 20, 20), font=title_font)
    draw.text(
        (pad, subtitle_y),
        "Rows show subgroup-matched comparisons: Real (left) vs Fused (right).",
        fill=(70, 70, 70),
        font=subtitle_font,
    )

    x_block_left = pad
    x_block_right = pad + block_w + block_gap
    x_real_left = x_block_left
    x_fused_left = x_block_left + tile_w + col_gap
    x_real_right = x_block_right
    x_fused_right = x_block_right + tile_w + col_gap
    draw.text((x_real_left, top_h), "Real", fill=(15, 15, 15), font=model_font)
    draw.text((x_fused_left, top_h), "Fused", fill=(15, 15, 15), font=model_font)
    draw.text((x_real_right, top_h), "Real", fill=(15, 15, 15), font=model_font)
    draw.text((x_fused_right, top_h), "Fused", fill=(15, 15, 15), font=model_font)

    # Vertical divider between 4-row and 3-row blocks.
    divider_x = x_block_left + block_w + (block_gap // 2)
    draw.line([(divider_x, top_h - 4), (divider_x, canvas_h - pad)], fill=(150, 150, 150), width=2)

    for i, (group_type, group_name, real_img, fused_img) in enumerate(left_rows):
        y0 = y_base + i * (tile_h + row_gap)
        row_label = f"{group_type.upper()}: {group_name.replace('_', ' ').title()}"
        draw.text((x_block_left, y0), row_label, fill=(35, 35, 35), font=row_font)
        tile_y = y0 + 34
        canonical_demo = real_demo.get(real_img.stem, {})
        _draw_tile(
            canvas, draw, real_img, group_type, group_name, real_demo, x_real_left, tile_y, tile_w, main_font, meta_font,
            demo_override=canonical_demo,
        )
        _draw_tile(
            canvas, draw, fused_img, group_type, group_name, fused_demo, x_fused_left, tile_y, tile_w, main_font, meta_font,
            demo_override=canonical_demo,
        )

    for i, (group_type, group_name, real_img, fused_img) in enumerate(right_rows):
        y0 = y_base + i * (tile_h + row_gap)
        row_label = f"{group_type.upper()}: {group_name.replace('_', ' ').title()}"
        draw.text((x_block_right, y0), row_label, fill=(35, 35, 35), font=row_font)
        tile_y = y0 + 36
        canonical_demo = real_demo.get(real_img.stem, {})
        _draw_tile(
            canvas, draw, real_img, group_type, group_name, real_demo, x_real_right, tile_y, tile_w, main_font, meta_font,
            demo_override=canonical_demo,
        )
        _draw_tile(
            canvas, draw, fused_img, group_type, group_name, fused_demo, x_fused_right, tile_y, tile_w, main_font, meta_font,
            demo_override=canonical_demo,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-subgroup Real-vs-Fused Grad-CAM comparison image.")
    parser.add_argument(
        "--real-root",
        type=str,
        default="ef_prediction/gradcam_results/patients/real",
        help="Root directory with real subgroup folders.",
    )
    parser.add_argument(
        "--fused-root",
        type=str,
        default="ef_prediction/gradcam_results/patients/fused",
        help="Root directory with fused subgroup folders.",
    )
    parser.add_argument(
        "--real-manifest",
        type=str,
        default="data/processed_full/val_manifest.csv",
        help="Manifest for real demographics lookup.",
    )
    parser.add_argument(
        "--fused-manifest",
        type=str,
        default="perfect_synthetic_copies/perfect_copies_val.csv",
        help="Manifest for fused demographics lookup.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ef_prediction/gradcam_results/figures/real_vs_fused_gradcam_comparison.png",
        help="Output comparison image path.",
    )
    args = parser.parse_args()

    create_comparison(
        Path(args.real_root),
        Path(args.fused_root),
        Path(args.real_manifest),
        Path(args.fused_manifest),
        Path(args.output),
    )
    print(f"Saved comparison: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
