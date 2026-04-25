"""
Build a single panel image with one representative Grad-CAM per demographic subgroup.

Selection strategy per subgroup:
1) Prefer high-confidence Grad-CAM overlays (strong hot region intensity).
2) Encourage spatial diversity of highlighted regions across selected subgroups.

Example:
  python -m ef_prediction.build_demographic_gradcam_panel \
      --source ef_prediction/gradcam_results/patients/real \
      --output ef_prediction/gradcam_results/figures/demographic_subgroup_panel_real.png
"""

from __future__ import annotations

import argparse
import ast
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


GROUPS: list[tuple[str, str]] = [
    ("sex", "female"),
    ("sex", "male"),
    ("age", "0-1"),
    ("age", "2-5"),
    ("age", "6-10"),
    ("age", "11-15"),
    ("age", "16-18"),
    ("bmi", "underweight"),
    ("bmi", "normal"),
    ("bmi", "overweight"),
    ("bmi", "obese"),
]


@dataclass
class Candidate:
    path: Path
    strength: float
    centroid_xy: tuple[float, float]


@dataclass
class Selection:
    group_type: str
    group_name: str
    candidate: Candidate
    display_img: np.ndarray  # RGB
    sex_value: str | None = None
    exact_age_years: float | None = None
    bmi_value: float | None = None
    bmi_category: str | None = None


def _panel_crop(img_bgr: np.ndarray) -> np.ndarray:
    """Crop the top square Grad-CAM overlay area (caption strip excluded)."""
    h, w = img_bgr.shape[:2]
    side = min(w, h)
    return img_bgr[:side, :side]


def _heat_mask_and_strength(overlay_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Detect hot (red/yellow) regions in JET overlay and return a score.
    Score balances focus strength and compactness.
    """
    hsv = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2HSV)
    # Red wrap-around in HSV: low and high hue ranges.
    low1 = np.array([0, 90, 80], dtype=np.uint8)
    high1 = np.array([20, 255, 255], dtype=np.uint8)
    low2 = np.array([160, 90, 80], dtype=np.uint8)
    high2 = np.array([179, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, low1, high1) | cv2.inRange(hsv, low2, high2)

    red_channel = overlay_bgr[..., 2].astype(np.float32) / 255.0
    focus_strength = float(np.percentile(red_channel, 98))
    area_ratio = float(mask.mean() / 255.0)
    # Prefer visible but not over-diffuse highlights.
    compactness_bonus = math.exp(-abs(area_ratio - 0.14) * 4.0)
    score = focus_strength * (0.75 + 0.25 * compactness_bonus)
    return mask, score


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return (0.5, 0.5)
    x = float(xs.mean() / max(1, mask.shape[1] - 1))
    y = float(ys.mean() / max(1, mask.shape[0] - 1))
    return (x, y)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _rank_candidates(folder: Path) -> list[Candidate]:
    cands: list[Candidate] = []
    for p in sorted(folder.glob("*.png")):
        img = cv2.imread(str(p))
        if img is None:
            continue
        overlay = _panel_crop(img)
        mask, strength = _heat_mask_and_strength(overlay)
        centroid = _mask_centroid(mask)
        cands.append(Candidate(path=p, strength=strength, centroid_xy=centroid))
    return sorted(cands, key=lambda c: c.strength, reverse=True)


def _choose_diverse(cands: list[Candidate], existing: list[Selection]) -> Candidate | None:
    if not cands:
        return None
    if not existing:
        return cands[0]

    used_stems = {s.candidate.path.stem for s in existing}
    best: Candidate | None = None
    best_score = -1.0
    for c in cands[:250]:
        if c.path.stem in used_stems:
            continue
        min_dist = min(_distance(c.centroid_xy, s.candidate.centroid_xy) for s in existing)
        # Strong CAM + spatial diversity against already selected groups.
        score = 0.68 * c.strength + 0.32 * min_dist
        if score > best_score:
            best_score = score
            best = c
    return best if best is not None else cands[0]


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _format_age(age_val: float | None) -> str | None:
    if age_val is None or (isinstance(age_val, float) and math.isnan(age_val)):
        return None
    if abs(age_val - round(age_val)) < 1e-6:
        return f"{int(round(age_val))}"
    return f"{age_val:.1f}"


def _format_sex(sex_val: str | None) -> str | None:
    if not sex_val:
        return None
    s = str(sex_val).strip().upper()
    if s == "M":
        return "Male"
    if s == "F":
        return "Female"
    return str(sex_val).strip().title()


def _format_bmi(bmi_val: float | None) -> str | None:
    if bmi_val is None or (isinstance(bmi_val, float) and math.isnan(bmi_val)):
        return None
    return f"{bmi_val:.1f}"


def _bmi_category_from_value(bmi_val: float | None) -> str | None:
    if bmi_val is None or (isinstance(bmi_val, float) and math.isnan(bmi_val)):
        return None
    if bmi_val < 18.5:
        return "Underweight"
    if bmi_val < 25.0:
        return "Normal"
    if bmi_val < 30.0:
        return "Overweight"
    return "Obese"


def _stem_from_row(row: pd.Series) -> str | None:
    for k in ("processed_path", "original_path", "file_name"):
        if k in row and pd.notna(row[k]):
            s = Path(str(row[k])).stem
            if s:
                return s
    return None


def _main_group_line(selection: Selection) -> str:
    label = selection.group_name.replace("_", " ").title()
    if selection.group_type == "sex":
        return f"Sex: {label}"
    if selection.group_type == "age":
        exact_age = _format_age(selection.exact_age_years)
        if exact_age is not None:
            return f"Age: {exact_age} years"
        return f"Age: {selection.group_name} years"
    return f"BMI: {label}"


def _build_demo_lookup(manifest_path: Path) -> dict[str, dict[str, float | str | None]]:
    df = pd.read_csv(manifest_path)
    out: dict[str, dict[str, float | str | None]] = {}
    age_bins = {0: "0-1", 1: "2-5", 2: "6-10", 3: "11-15", 4: "16-18"}
    bmi_bins = {0: "Underweight", 1: "Normal", 2: "Overweight", 3: "Obese"}
    for _, row in df.iterrows():
        stem = _stem_from_row(row)
        if not stem:
            continue
        age_val = float(row["age"]) if "age" in row and pd.notna(row["age"]) else None
        sex_val = str(row["sex"]) if "sex" in row and pd.notna(row["sex"]) else None
        bmi_val: float | None = None
        bmi_cat: str | None = None
        if "bmi" in row and pd.notna(row["bmi"]):
            bmi_val = float(row["bmi"])
        else:
            wt_ok = "weight" in row and pd.notna(row["weight"])
            ht_ok = "height" in row and pd.notna(row["height"])
            if wt_ok and ht_ok:
                h_m = float(row["height"]) / 100.0
                if h_m > 0:
                    bmi_val = float(row["weight"]) / (h_m * h_m)

        # Fused manifest fallback: parse encoded demographics vector.
        if "demographics" in row and pd.notna(row["demographics"]):
            try:
                vec = ast.literal_eval(str(row["demographics"]))
                if isinstance(vec, list) and len(vec) >= 11:
                    si = int(np.argmax(vec[0:2]))
                    ai = int(np.argmax(vec[2:7]))
                    bi = int(np.argmax(vec[7:11]))
                    if sex_val is None:
                        sex_val = "F" if si == 0 else "M"
                    if age_val is None:
                        # Exact age is unavailable in this manifest; use bin midpoint for display fallback.
                        age_mid = {0: 0.5, 1: 3.5, 2: 8.0, 3: 13.0, 4: 17.0}
                        age_val = age_mid.get(ai)
                    if bmi_cat is None:
                        bmi_cat = bmi_bins.get(bi)
            except (ValueError, SyntaxError):
                pass

        if bmi_cat is None:
            bmi_cat = _bmi_category_from_value(bmi_val)
        out[stem] = {"age": age_val, "sex": sex_val, "bmi": bmi_val, "bmi_category": bmi_cat}
    return out


def _load_times_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    # Try common Times New Roman font files across Linux installs.
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


def _render_panel(selections: list[Selection], output_path: Path, model_label: str, tile_w: int = 360) -> None:
    tile_h = tile_w + 156
    pad = 24
    row_gap = 34
    head_h = 120
    group_title_h = 38
    max_cols = 5  # age row
    canvas_w = pad + max_cols * (tile_w + pad)
    canvas_h = head_h + pad + 3 * (group_title_h + tile_h) + 2 * row_gap + pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    font_title = _load_times_font(38, bold=True)
    font_subtitle = _load_times_font(24, bold=False)
    font_group = _load_times_font(30, bold=True)
    font_caption = _load_times_font(30, bold=True)
    font_meta = _load_times_font(22, bold=False)

    title = f"Representative Grad-CAM by Demographic Subgroup ({model_label})"
    draw.text((pad, 18), title, fill=(20, 20, 20), font=font_title)
    draw.text(
        (pad, 64),
        "Sex (2), Age (5), BMI (4) subgroup layout with per-image demographics.",
        fill=(65, 65, 65),
        font=font_subtitle,
    )

    rows_data = [
        ("Sex", [s for s in selections if s.group_type == "sex"]),
        ("Age", [s for s in selections if s.group_type == "age"]),
        ("BMI", [s for s in selections if s.group_type == "bmi"]),
    ]

    y_cursor = head_h + pad
    for row_name, row_items in rows_data:
        draw.text((pad, y_cursor), row_name, fill=(20, 20, 20), font=font_group)
        y0 = y_cursor + group_title_h

        x_start = pad

        for i, sel in enumerate(row_items):
            x0 = x_start + i * (tile_w + pad)

            img = Image.fromarray(sel.display_img).resize((tile_w, tile_w), Image.Resampling.BILINEAR)
            canvas.paste(img, (x0, y0))

            main_line = _main_group_line(sel)
            sex_text = _format_sex(sel.sex_value) or "Unknown"
            age_fmt = _format_age(sel.exact_age_years)
            age_text = f"{age_fmt} years" if age_fmt is not None else "Unknown"
            bmi_cat = sel.bmi_category or _bmi_category_from_value(sel.bmi_value)
            if sel.group_type == "bmi":
                bmi_text = sel.group_name.replace("_", " ").title()
            else:
                bmi_text = bmi_cat or "Unknown"

            y1 = y0 + tile_w + 8
            y2 = y1 + 42
            y3 = y2 + 34

            # Keep long headings (e.g., BMI: Underweight) inside tile width.
            cap_font = font_caption
            if draw.textlength(main_line, font=font_caption) > (tile_w - 6):
                cap_font = _load_times_font(26, bold=True)
            if draw.textlength(main_line, font=cap_font) > (tile_w - 6):
                cap_font = _load_times_font(22, bold=True)

            draw.text((x0, y1), main_line, fill=(15, 15, 15), font=cap_font)
            if sel.group_type == "sex":
                draw.text((x0, y2), f"Age: {age_text}", fill=(35, 35, 35), font=font_meta)
                draw.text((x0, y3), f"BMI: {bmi_text}", fill=(35, 35, 35), font=font_meta)
            elif sel.group_type == "age":
                draw.text((x0, y2), f"Sex: {sex_text}", fill=(35, 35, 35), font=font_meta)
                draw.text((x0, y3), f"BMI: {bmi_text}", fill=(35, 35, 35), font=font_meta)
            else:
                draw.text((x0, y2), f"Sex: {sex_text}", fill=(35, 35, 35), font=font_meta)
                draw.text((x0, y3), f"Age: {age_text}", fill=(35, 35, 35), font=font_meta)

        y_cursor = y0 + tile_h + row_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_panel(source_root: Path, output_path: Path, manifest_path: Path, model_label: str) -> list[Selection]:
    demo_lookup = _build_demo_lookup(manifest_path)
    selections: list[Selection] = []

    for group_type, group_name in GROUPS:
        folder = source_root / group_type / group_name
        if not folder.is_dir():
            continue
        ranked = _rank_candidates(folder)
        pick = _choose_diverse(ranked, selections)
        if pick is None:
            continue
        img = _load_rgb(pick.path)
        overlay = _panel_crop(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        selections.append(
            Selection(
                group_type=group_type,
                group_name=group_name,
                candidate=pick,
                display_img=overlay_rgb,
                sex_value=(demo_lookup.get(pick.path.stem) or {}).get("sex"),
                exact_age_years=(demo_lookup.get(pick.path.stem) or {}).get("age"),  # type: ignore[arg-type]
                bmi_value=(demo_lookup.get(pick.path.stem) or {}).get("bmi"),  # type: ignore[arg-type]
                bmi_category=(demo_lookup.get(pick.path.stem) or {}).get("bmi_category"),  # type: ignore[arg-type]
            )
        )

    if not selections:
        raise RuntimeError(f"No subgroup images found under: {source_root}")

    _render_panel(selections, output_path, model_label=model_label)
    return selections


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a single multi-subgroup Grad-CAM panel.")
    parser.add_argument(
        "--source",
        type=str,
        default="ef_prediction/gradcam_results/patients/real",
        help="Root with sex/, age/, bmi/ subgroup folders containing PNG overlays.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ef_prediction/gradcam_results/figures/demographic_subgroup_panel_real.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/processed_full/val_manifest.csv",
        help="Manifest CSV used to map selected patient to exact age.",
    )
    parser.add_argument(
        "--model-label",
        type=str,
        default=None,
        help="Title suffix, e.g. 'Real Model' or 'Fused Model'. Auto-inferred if omitted.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    manifest = Path(args.manifest)
    model_label = args.model_label
    if not model_label:
        model_label = "Fused Model" if "fused" in str(source).lower() or "fused" in str(output).lower() else "Real Model"
    selections = build_panel(source, output, manifest, model_label=model_label)
    print(f"Saved panel: {output.resolve()}")
    print(f"Subgroups included: {len(selections)}")


if __name__ == "__main__":
    main()
