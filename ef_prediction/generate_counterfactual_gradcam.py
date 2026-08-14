"""
Generate counterfactual Grad-CAM for one patient by changing demographics only.

This keeps the video fixed and edits one demographic attribute (sex / age / bmi),
so attribution shifts can be inspected directly.

Examples (run from repo root):
  python -m ef_prediction.generate_counterfactual_gradcam --model real --index 25 --feature sex
  python -m ef_prediction.generate_counterfactual_gradcam --model fused --index 25 --feature age --to 16-18
  python -m ef_prediction.generate_counterfactual_gradcam --model real --index 25 --feature bmi --to overweight
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from .dataset import DualVideoEFDataset as RealDataset
from .dataset_demographics import DualVideoEFDataset as FusedDataset
from .demographics_utils import AGE_MAP, BMI_MAP
from .gradcam_fused import GradCAM as GradCAMFused
from .gradcam_real import GradCAM as GradCAMReal
from .models.pt_efnet_fused import PTEFNetFused
from .models.pt_efnet_real import PTEFNetReal

SEX_LABEL = {0: "female", 1: "male"}
AGE_LABEL = {v: k for k, v in AGE_MAP.items()}
BMI_LABEL = {v: k for k, v in BMI_MAP.items()}


def _decode_demo(demo: torch.Tensor) -> tuple[int, int, int]:
    d = demo.detach().cpu().numpy().ravel()
    sex_i = int(np.argmax(d[0:2]))
    age_i = int(np.argmax(d[2:7]))
    bmi_i = int(np.argmax(d[7:11]))
    return sex_i, age_i, bmi_i


def _encode_demo(sex_i: int, age_i: int, bmi_i: int, device: torch.device) -> torch.Tensor:
    v = np.zeros(11, dtype=np.float32)
    v[sex_i] = 1.0
    v[2 + age_i] = 1.0
    v[7 + bmi_i] = 1.0
    return torch.from_numpy(v).to(device=device).unsqueeze(0)


def _apply_change(
    sex_i: int,
    age_i: int,
    bmi_i: int,
    feature: str,
    to_value: str | None,
) -> tuple[int, int, int]:
    if feature == "sex":
        if to_value is None:
            sex_i = 1 - sex_i
        else:
            t = to_value.strip().lower()
            sex_i = 1 if t in {"m", "male"} else 0
    elif feature == "age":
        if to_value is None:
            age_i = 4 if age_i != 4 else 0
        else:
            if to_value not in AGE_MAP:
                raise ValueError(f"Invalid age bin: {to_value}. Use one of {list(AGE_MAP.keys())}")
            age_i = AGE_MAP[to_value]
    elif feature == "bmi":
        if to_value is None:
            bmi_i = (bmi_i + 1) % 4
        else:
            t = to_value.strip().lower()
            if t not in BMI_MAP:
                raise ValueError(f"Invalid bmi class: {to_value}. Use one of {list(BMI_MAP.keys())}")
            bmi_i = BMI_MAP[t]
    else:
        raise ValueError(f"Unsupported feature: {feature}")
    return sex_i, age_i, bmi_i


def _overlay(cam: np.ndarray, frame_gray: np.ndarray, size: int) -> np.ndarray:
    cam = cv2.resize(cam, (size, size))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    frame_rgb = np.stack([frame_gray] * 3, axis=-1)
    frame_rgb = (frame_rgb * 255).astype(np.uint8)
    return cv2.addWeighted(frame_rgb, 0.5, heatmap, 0.5, 0)


def _caption_line(sex_i: int, age_i: int, bmi_i: int) -> str:
    return f"Sex={SEX_LABEL[sex_i]} | Age={AGE_LABEL[age_i]} | BMI={BMI_LABEL[bmi_i]}"


def _draw_pair(
    left_img: np.ndarray,
    right_img: np.ndarray,
    left_title: str,
    right_title: str,
    out_path: Path,
) -> None:
    h, w = left_img.shape[:2]
    pad = 18
    top = 70
    bot = 22
    gap = 24
    canvas = np.full((top + h + bot, w * 2 + gap + pad * 2, 3), 245, dtype=np.uint8)
    x1 = pad
    x2 = pad + w + gap
    y = top
    canvas[y:y + h, x1:x1 + w] = left_img
    canvas[y:y + h, x2:x2 + w] = right_img
    cv2.putText(canvas, left_title, (x1, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(canvas, right_title, (x2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (25, 25, 25), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def main() -> None:
    p = argparse.ArgumentParser(description="Counterfactual Grad-CAM by editing one demographic feature.")
    p.add_argument("--model", choices=["real", "fused"], default="real")
    p.add_argument("--index", type=int, default=0, help="Row index in validation dataset.")
    p.add_argument("--feature", choices=["sex", "age", "bmi"], default="sex")
    p.add_argument("--to", type=str, default=None, help="Target value (optional).")
    p.add_argument("--config", type=str, default="ef_prediction/config.yaml")
    p.add_argument("--run", type=int, default=1, help="Fused checkpoint run id.")
    p.add_argument(
        "--output",
        type=str,
        default="ef_prediction/gradcam_results/figures/counterfactual_gradcam.png",
    )
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = int(cfg["model"]["video_size"])

    if args.model == "real":
        ds = RealDataset(
            manifest_path=cfg["data"]["val_manifest"],
            video_root_dir=cfg["data"]["original_video_dir"],
            video_length=cfg["model"]["video_length"],
            video_size=size,
            fused=False,
        )
        backbone = cfg["model"].get("backbone", "resnet34")
        model = PTEFNetReal(backbone=backbone).to(device)
        model.load_state_dict(torch.load("ef_prediction/checkpoints/real/best.pth", map_location=device))
        model.train()
        gradcam = GradCAMReal(model, model.cnn[7])

        video, _, _, _, _, demo_vec = ds[args.index]
        video = video.unsqueeze(0).to(device)
        demo = demo_vec.unsqueeze(0).to(device).float()
        frame = video[0, 0, video.size(2) // 2].detach().cpu().numpy()
        cam_base = gradcam.generate(video, demo)
        pred_base, _ = model(video, demo)
    else:
        ds = FusedDataset(
            manifest_path=cfg["data"]["val_manifest_fused"],
            video_root_dir=cfg["data"]["original_video_dir"],
            synthetic_root_dir=cfg["data"]["synthetic_video_dir"],
            video_length=cfg["model"]["video_length"],
            video_size=size,
            fused=True,
        )
        model = PTEFNetFused(**PTEFNetFused.kwargs_from_cfg(cfg)).to(device)
        ckpt = f"ef_prediction/checkpoints/fused/run_{args.run}_best.pth"
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.train()
        gradcam = GradCAMFused(model, model.cnn[7])

        real_v, syn_v, _, _, _, _, demo_vec = ds[args.index]
        real_v = real_v.unsqueeze(0).to(device)
        syn_v = syn_v.unsqueeze(0).to(device)
        demo = demo_vec.unsqueeze(0).to(device).float()
        frame = real_v[0, 0, real_v.size(2) // 2].detach().cpu().numpy()
        cam_base = gradcam.generate(real_v, syn_v, demo)
        pred_base, _ = model(real_v, syn_v, demo)

    sex_i, age_i, bmi_i = _decode_demo(demo[0])
    sex2, age2, bmi2 = _apply_change(sex_i, age_i, bmi_i, args.feature, args.to)
    demo_cf = _encode_demo(sex2, age2, bmi2, device)

    if args.model == "real":
        cam_cf = gradcam.generate(video, demo_cf)
        pred_cf, _ = model(video, demo_cf)
    else:
        cam_cf = gradcam.generate(real_v, syn_v, demo_cf)
        pred_cf, _ = model(real_v, syn_v, demo_cf)

    img_base = _overlay(cam_base, frame, size)
    img_cf = _overlay(cam_cf, frame, size)
    t1 = f"Original ({_caption_line(sex_i, age_i, bmi_i)}) | EF={float(pred_base.item()*100):.1f}"
    t2 = f"Changed ({_caption_line(sex2, age2, bmi2)}) | EF={float(pred_cf.item()*100):.1f}"
    out = Path(args.output)
    _draw_pair(img_base, img_cf, t1, t2, out)
    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()

