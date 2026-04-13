"""
Generate synthetic videos using a trained conditional C3D DCGAN checkpoint.

Architecture must match training (see c3dgan_arch.py / train_c3dgan.py --size).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

try:
    import imageio
except ImportError:
    imageio = None

from c3dgan_arch import Generator


def create_condition_vector(sex: int, age: int, bmi: int) -> torch.Tensor:
    """sex: 0=F, 1=M | age: 0..4 | bmi: 0..3"""
    cond = torch.zeros(11)
    cond[sex] = 1.0
    cond[2 + age] = 1.0
    cond[7 + bmi] = 1.0
    return cond


def save_video(tensor: torch.Tensor, path: str, fps: int = 30) -> None:
    """tensor [1,1,T,H,W] in [-1,1] -> .npy + optional mp4."""
    video = tensor.squeeze().detach().cpu().numpy()
    video = (video + 1.0) * 127.5
    video = np.clip(video, 0, 255).astype(np.uint8)

    np.save(path.replace(".mp4", ".npy"), video)

    if imageio is None:
        return
    try:
        frames = [video[t] for t in range(video.shape[0])]
        frames_rgb = np.stack([frames, frames, frames], axis=-1)
        imageio.mimwrite(path, frames_rgb, fps=fps, codec="libx264", quality=8)
    except Exception:
        pass


def generate_samples(
    checkpoint: str,
    num_samples: int,
    output_dir: str,
    device: str,
    z_dim: int = 128,
    size: int = 64,
) -> None:
    netG = Generator(z_dim=z_dim, cond_dim=11, size=size).to(device)
    state = torch.load(checkpoint, map_location=device)
    netG.load_state_dict(state)
    netG.eval()

    print(f"Loaded generator: {checkpoint}")
    print(f"Generating {num_samples} videos at {size}x{size} x T=32 -> {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    sex_labels = {0: "F", 1: "M"}
    age_labels = {0: "0-1y", 1: "2-5y", 2: "6-10y", 3: "11-15y", 4: "16-18y"}
    bmi_labels = {0: "underweight", 1: "normal", 2: "overweight", 3: "obese"}

    combinations = [(s, a, b) for s in [0, 1] for a in range(5) for b in range(4)]

    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="Generating"):
            sex, age, bmi = combinations[idx % len(combinations)]
            cond = create_condition_vector(sex, age, bmi).unsqueeze(0).to(device)
            noise = torch.randn(1, z_dim, device=device)
            fake = netG(noise, cond)

            name = (
                f"synth_{idx:04d}_sex{sex_labels[sex]}_"
                f"age{age_labels[age]}_bmi{bmi_labels[bmi]}.mp4"
            )
            save_video(fake, os.path.join(output_dir, name))

    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate videos with trained C3D DCGAN")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--output_dir", type=str, default="generated_videos")
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--size", type=int, default=128, choices=[32, 64, 128, 256])
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            dev = "mps"
        elif torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
    else:
        dev = args.device

    generate_samples(
        checkpoint=args.checkpoint,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        device=dev,
        z_dim=args.z_dim,
        size=args.size,
    )
