import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    import torchvision
    from torchvision.models.video import r3d_18, R3D_18_Weights
except Exception:
    torchvision = None
    r3d_18 = None
    R3D_18_Weights = None


def load_video(path: Path, target_frames: int, target_size: int) -> Optional[torch.Tensor]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: List[np.ndarray] = []
    for _ in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        return None
    video = np.array(frames, dtype=np.float32) / 255.0  # [T, H, W]
    # sample/pad to target_frames
    if len(video) > target_frames:
        idx = np.linspace(0, len(video) - 1, target_frames).astype(int)
        video = video[idx]
    elif len(video) < target_frames:
        pad = target_frames - len(video)
        video = np.concatenate([video, np.repeat(video[-1:], pad, axis=0)], axis=0)
    # to 3-channel and CHWT
    video = np.repeat(video[:, None, :, :], 3, axis=1)  # [T, 3, H, W]
    video = torch.from_numpy(video).permute(1, 0, 2, 3).contiguous()  # [3, T, H, W]
    return video


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_backbone(device: torch.device) -> nn.Module:
    if r3d_18 is None:
        raise RuntimeError("torchvision is not available. Please install torchvision.")
    try:
        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights)
    except Exception:
        print("Warning: Failed to load pretrained weights. Falling back to random init.")
        model = r3d_18(weights=None)
    # replace classifier head with identity to output features
    model.fc = Identity()
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def embed_batch(model: nn.Module, batch_videos: torch.Tensor, device: torch.device) -> torch.Tensor:
    batch_videos = batch_videos.to(device, non_blocking=True)
    feats = model(batch_videos)  # [B, C]
    # normalize for cosine similarity later
    feats = torch.nn.functional.normalize(feats, dim=1)
    return feats.cpu()


def main():
    parser = argparse.ArgumentParser(description="Extract R3D-18 embeddings for videos")
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--extensions", type=str, nargs="+", default=[".avi", ".mp4", ".mov"])
    parser.add_argument("--target_frames", type=int, default=32)
    parser.add_argument("--target_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = get_backbone(device)

    files: List[Path] = []
    for ext in args.extensions:
        files.extend(sorted(video_dir.glob(f"*{ext}")))
    if len(files) == 0:
        print("No videos found.")
        return

    # mapping for reference
    id_map = {}
    for idx, p in enumerate(files):
        id_map[idx] = p.name
    with open(output_dir / "id_map.json", "w") as f:
        json.dump(id_map, f, indent=2)

    # process
    batch: List[torch.Tensor] = []
    batch_meta: List[Tuple[int, Path]] = []
    for idx, p in tqdm(list(enumerate(files)), desc="Embedding videos"):
        out_path = output_dir / (p.stem + ".npy")
        if out_path.exists():
            continue
        vid = load_video(p, args.target_frames, args.target_size)
        if vid is None:
            continue
        batch.append(vid.unsqueeze(0))  # [1, 3, T, H, W]
        batch_meta.append((idx, p))
        if len(batch) == args.batch_size:
            batch_tensor = torch.cat(batch, dim=0)
            feats = embed_batch(model, batch_tensor, device)
            for (bid, bpath), vec in zip(batch_meta, feats):
                np.save(output_dir / (bpath.stem + ".npy"), vec.numpy().astype(np.float32))
            batch.clear()
            batch_meta.clear()
    if len(batch) > 0:
        batch_tensor = torch.cat(batch, dim=0)
        feats = embed_batch(model, batch_tensor, device)
        for (bid, bpath), vec in zip(batch_meta, feats):
            np.save(output_dir / (bpath.stem + ".npy"), vec.numpy().astype(np.float32))

    print(f"Done. Saved embeddings to {str(output_dir)}")


if __name__ == "__main__":
    main()

