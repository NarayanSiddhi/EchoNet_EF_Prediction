import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

import cv2

try:
    from torchvision.models.video import r3d_18, R3D_18_Weights
except Exception:
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
    if len(video) > target_frames:
        idx = np.linspace(0, len(video) - 1, target_frames).astype(int)
        video = video[idx]
    elif len(video) < target_frames:
        pad = target_frames - len(video)
        video = np.concatenate([video, np.repeat(video[-1:], pad, axis=0)], axis=0)
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
    model.fc = Identity()
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def embed(model: nn.Module, video: torch.Tensor, device: torch.device) -> np.ndarray:
    video = video.unsqueeze(0).to(device)  # [1, 3, T, H, W]
    feat = model(video)  # [1, D]
    feat = torch.nn.functional.normalize(feat, dim=1)
    return feat.squeeze(0).cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Query FAISS index with a video")
    parser.add_argument("--query_path", type=str, default="")
    parser.add_argument("--query_id", type=str, default="")
    parser.add_argument("--video_dir", type=str, default="data/processed/videos")
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--id_map", type=str, required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--target_frames", type=int, default=32)
    parser.add_argument("--target_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if faiss is None:
        raise RuntimeError("faiss is not installed. Please install faiss-cpu or faiss-gpu.")

    # load id map
    with open(args.id_map, "r") as f:
        id_map: Dict[int, str] = {int(k): v for k, v in json.load(f).items()}
    inv_map: Dict[str, int] = {v: k for k, v in id_map.items()}

    # load index
    index = faiss.read_index(args.index)

    # prepare query embedding
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = get_backbone(device)

    if args.query_path:
        qpath = Path(args.query_path)
        video = load_video(qpath, args.target_frames, args.target_size)
        if video is None:
            print("Could not load query video.")
            return
        q = embed(model, video, device)  # [D]
    elif args.query_id:
        # find the file by id (filename)
        vdir = Path(args.video_dir)
        candidate = vdir / args.query_id
        if not candidate.exists():
            print(f"Could not find {candidate}")
            return
        video = load_video(candidate, args.target_frames, args.target_size)
        if video is None:
            print("Could not load query video.")
            return
        q = embed(model, video, device)
    else:
        print("Provide --query_path or --query_id")
        return

    # cosine similarity search (index is IP on normalized vectors)
    q = q[None, :]  # [1, D]
    D, I = index.search(q.astype(np.float32), args.topk)
    sims = D[0].tolist()
    ids = I[0].tolist()
    results = [(id_map[i], sims[j]) for j, i in enumerate(ids) if i >= 0]

    print("\nTop-K similar videos:")
    for name, sim in results:
        print(f"- {name}\t(sim={sim:.4f})")


if __name__ == "__main__":
    main()

