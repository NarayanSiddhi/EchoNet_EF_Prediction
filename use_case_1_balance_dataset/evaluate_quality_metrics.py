#!/usr/bin/env python3
"""
Evaluate Use Case 1 synthetic videos using FID (and optionally FVD).

FID: Inception v3 features (memory-safe chunked forward).
FVD: optional I3D when --i3d_weights points to a valid checkpoint file.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import List, Optional, Tuple

import imageio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import linalg
from tqdm import tqdm

try:
    from pytorch_i3d import InceptionI3d

    I3D_AVAILABLE = True
except ImportError:
    I3D_AVAILABLE = False

try:
    import torchvision.models as models
    from torchvision.models import Inception_V3_Weights

    INCEPTION_AVAILABLE = True
except ImportError:
    INCEPTION_AVAILABLE = False


def _load_inception_v3(device: str):
    if not INCEPTION_AVAILABLE:
        raise RuntimeError("torchvision required for FID")
    try:
        w = Inception_V3_Weights.IMAGENET1K_V1
        inception = models.inception_v3(weights=w, transform_input=False)
    except Exception:
        inception = models.inception_v3(pretrained=True, transform_input=False)
    inception.fc = torch.nn.Identity()
    inception.eval()
    return inception.to(device)


class VideoFeatureExtractor:
    def __init__(self, device: str = "cuda", i3d_weights_path: Optional[str] = None):
        self.device = device
        self.inception_model = None
        self.i3d_model = None
        if INCEPTION_AVAILABLE:
            self.inception_model = _load_inception_v3(device)
        self._i3d_weights_path = i3d_weights_path

    def load_i3d_if_needed(self) -> bool:
        if self.i3d_model is not None:
            return True
        if not I3D_AVAILABLE:
            return False
        path = self._i3d_weights_path
        if not path or not Path(path).is_file():
            return False
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load(path, map_location=self.device))
        i3d.eval()
        self.i3d_model = i3d.to(self.device)
        return True

    def extract_inception_features(
        self, video_tensor: torch.Tensor, frames_per_chunk: int = 16
    ) -> np.ndarray:
        if self.inception_model is None:
            raise RuntimeError("Inception model not loaded")
        if video_tensor.dim() == 4:
            video_tensor = video_tensor.unsqueeze(0)
        B, T, C, H, W = video_tensor.shape
        flat = video_tensor.view(B * T, C, H, W)
        flat = F.interpolate(flat, size=(299, 299), mode="bilinear", align_corners=False)
        if C == 1:
            flat = flat.repeat(1, 3, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        flat = (flat - mean) / std

        parts: List[torch.Tensor] = []
        n = flat.shape[0]
        with torch.no_grad():
            for s in range(0, n, frames_per_chunk):
                chunk = flat[s : s + frames_per_chunk]
                parts.append(self.inception_model(chunk))
        features = torch.cat(parts, dim=0)
        features = features.view(B, T, -1).mean(dim=1)
        return features.cpu().numpy()

    def extract_i3d_features(self, video_tensor: torch.Tensor) -> np.ndarray:
        if self.i3d_model is None:
            raise RuntimeError("I3D model not loaded")
        if video_tensor.dim() == 4:
            video_tensor = video_tensor.unsqueeze(0)
        B, C, T, H, W = video_tensor.shape
        if C == 1:
            video_tensor = video_tensor.repeat(1, 3, 1, 1, 1)
        B, C, T, H, W = video_tensor.shape
        x = F.interpolate(
            video_tensor.view(B * C, T, H, W), size=(224, 224), mode="bilinear", align_corners=False
        )
        video_tensor = x.view(B, C, T, 224, 224)
        mean = torch.tensor([0.45, 0.45, 0.45], device=self.device).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225], device=self.device).view(1, 3, 1, 1, 1)
        video_tensor = (video_tensor - mean) / std
        with torch.no_grad():
            feats = self.i3d_model.extract_features(video_tensor)
        feats = F.adaptive_avg_pool3d(feats, (1, 1, 1)).squeeze()
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        return feats.cpu().numpy()


def load_video(
    video_path: Path, target_frames: int = 32, target_size: Tuple[int, int] = (128, 128)
) -> Optional[torch.Tensor]:
    try:
        reader = imageio.get_reader(str(video_path))
        frames = []
        for frame in reader:
            if len(frame.shape) == 3:
                frame = frame[:, :, 0] if frame.shape[2] >= 1 else frame.mean(axis=2)
            frames.append(frame.astype(np.float32) / 255.0)
        reader.close()
        if len(frames) == 0:
            return None
        video = torch.from_numpy(np.stack(frames, axis=0)).float()
        if len(video) > target_frames:
            idx = np.linspace(0, len(video) - 1, target_frames).astype(int)
            video = video[idx]
        elif len(video) < target_frames:
            pad = video[-1:].repeat(target_frames - len(video), 1, 1)
            video = torch.cat([video, pad], dim=0)
        if video.shape[1] != target_size[0] or video.shape[2] != target_size[1]:
            video = video.unsqueeze(1)
            video = F.interpolate(video, size=target_size, mode="bilinear", align_corners=False)
            video = video.squeeze(1)
        return video.unsqueeze(1)
    except Exception as e:
        print(f"Error loading {video_path}: {e}")
        return None


def calculate_fid(real_features: np.ndarray, synthetic_features: np.ndarray, eps: float = 1e-6) -> float:
    mu1 = real_features.mean(axis=0)
    mu2 = synthetic_features.mean(axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(synthetic_features, rowvar=False)
    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    covmean = (covmean + covmean.T) / 2.0
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return max(0.0, float(fid))


def extract_features_from_videos(
    video_paths: List[Path],
    extractor: VideoFeatureExtractor,
    batch_size: int = 1,
    use_fid: bool = True,
    use_fvd: bool = False,
    frames_per_chunk: int = 16,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    fid_features: List[np.ndarray] = []
    fvd_features: List[np.ndarray] = []

    for i in tqdm(range(0, len(video_paths), batch_size), desc="Extracting features"):
        batch_paths = video_paths[i : i + batch_size]
        batch_videos = []
        for path in batch_paths:
            v = load_video(path)
            if v is not None:
                batch_videos.append(v)
        if not batch_videos:
            continue
        video_batch = torch.stack(batch_videos).to(extractor.device)

        if use_fid:
            try:
                fid_batch = extractor.extract_inception_features(video_batch, frames_per_chunk=frames_per_chunk)
                fid_features.append(fid_batch)
            except Exception as e:
                print(f"Error extracting FID features: {e}")

        if use_fvd and extractor.i3d_model is not None:
            try:
                vb = video_batch.permute(0, 2, 1, 3, 4)
                fvd_batch = extractor.extract_i3d_features(vb)
                fvd_features.append(fvd_batch)
            except Exception as e:
                print(f"Error extracting FVD features: {e}")

        if extractor.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    fid_array = np.vstack(fid_features) if fid_features else None
    fvd_array = np.vstack(fvd_features) if fvd_features else None
    return fid_array, fvd_array


def main():
    parser = argparse.ArgumentParser(description="Evaluate synthetic vs real videos (FID / optional FVD)")
    parser.add_argument("--paired_manifest", type=str, default=None,
                        help="CSV with columns file_path and source (values: real, synthetic)")
    parser.add_argument("--real_manifest", type=str, default=None)
    parser.add_argument("--synthetic_manifest", type=str, default=None)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--real_video_col", type=str, default="processed_path")
    parser.add_argument("--synthetic_video_col", type=str, default="processed_path")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frames_per_chunk", type=int, default=16,
                        help="Max frames per Inception forward (lower = less GPU memory)")
    parser.add_argument("--output_file", type=str, default="uc1_quality_metrics.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--i3d_weights", type=str, default=None, help="Path to rgb_imagenet.pt for FVD")
    parser.add_argument("--use_fvd", action="store_true", default=False)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"Using device: {device}")

    video_dir = Path(args.video_dir)

    if args.paired_manifest:
        df = pd.read_csv(args.paired_manifest)
        if "source" not in df.columns or "file_path" not in df.columns:
            print("paired_manifest must contain columns: file_path, source")
            return
        real_df = df[df["source"].str.lower() == "real"]
        syn_df = df[df["source"].str.lower() == "synthetic"]
        if len(real_df) > args.num_samples:
            real_df = real_df.sample(n=args.num_samples, random_state=42)
        if len(syn_df) > args.num_samples:
            syn_df = syn_df.sample(n=args.num_samples, random_state=42)
        real_paths = [video_dir / p for p in real_df["file_path"]]
        synthetic_paths = [video_dir / p for p in syn_df["file_path"]]
    else:
        if not args.real_manifest or not args.synthetic_manifest:
            print("Provide --paired_manifest OR both --real_manifest and --synthetic_manifest")
            return
        real_manifest = pd.read_csv(args.real_manifest)
        synthetic_manifest = pd.read_csv(args.synthetic_manifest)
        if len(real_manifest) > args.num_samples:
            real_manifest = real_manifest.sample(n=args.num_samples, random_state=42)
        if len(synthetic_manifest) > args.num_samples:
            synthetic_manifest = synthetic_manifest.sample(n=args.num_samples, random_state=42)
        real_paths = [video_dir / path for path in real_manifest[args.real_video_col]]
        synthetic_paths = [video_dir / path for path in synthetic_manifest[args.synthetic_video_col]]

    real_paths = [p for p in real_paths if p.exists()]
    synthetic_paths = [p for p in synthetic_paths if p.exists()]
    print(f"Found {len(real_paths)} real videos and {len(synthetic_paths)} synthetic videos")

    if len(real_paths) == 0 or len(synthetic_paths) == 0:
        print("Error: No videos found after resolving paths.")
        return

    use_fvd = args.use_fvd and I3D_AVAILABLE and args.i3d_weights and Path(args.i3d_weights).is_file()
    if args.use_fvd and not use_fvd:
        print("FVD skipped: install pytorch_i3d and pass --i3d_weights to a valid .pt file")

    extractor = VideoFeatureExtractor(device=device, i3d_weights_path=args.i3d_weights)
    if use_fvd:
        extractor.load_i3d_if_needed()

    print("\nExtracting features from real videos...")
    real_fid, real_fvd = extract_features_from_videos(
        real_paths,
        extractor,
        args.batch_size,
        use_fid=True,
        use_fvd=use_fvd,
        frames_per_chunk=args.frames_per_chunk,
    )
    print("\nExtracting features from synthetic videos...")
    syn_fid, syn_fvd = extract_features_from_videos(
        synthetic_paths,
        extractor,
        args.batch_size,
        use_fid=True,
        use_fvd=use_fvd,
        frames_per_chunk=args.frames_per_chunk,
    )

    results: dict = {
        "num_real_videos": len(real_paths),
        "num_synthetic_videos": len(synthetic_paths),
        "fid": None,
        "fvd": None,
        "device": device,
        "frames_per_chunk": args.frames_per_chunk,
    }

    if real_fid is not None and syn_fid is not None:
        print("\nCalculating FID...")
        fid_score = calculate_fid(real_fid, syn_fid)
        results["fid"] = float(fid_score)
        print(f"FID Score: {fid_score:.2f}")

    if use_fvd and real_fvd is not None and syn_fvd is not None:
        print("\nCalculating FVD...")
        fvd_score = calculate_fid(real_fvd, syn_fvd)
        results["fvd"] = float(fvd_score)
        print(f"FVD Score: {fvd_score:.2f}")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
