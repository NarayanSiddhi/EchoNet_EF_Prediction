import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import faiss  # type: ignore
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import gridspec

try:
    from torchvision.models.video import r3d_18, R3D_18_Weights
except Exception:
    r3d_18 = None
    R3D_18_Weights = None


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_video_frames(path: Path, target_frames: int = 6, target_size: int = 128) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: List[np.ndarray] = []
    for _ in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (target_size, target_size), interpolation=cv2.INTER_AREA)
        frames.append(g)
    cap.release()
    if len(frames) == 0:
        return []
    if len(frames) >= target_frames:
        idx = np.linspace(0, len(frames) - 1, target_frames).astype(int)
        frames = [frames[i] for i in idx]
    else:
        frames = frames + [frames[-1]] * (target_frames - len(frames))
    return frames


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_backbone(device: torch.device) -> nn.Module:
    if r3d_18 is None:
        raise RuntimeError("torchvision not available")
    try:
        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights)
    except Exception:
        model = r3d_18(weights=None)
    model.fc = Identity()
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def embed_video(model: nn.Module, path: Path, target_frames: int = 32, target_size: int = 128, device: torch.device = torch.device("cpu")) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: List[np.ndarray] = []
    for _ in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (target_size, target_size), interpolation=cv2.INTER_AREA)
        frames.append(g)
    cap.release()
    if len(frames) == 0:
        return None
    video = np.array(frames, dtype=np.float32) / 255.0
    if len(video) > target_frames:
        idx = np.linspace(0, len(video) - 1, target_frames).astype(int)
        video = video[idx]
    else:
        pad = target_frames - len(video)
        video = np.concatenate([video, np.repeat(video[-1:], pad, axis=0)], axis=0)
    video = np.repeat(video[:, None, :, :], 3, axis=1)  # [T,3,H,W]
    tensor = torch.from_numpy(video).permute(1, 0, 2, 3).unsqueeze(0).to(device)  # [1,3,T,H,W]
    feat = model(tensor)
    feat = torch.nn.functional.normalize(feat, dim=1)
    return feat.squeeze(0).cpu().numpy().astype(np.float32)


def draw_system_diagram(out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.axis("off")
    boxes = ["Preprocess", "R3D-18 Embedding", "FAISS Index", "Query", "Top-k Results"]
    x_positions = [0.05, 0.27, 0.52, 0.74, 0.88]
    for label, x in zip(boxes, x_positions):
        rect = FancyBboxPatch((x, 0.35), 0.16, 0.3, boxstyle="round,pad=0.02,rounding_size=0.02",
                              edgecolor="black", facecolor="#e8f0fe", transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + 0.08, 0.5, label, ha="center", va="center", fontsize=11, transform=ax.transAxes)
    # arrows
    def arrow(x1, x2):
        ax.annotate("", xy=(x2, 0.5), xytext=(x1, 0.5), xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", lw=1.5))
    arrow(0.21, 0.27)  # Preprocess -> Emb
    arrow(0.43, 0.52)  # Emb -> FAISS
    arrow(0.68, 0.74)  # FAISS -> Query (conceptually indicates search)
    arrow(0.90, 0.88)  # Query -> Top-k (return)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _resolve_video_path(video_dir: Path, name: str) -> Optional[Path]:
    cand = video_dir / name
    if cand.exists():
        return cand
    for ext in [".mp4", ".avi", ".mov"]:
        p = video_dir / f"{name}{ext}"
        if p.exists():
            return p
    return None


def compose_query_panel(query_name: str, neighbor_list: List[Tuple[str, float]], video_dir: Path, out_path: Path, frames_per_row: int = 6, size: int = 128):
    rows = [query_name] + [n for n, _ in neighbor_list[:3]]
    sims = [1.0] + [s for _, s in neighbor_list[:3]]
    num_rows = len(rows)
    fig = plt.figure(figsize=(frames_per_row * 1.2, num_rows * 1.4))
    gs = gridspec.GridSpec(num_rows, frames_per_row, wspace=0.02, hspace=0.08)
    for r, (vid, sim) in enumerate(zip(rows, sims)):
        vpath = _resolve_video_path(video_dir, vid)
        if vpath is None:
            frames = [np.zeros((size, size), dtype=np.uint8)] * frames_per_row
        else:
            frames = load_video_frames(vpath, target_frames=frames_per_row, target_size=size)
            if len(frames) < frames_per_row:
                frames = frames + [frames[-1]] * (frames_per_row - len(frames))
        for c in range(frames_per_row):
            ax = fig.add_subplot(gs[r, c])
            if c == 0:
                ax.set_title(f"{vid}\nsim={sim:.3f}", fontsize=8, loc="left")
            ax.imshow(frames[c], cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_embeddings(emb_dir: Path, max_n: int = 1000) -> Tuple[np.ndarray, List[str]]:
    npys = sorted(emb_dir.glob("*.npy"))
    if len(npys) == 0:
        return np.zeros((0, 0), dtype=np.float32), []
    if len(npys) > max_n:
        npys = random.sample(npys, max_n)
    vecs = []
    names = []
    for p in npys:
        v = np.load(str(p)).astype(np.float32)
        if v.ndim == 1:
            vecs.append(v[None, :])
        else:
            vecs.append(v)
        names.append(p.stem)
    X = np.concatenate(vecs, axis=0)
    return X, names


def plot_histogram_top1_sims(index_path: Path, id_map_path: Path, emb_dir: Path, video_dir: Path, out_path: Path, num_queries: int = 200):
    index = faiss.read_index(str(index_path))
    with open(id_map_path, "r") as f:
        id_map: Dict[int, str] = {int(k): v for k, v in json.load(f).items()}
    inv_map: Dict[str, int] = {v: k for k, v in id_map.items()}
    # load a subset of embeddings to use as queries
    candidates = sorted(emb_dir.glob("*.npy"))
    if len(candidates) == 0:
        return
    qs = random.sample(candidates, min(num_queries, len(candidates)))
    sims = []
    for qn in qs:
        qv = np.load(str(qn)).astype(np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-12)
        D, I = index.search(qv[None, :], 2)  # include self
        # take top1 that is not self if possible
        sim_list = D[0].tolist()
        id_list = I[0].tolist()
        if len(id_list) > 1 and id_map.get(id_list[0], "") == qn.stem:
            sims.append(sim_list[1])
        else:
            sims.append(sim_list[0])
    plt.figure(figsize=(5, 3))
    plt.hist(sims, bins=20, color="#4a90e2", edgecolor="white")
    plt.xlabel("Cosine similarity (top-1 neighbor)")
    plt.ylabel("Count")
    plt.title("Similarity histogram (demo queries)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_tsne(emb_dir: Path, out_path: Path, max_n: int = 1000, random_state: int = 42):
    X, names = load_embeddings(emb_dir, max_n=max_n)
    if X.size == 0:
        return
    tsne = TSNE(n_components=2, init="random", learning_rate="auto", perplexity=30, random_state=random_state)
    Y = tsne.fit_transform(X)
    plt.figure(figsize=(5, 4))
    plt.scatter(Y[:, 0], Y[:, 1], s=6, c="#2d9cdb", alpha=0.7)
    plt.xticks([]); plt.yticks([])
    plt.title("t-SNE of video embeddings (demo)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def retrieve_topk(index_path: Path, id_map_path: Path, query_name: str, topk: int = 5) -> List[Tuple[str, float]]:
    index = faiss.read_index(str(index_path))
    with open(id_map_path, "r") as f:
        id_map: Dict[int, str] = {int(k): v for k, v in json.load(f).items()}
    inv_map: Dict[str, int] = {v: k for k, v in id_map.items()}
    if query_name not in inv_map:
        raise ValueError(f"{query_name} not found in id map")
    qid = inv_map[query_name]
    # read vector from embeddings directory by name is simpler for composition, but index does not expose vectors easily.
    # We use the index to get neighbors directly:
    # construct a unit vector equal to the indexed vector by retrieving its own vector via reconstruct (available on flat index).
    if not hasattr(index, "reconstruct"):
        raise RuntimeError("Index does not support reconstruct()")
    qvec = index.reconstruct(qid)
    qvec = qvec / (np.linalg.norm(qvec) + 1e-12)
    D, I = index.search(qvec[None, :].astype(np.float32), topk)
    sims = D[0].tolist()
    ids = I[0].tolist()
    return [(id_map[i], sims[j]) for j, i in enumerate(ids)]


def main():
    parser = argparse.ArgumentParser(description="Generate CBVR report figures")
    parser.add_argument("--video_dir", type=str, default="cbvr_retrieval/demo_videos")
    parser.add_argument("--emb_dir", type=str, default="cbvr_retrieval/embeddings_demo")
    parser.add_argument("--index", type=str, default="cbvr_retrieval/faiss_index_cosine_demo.bin")
    parser.add_argument("--id_map", type=str, default="cbvr_retrieval/id_map_demo.json")
    parser.add_argument("--out_dir", type=str, default="cbvr_retrieval/report_figures")
    parser.add_argument("--queries", type=str, nargs="*", default=[])
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    emb_dir = Path(args.emb_dir)
    index_path = Path(args.index)
    id_map_path = Path(args.id_map)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # 1) System diagram
    draw_system_diagram(out_dir / "system_diagram.png")

    # 2) Query panels (2 samples)
    # If not provided, pick two random from available demo files
    if not args.queries:
        vids = sorted(list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")))
        names = [p.name for p in vids]
        random.seed(42)
        args.queries = random.sample(names, k=min(2, len(names)))
    for q in args.queries:
        neighbors = retrieve_topk(index_path, id_map_path, q, topk=4)
        # compose panel with query + top3 neighbors
        compose_query_panel(q, neighbors[1:4], video_dir, out_dir / f"panel_{Path(q).stem}.png")

    # 3) Similarity histogram
    plot_histogram_top1_sims(index_path, id_map_path, emb_dir, video_dir, out_dir / "similarity_histogram.png", num_queries=200)

    # 4) t-SNE plot
    plot_tsne(emb_dir, out_dir / "tsne_embeddings.png", max_n=1000)

    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()

