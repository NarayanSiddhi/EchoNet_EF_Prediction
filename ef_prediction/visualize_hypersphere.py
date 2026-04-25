"""
3D unit-sphere visualization (FairHICON Fig. 2 style).

Pipeline (typical for papers that show embeddings on S^2):
  1) L2-normalize each high-dimensional embedding -> on S^{d-1}.
  2) Project to 3D with PCA (linear view of main variance directions).
  3) L2-normalize the 3D points again -> lie on the unit sphere for plotting.
  4) Draw a wireframe sphere + scatter (color = sex / age / BMI / EF).

Note: FairHICON's purple/blue/red "territories" come from *separate* encoders
(EC, EM, EF) and a hierarchical contrastive objective on transcriptomics.
Your fused EF model has one pooled video path + demo_emb; you get the same
*display* recipe, not the same biology unless you redesign training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ef_prediction.dataset_demographics import DualVideoEFDataset
from ef_prediction.models.pt_efnet_fused import PTEFNetFused


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def unit_sphere_mesh(n_u: int = 48, n_v: int = 24):
    u = np.linspace(0.0, 2.0 * np.pi, n_u)
    v = np.linspace(0.0, np.pi, n_v)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    return xs, ys, zs


def parse_args():
    p = argparse.ArgumentParser(description="3D hypersphere PCA plot of fused embeddings")
    p.add_argument(
        "--embedding",
        choices=("contrastive", "head_input"),
        default="head_input",
        help="Same as visualize_embeddings.py",
    )
    p.add_argument(
        "--checkpoint",
        default="ef_prediction/checkpoints/fused/run_1_best.pth",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--color-by",
        choices=("sex", "age", "bmi", "ef"),
        default="sex",
    )
    p.add_argument(
        "--ef-threshold",
        type=float,
        default=0.5,
        help="If --color-by ef and --ef-binary, EF > threshold (in [0,1] scale) = high.",
    )
    p.add_argument(
        "--ef-mode",
        choices=("continuous", "binary"),
        default="continuous",
        help="continuous = color by EF; binary = two classes by --ef-threshold",
    )
    p.add_argument("--sphere-grid-u", type=int, default=40)
    p.add_argument("--sphere-grid-v", type=int, default=24)
    p.add_argument("--point-size", type=float, default=12.0)
    p.add_argument("--alpha", type=float, default=0.85)
    p.add_argument(
        "--output-dir",
        default="ef_prediction/embedding_plots",
    )
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()

    with open("ef_prediction/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DualVideoEFDataset(
        manifest_path=cfg["data"]["val_manifest_fused"],
        video_root_dir=cfg["data"]["original_video_dir"],
        synthetic_root_dir=cfg["data"]["synthetic_video_dir"],
        video_length=cfg["model"]["video_length"],
        video_size=cfg["model"]["video_size"],
        fused=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = PTEFNetFused(**PTEFNetFused.kwargs_from_cfg(cfg)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    emb_list, sex_l, age_l, bmi_l, ef_l = [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            real_video, syn_video, ef, sex, age, bmi, demo_vec = batch
            real_video = real_video.to(device)
            syn_video = syn_video.to(device)
            demo_vec = demo_vec.to(device).float()
            _, z = model(
                real_video,
                syn_video,
                demo_vec,
                return_embedding=args.embedding,
            )
            emb_list.append(z.cpu().numpy())
            sex_l.append(sex.cpu().numpy())
            age_l.append(age.cpu().numpy())
            bmi_l.append(bmi.cpu().numpy())
            ef_l.append(ef.cpu().numpy())

    X = np.concatenate(emb_list, axis=0).astype(np.float64)
    sex = np.concatenate(sex_l)
    age = np.concatenate(age_l)
    bmi = np.concatenate(bmi_l)
    ef = np.concatenate(ef_l).ravel()

    # Step 1: high-D unit norm (hypersphere in R^d)
    Xn = l2_normalize_rows(X)
    # Step 2–3: PCA -> R^3, then unit norm on S^2
    pca = PCA(n_components=min(3, Xn.shape[1]), random_state=42)
    Z3 = pca.fit_transform(Xn)
    Z3 = l2_normalize_rows(Z3)

    if args.color_by == "sex":
        c, cmap, label = sex, "coolwarm", "sex (0/1)"
    elif args.color_by == "age":
        c, cmap, label = age, "viridis", "age bin"
    elif args.color_by == "bmi":
        c, cmap, label = bmi, "viridis", "BMI bin"
    else:
        if args.ef_mode == "binary":
            c = (ef > args.ef_threshold).astype(np.float32)
            cmap, label = "coolwarm", f"EF high (> {args.ef_threshold})"
        else:
            c, cmap, label = ef, "plasma", "EF (0–1)"

    xs, ys, zs = unit_sphere_mesh(args.sphere_grid_u, args.sphere_grid_v)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(
        xs,
        ys,
        zs,
        rstride=2,
        cstride=2,
        color="0.5",
        linewidth=0.4,
        alpha=0.35,
    )

    sc = ax.scatter(
        Z3[:, 0],
        Z3[:, 1],
        Z3[:, 2],
        c=c,
        cmap=cmap,
        s=args.point_size,
        alpha=args.alpha,
        depthshade=True,
        edgecolors="none",
    )
    cb = plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.08)
    cb.set_label(label)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("PC1 (on $S^2$)")
    ax.set_ylabel("PC2 (on $S^2$)")
    ax.set_zlabel("PC3 (on $S^2$)")
    var3 = pca.explained_variance_ratio_.sum()
    title = (
        f"Unit sphere PCA | {args.embedding} | color={args.color_by}\n"
        f"3D explained var ratio ≈ {var3:.2%}"
    )
    ax.set_title(title, fontsize=10)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hypersphere3d_{args.color_by}_{args.embedding}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    print(f"Embeddings: N={X.shape[0]}, dim={X.shape[1]} -> PCA 3D -> L2 on S^2")


if __name__ == "__main__":
    main()
