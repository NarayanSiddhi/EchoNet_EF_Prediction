"""
Author: Siddhi Narayan
Project: EF Prediction (Hierarchical Contrastive Learning)
Purpose: Visualize embeddings using t-SNE and UMAP
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
import yaml
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from ef_prediction.dataset_demographics import DualVideoEFDataset
from ef_prediction.models.pt_efnet_fused import PTEFNetFused

OUTPUT_DIR = "ef_prediction/embedding_plots"

# Binary sex: red vs cyan (F=0, M=1 in dataset).
SEX_SCATTER_CMAP = ListedColormap(["#D62728", "#17BECF"])


def plot(title, data, labels, cmap, *, sex_binary=False):
    plt.figure(figsize=(6, 5))
    scatter_kw = dict(c=labels, cmap=cmap, s=10)
    if sex_binary:
        scatter_kw["vmin"] = 0
        scatter_kw["vmax"] = 1
    scatter = plt.scatter(data[:, 0], data[:, 1], **scatter_kw)
    plt.title(title)
    cb = plt.colorbar(scatter, ticks=[0, 1] if sex_binary else None)
    if sex_binary:
        cb.set_ticklabels(["0", "1"])
    plt.grid(True)
    plt.tight_layout()

    filename = title.replace(" ", "_") + ".png"
    path = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(path)
    print(f"📁 Saved: {path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="t-SNE / UMAP of fused EF embeddings")
    parser.add_argument(
        "--embedding",
        choices=("head_input", "contrastive"),
        default="head_input",
        help="head_input = concat(pooled, demo_emb) fed to EF head; contrastive = projection z",
    )
    parser.add_argument(
        "--checkpoint",
        default="ef_prediction/checkpoints/fused/run_1_best.pth",
        help="Path to fused model weights",
    )
    args = parser.parse_args()

    with open("ef_prediction/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = DualVideoEFDataset(
        manifest_path=cfg["data"]["val_manifest_fused"],
        video_root_dir=cfg["data"]["original_video_dir"],
        synthetic_root_dir=cfg["data"]["synthetic_video_dir"],
        video_length=cfg["model"]["video_length"],
        video_size=cfg["model"]["video_size"],
        fused=True,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    model = PTEFNetFused(**PTEFNetFused.kwargs_from_cfg(cfg)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    embeddings = []
    sex_labels = []
    age_labels = []
    bmi_labels = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) != 7:
                raise ValueError("Dataset output format mismatch")
            real_video, syn_video, ef, sex, age, bmi, demo_vec = batch

            real_video = real_video.to(device)
            syn_video = syn_video.to(device)
            demo_vec = demo_vec.to(device).float()

            _, emb = model(
                real_video,
                syn_video,
                demo_vec,
                return_embedding=args.embedding,
            )
            embeddings.append(emb.cpu().numpy())
            sex_labels.append(sex.cpu().numpy())
            age_labels.append(age.cpu().numpy())
            bmi_labels.append(bmi.cpu().numpy())

    embeddings = np.concatenate(embeddings)
    sex_labels = np.concatenate(sex_labels)
    age_labels = np.concatenate(age_labels)
    bmi_labels = np.concatenate(bmi_labels)

    print("✅ Embeddings shape:", embeddings.shape, "| mode:", args.embedding)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_tsne = tsne.fit_transform(embeddings)

    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    z_umap = umap_model.fit_transform(embeddings)

    em = args.embedding
    plot(f"t-SNE - SEX {em}", z_tsne, sex_labels, cmap=SEX_SCATTER_CMAP, sex_binary=True)
    plot(f"UMAP - SEX {em}", z_umap, sex_labels, cmap=SEX_SCATTER_CMAP, sex_binary=True)
    plot(f"t-SNE - AGE {em}", z_tsne, age_labels, cmap="viridis")
    plot(f"UMAP - AGE {em}", z_umap, age_labels, cmap="viridis")
    plot(f"t-SNE - BMI {em}", z_tsne, bmi_labels, cmap="viridis")
    plot(f"UMAP - BMI {em}", z_umap, bmi_labels, cmap="viridis")
    print("🎉 Visualization Complete!")


if __name__ == "__main__":
    main()
