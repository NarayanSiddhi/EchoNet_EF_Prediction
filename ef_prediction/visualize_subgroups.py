"""
Subgroup Visualization Script
Author: Siddhi Narayan

Purpose:
Generate t-SNE and UMAP plots for each demographic subgroup
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from torch.utils.data import DataLoader
import yaml

from ef_prediction.dataset_demographics import DualVideoEFDataset
from ef_prediction.models.pt_efnet_fused import PTEFNetFused

# ---------------- CONFIG ----------------
with open("ef_prediction/config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- OUTPUT DIR ----------------
OUTPUT_DIR = "ef_prediction/embedding_subgroup"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- DATA ----------------
dataset = DualVideoEFDataset(
    manifest_path=cfg["data"]["val_manifest_fused"],
    video_root_dir=cfg["data"]["original_video_dir"],
    synthetic_root_dir=cfg["data"]["synthetic_video_dir"],
    video_length=cfg["model"]["video_length"],
    video_size=cfg["model"]["video_size"],
    fused=True
)

loader = DataLoader(dataset, batch_size=8, shuffle=False)

# ---------------- MODEL ----------------
backbone = cfg["model"].get("backbone", "resnet34")
model = PTEFNetFused(backbone=backbone).to(device)
model.load_state_dict(
    torch.load("ef_prediction/checkpoints/fused/run_1_best.pth", map_location=device)
)
model.eval()

# ---------------- EXTRACT EMBEDDINGS ----------------
embeddings = []
sex_labels = []
age_labels = []
bmi_labels = []

with torch.no_grad():
    for real_video, syn_video, ef, sex, age, bmi, demo_vec in loader:

        real_video = real_video.to(device)
        syn_video = syn_video.to(device)
        demo_vec = demo_vec.to(device).float()

        _, z = model(real_video, syn_video, demo_vec)

        embeddings.append(z.cpu().numpy())
        sex_labels.append(sex.numpy())
        age_labels.append(age.numpy())
        bmi_labels.append(bmi.numpy())

# concat
embeddings = np.concatenate(embeddings)
sex_labels = np.concatenate(sex_labels)
age_labels = np.concatenate(age_labels)
bmi_labels = np.concatenate(bmi_labels)

print("✅ Embeddings shape:", embeddings.shape)

# ---------------- REDUCE DIM ----------------
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
z_tsne = tsne.fit_transform(embeddings)

umap_model = umap.UMAP(n_components=2, random_state=42)
z_umap = umap_model.fit_transform(embeddings)

# ---------------- PLOT FUNCTION ----------------
def plot_subset(title, data, labels, mask, cmap):
    subset_data = data[mask]
    subset_labels = labels[mask]

    if len(subset_data) < 20:
        return

    plt.figure(figsize=(6,5))
    scatter = plt.scatter(
        subset_data[:, 0],
        subset_data[:, 1],
        c=subset_labels,
        cmap=cmap,
        s=10
    )

    plt.title(title)
    plt.colorbar(scatter)
    plt.grid(True)
    plt.tight_layout()

    filename = title.replace(" ", "_") + ".png"
    path = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(path)
    print(f"📁 Saved: {path}")
    plt.close()

# ---------------- SEX GROUPS ----------------
print("\n🔵 Generating SEX subgroup plots...")
for s in np.unique(sex_labels):
    mask = sex_labels == s

    plot_subset(f"t-SNE_SEX_{s}", z_tsne, sex_labels, mask, "tab10")
    plot_subset(f"UMAP_SEX_{s}", z_umap, sex_labels, mask, "tab10")

# ---------------- AGE GROUPS ----------------
print("\n🟢 Generating AGE subgroup plots...")
for a in np.unique(age_labels):
    mask = age_labels == a

    plot_subset(f"t-SNE_AGE_{a}", z_tsne, age_labels, mask, "viridis")
    plot_subset(f"UMAP_AGE_{a}", z_umap, age_labels, mask, "viridis")

# ---------------- BMI GROUPS ----------------
print("\n🟡 Generating BMI subgroup plots...")
for b in np.unique(bmi_labels):
    mask = bmi_labels == b

    plot_subset(f"t-SNE_BMI_{b}", z_tsne, bmi_labels, mask, "coolwarm")
    plot_subset(f"UMAP_BMI_{b}", z_umap, bmi_labels, mask, "coolwarm")

print("\n🎉 All subgroup visualizations saved!")