"""
Author: Siddhi Narayan
Project: EF Prediction (Hierarchical Contrastive Learning)
Purpose: Visualize embeddings using t-SNE and UMAP
"""

# =========================
# 📦 Imports
# =========================
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from torch.utils.data import DataLoader
import yaml

from ef_prediction.dataset_demographics import DualVideoEFDataset
from ef_prediction.models.pt_efnet_fused import PTEFNetFused


# =========================
# ⚙️ Config + Device
# =========================
with open("ef_prediction/config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 📊 Load Dataset
# =========================
dataset = DualVideoEFDataset(
    manifest_path=cfg["data"]["val_manifest_fused"],
    video_root_dir=cfg["data"]["original_video_dir"],
    synthetic_root_dir=cfg["data"]["synthetic_video_dir"],
    video_length=cfg["model"]["video_length"],
    video_size=cfg["model"]["video_size"],
    fused=True
)

loader = DataLoader(dataset, batch_size=8, shuffle=False)


# =========================
# 🤖 Load Model
# =========================
backbone = cfg["model"].get("backbone", "resnet34")
model = PTEFNetFused(backbone=backbone).to(device)
model.load_state_dict(
    torch.load("ef_prediction/checkpoints/fused/run_1_best.pth", map_location=device)
)
model.eval()


# =========================
# 🔍 Extract Embeddings
# =========================
embeddings = []
sex_labels = []
age_labels = []
bmi_labels = []

with torch.no_grad():
    for batch in loader:

        # SAFELY unpack batch
        if len(batch) == 7:
            real_video, syn_video, ef, sex, age, bmi, demo_vec = batch
        else:
            raise ValueError("Dataset output format mismatch")

        real_video = real_video.to(device)
        syn_video = syn_video.to(device)
        demo_vec = demo_vec.to(device).float()

        # Extract embedding
        _, z = model(real_video, syn_video, demo_vec)

        embeddings.append(z.cpu().numpy())

        # Safe conversion
        sex_labels.append(sex.cpu().numpy())
        age_labels.append(age.cpu().numpy())
        bmi_labels.append(bmi.cpu().numpy())


# =========================
# 📏 Convert to Arrays
# =========================
embeddings = np.concatenate(embeddings)
sex_labels = np.concatenate(sex_labels)
age_labels = np.concatenate(age_labels)
bmi_labels = np.concatenate(bmi_labels)

print("✅ Embeddings shape:", embeddings.shape)


# =========================
# 📉 t-SNE
# =========================
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
z_tsne = tsne.fit_transform(embeddings)


# =========================
# 📊 UMAP
# =========================
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
z_umap = umap_model.fit_transform(embeddings)


# =========================
# 🎨 Plot Function
# =========================
import os

OUTPUT_DIR = "ef_prediction/embedding_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot(title, data, labels, cmap):
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(
        data[:, 0],
        data[:, 1],
        c=labels,
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


# =========================
# 📊 Generate Plots
# =========================

# SEX (categorical)
plot("t-SNE - SEX", z_tsne, sex_labels, cmap='tab10')
plot("UMAP - SEX", z_umap, sex_labels, cmap='tab10')

# AGE (continuous)
plot("t-SNE - AGE", z_tsne, age_labels, cmap='viridis')
plot("UMAP - AGE", z_umap, age_labels, cmap='viridis')

# BMI (continuous)
plot("t-SNE - BMI", z_tsne, bmi_labels, cmap='viridis')
plot("UMAP - BMI", z_umap, bmi_labels, cmap='viridis')
print("🎉 Visualization Complete!")
