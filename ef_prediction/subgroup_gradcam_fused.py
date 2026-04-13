import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

from .dataset_demographics import DualVideoEFDataset
from .demographics_utils import row_to_demo_vector
from .models.pt_efnet_fused import PTEFNetFused


def subgroup_match(name: str, demo: np.ndarray) -> bool:
    """
    Fused perfect-copy CSVs use 11-D demographics:
    sex(2) + age bins 0-1,2-5,6-10,11-15,16-18 + BMI under/normal/over/obese.
    Training manifests may use 14-D: sex(2) + age(8) + bmi(4).
    """
    d = np.asarray(demo, dtype=float)
    if d.size == 11:
        if name == "Male":
            return int(np.argmax(d[0:2])) == 1
        if name == "Female":
            return int(np.argmax(d[0:2])) == 0
        if name == "Early childhood":
            return d[2] > 0.5 or d[3] > 0.5
        if name == "Middle childhood":
            return d[4] > 0.5
        if name == "Adolescent":
            return d[5] > 0.5 or d[6] > 0.5
        if name == "Under weight":
            return d[7] > 0.5
        if name == "Normal weight":
            return d[8] > 0.5
        if name == "Over weight":
            return d[9] > 0.5 or d[10] > 0.5
        return False

    if d.size >= 14:
        sex = int(np.argmax(d[0:2]))
        age_i = int(np.argmax(d[2:10]))
        bmi_i = int(np.argmax(d[10:14]))
        if name == "Male":
            return sex == 1
        if name == "Female":
            return sex == 0
        if name == "Early childhood":
            return age_i in (0, 1, 2, 3)
        if name == "Middle childhood":
            return age_i in (4, 5)
        if name == "Adolescent":
            return age_i in (6, 7)
        if name == "Under weight":
            return bmi_i == 0
        if name == "Normal weight":
            return bmi_i == 1
        if name == "Over weight":
            return bmi_i in (2, 3)
        return False

    return False


# =========================================================
# GradCAM
# =========================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, real_video, syn_video, demo_vec):

        self.model.zero_grad()
        ef_pred, _ = self.model(real_video, syn_video, demo_vec)
        ef_pred.backward(torch.ones_like(ef_pred))

        gradients = self.gradients
        activations = self.activations

        B = real_video.size(0)
        T = real_video.size(2)

        activations = activations.view(
            B, T,
            activations.size(1),
            activations.size(2),
            activations.size(3)
        )

        gradients = gradients.view(
            B, T,
            gradients.size(1),
            gradients.size(2),
            gradients.size(3)
        )

        # Middle frame CAM
        t = T // 2
        act = activations[:, t]
        grad = gradients[:, t]

        weights = torch.mean(grad, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * act, dim=1)

        cam = torch.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        # Normalize CAM
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


# =========================================================
# Overlay Function (Robust Version)
# =========================================================
def overlay_cam(frame_tensor, cam):

    frame = frame_tensor.cpu().numpy()

    # Handle grayscale ultrasound (C=1)
    if frame.shape[0] == 1:
        frame = frame[0]
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
        frame = np.uint8(frame * 255)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        frame = frame.transpose(1, 2, 0)
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
        frame = np.uint8(frame * 255)

    H, W, _ = frame.shape

    cam_resized = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized),
        cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    return overlay


# =========================================================
# MAIN
# =========================================================
def main():

    with open("ef_prediction/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DualVideoEFDataset(
        manifest_path=cfg["data"]["val_manifest_fused"],
        video_root_dir=cfg["data"]["original_video_dir"],
        synthetic_root_dir=cfg["data"]["synthetic_video_dir"],
        video_length=cfg["model"]["video_length"],
        video_size=cfg["model"]["video_size"],
        fused=True
    )

    backbone = cfg["model"].get("backbone", "resnet34")
    model = PTEFNetFused(backbone=backbone).to(device)
    model.load_state_dict(
        torch.load(
            "ef_prediction/checkpoints/fused/run_1_best.pth",
            map_location=device,
        )
    )

    model.train()

    target_layer = model.cnn[7]
    gradcam = GradCAM(model, target_layer)

    miccai_groups = [
        "Early childhood",
        "Middle childhood",
        "Adolescent",
        "Under weight",
        "Normal weight",
        "Over weight",
        "Male",
        "Female",
    ]

    results = {}
    found = {name: False for name in miccai_groups}

    print("Generating single representative GradCAM per group...")

    for i in range(len(dataset.df)):
        row = dataset.df.iloc[i]
        demo = row_to_demo_vector(row)

        real_video = dataset.load_video(row["original_path"])
        syn_video = dataset.load_video(row["synthetic_path"])
        if real_video is None or syn_video is None:
            continue

        real_video_batch = real_video.unsqueeze(0).to(device)
        syn_video_batch = syn_video.unsqueeze(0).to(device)
        demo_t = (
            torch.from_numpy(row_to_demo_vector(row).copy())
            .float()
            .unsqueeze(0)
            .to(device)
        )

        cam = gradcam.generate(real_video_batch, syn_video_batch, demo_t)

        middle_frame = real_video[:, real_video.shape[1] // 2, :, :]

        for name in miccai_groups:
            if not found[name] and subgroup_match(name, demo):
                results[name] = overlay_cam(middle_frame, cam)
                found[name] = True

        if all(found.values()):
            break

    # =========================
    # MICCAI-STYLE LABELS
    # =========================
    miccai_labels = [
        "Early childhood\n(Age ≤ 6)",
        "Middle childhood\n(6 < Age ≤ 12)",
        "Adolescent\n(12 < Age ≤ 18)",
        "Under weight\n(BMI ≤ 18)",
        "Normal weight\n(18 < BMI ≤ 25)",
        "Over weight\n(25 ≤ BMI)",
        "Male",
        "Female"
    ]

    fig = plt.figure(figsize=(20, 6))

    for i, (name, label) in enumerate(zip(miccai_groups, miccai_labels)):
        ax = plt.subplot(1, 8, i + 1)
        if name in results:
            ax.imshow(results[name])
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    plt.subplots_adjust(wspace=0.05)

    save_dir = Path("ef_prediction/gradcam_results/subgroup")
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_dir / "miccai_representative_samples.png", dpi=300)
    plt.close()

    print("Saved MICCAI-style representative subgroup GradCAM grid.")


if __name__ == "__main__":
    main()