import torch
import cv2
import numpy as np
from pathlib import Path
from .dataset import DualVideoEFDataset
from .models.pt_efnet_real import PTEFNetReal
import yaml


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, video_tensor, demo_vec):

        self.model.zero_grad()

        ef_pred, _ = self.model(video_tensor, demo_vec)
        ef_pred.backward(torch.ones_like(ef_pred))

        gradients = self.gradients
        activations = self.activations

        # activations shape: (B*T, C, H, W)
        # reshape back to (B, T, C, H, W)

        B = video_tensor.size(0)
        T = video_tensor.size(2)

        activations = activations.view(B, T, activations.size(1), activations.size(2), activations.size(3))
        gradients = gradients.view(B, T, gradients.size(1), gradients.size(2), gradients.size(3))

        # Take middle frame
        t = T // 2
        act = activations[:, t]
        grad = gradients[:, t]

        weights = torch.mean(grad, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * act, dim=1)

        cam = torch.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        cam = cv2.resize(cam, (128, 128))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


def main():

    with open("ef_prediction/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DualVideoEFDataset(
        manifest_path=cfg["data"]["val_manifest"],
        video_root_dir=cfg["data"]["original_video_dir"],
        video_length=cfg["model"]["video_length"],
        video_size=cfg["model"]["video_size"],
        fused=False
    )

    backbone = cfg["model"].get("backbone", "resnet34")
    model = PTEFNetReal(backbone=backbone).to(device)
    model.load_state_dict(
        torch.load("ef_prediction/checkpoints/real/best.pth", map_location=device)
    )
    model.train()

    target_layer = model.cnn[7]
    gradcam = GradCAM(model, target_layer)

    save_dir = Path("ef_prediction/gradcam_results/real")
    save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(10):  # first 10 samples
        video, ef, _, _, _, demo_vec = dataset[i]
        video = video.unsqueeze(0).to(device)
        demo_vec = demo_vec.unsqueeze(0).to(device).float()
        cam = gradcam.generate(video, demo_vec)

        frame = video[0, 0, video.shape[2] // 2].cpu().numpy()

        # cam = gradcam.generate(frame_tensor)

        # frame = frame_tensor[0, 0].detach().cpu().numpy()

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam),
            cv2.COLORMAP_JET
        )

        overlay = 0.5 * heatmap + 0.5 * np.stack([frame]*3, axis=-1)*255

        cv2.imwrite(
            str(save_dir / f"gradcam_{i}.png"),
            overlay
        )

    print("Saved GradCAM images")


if __name__ == "__main__":
    main()