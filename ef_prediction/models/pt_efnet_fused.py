import torch
import torch.nn as nn
import torchvision.models as models


def build_resnet_cnn(backbone: str = "resnet34") -> nn.Sequential:
    name = (backbone or "resnet34").lower()
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    return nn.Sequential(*list(m.children())[:-1])


class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):
        w = torch.softmax(self.fc(x), dim=1)
        return (x * w).sum(dim=1)


class PTEFNetFused(nn.Module):
    """
    Real + synthetic streams share the same ResNet trunk, then fuse per frame.

    fusion_mode:
      - "concat": [real||syn] -> Linear(256->128) (original).
      - "gated":  learned gate in [0,1]^128 so fused = g*real + (1-g)*syn (often better when syn is noisier).
    """

    frame_dim = 128

    def __init__(
        self,
        hidden_dim: int = 128,
        demo_dim: int = 32,
        backbone: str = "resnet34",
        fusion_mode: str = "concat",
        dropout: float = 0.3,
    ):
        super().__init__()

        self.fusion_mode = (fusion_mode or "concat").lower()
        if self.fusion_mode not in ("concat", "gated"):
            raise ValueError("fusion_mode must be 'concat' or 'gated'")

        self.cnn = build_resnet_cnn(backbone)

        self.frame_proj = nn.Linear(512, self.frame_dim)

        if self.fusion_mode == "concat":
            self.fusion_proj = nn.Linear(self.frame_dim * 2, self.frame_dim)
            self.gate_net = None
        else:
            self.fusion_proj = None
            self.gate_net = nn.Sequential(
                nn.Linear(self.frame_dim * 2, self.frame_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.frame_dim, self.frame_dim),
                nn.Sigmoid(),
            )

        self.lstm = nn.LSTM(self.frame_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = TemporalAttention(hidden_dim * 2)

        pool_dim = hidden_dim * 2
        self.demo_encoder = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, demo_dim),
        )

        self.projection_head = nn.Sequential(
            nn.Linear(pool_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.head = nn.Sequential(
            nn.Linear(pool_dim + demo_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    @staticmethod
    def kwargs_from_cfg(cfg: dict) -> dict:
        m = cfg.get("model", {})
        return {
            "backbone": m.get("backbone", "resnet34"),
            "hidden_dim": int(m.get("fused_lstm_hidden", 128)),
            "demo_dim": int(m.get("fused_demo_dim", 32)),
            "fusion_mode": m.get("fused_fusion", "concat"),
            "dropout": float(m.get("fused_dropout", 0.3)),
        }

    def extract_features(self, video):
        B, C, T, H, W = video.shape

        if C == 1:
            video = video.repeat(1, 3, 1, 1, 1)

        video = video.permute(0, 2, 1, 3, 4)
        video = video.reshape(B * T, 3, H, W)

        feats = self.cnn(video).squeeze(-1).squeeze(-1)
        feats = feats.view(B, T, 512)

        return self.frame_proj(feats)

    def forward(self, real_video, syn_video, demo_vec):

        real_feats = self.extract_features(real_video)
        syn_feats = self.extract_features(syn_video)

        cat = torch.cat([real_feats, syn_feats], dim=2)
        if self.fusion_mode == "gated":
            assert self.gate_net is not None
            g = self.gate_net(cat)
            fused_frames = g * real_feats + (1.0 - g) * syn_feats
        else:
            assert self.fusion_proj is not None
            fused_frames = self.fusion_proj(cat)

        lstm_out, _ = self.lstm(fused_frames)
        pooled = self.attn(lstm_out)

        z = self.projection_head(pooled)
        demo_emb = self.demo_encoder(demo_vec.float())
        h = torch.cat([pooled, demo_emb], dim=1)
        ef = self.head(h).squeeze(1)

        return ef, z
