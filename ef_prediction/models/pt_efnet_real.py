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


class PTEFNetReal(nn.Module):
    def __init__(self, hidden_dim=256, demo_dim=32, backbone: str = "resnet34"):
        super().__init__()

        self.cnn = build_resnet_cnn(backbone)

        self.lstm = nn.LSTM(512, hidden_dim, batch_first=True, bidirectional=True)
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
            nn.Linear(pool_dim + demo_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, video, demo_vec):
        B, C, T, H, W = video.shape

        if C == 1:
            video = video.repeat(1, 3, 1, 1, 1)

        video = video.permute(0, 2, 1, 3, 4)
        video = video.reshape(B * T, 3, H, W)

        feats = self.cnn(video).squeeze(-1).squeeze(-1)
        feats = feats.view(B, T, 512)

        lstm_out, _ = self.lstm(feats)
        pooled = self.attn(lstm_out)

        z = self.projection_head(pooled)
        demo_emb = self.demo_encoder(demo_vec.float())
        fused = torch.cat([pooled, demo_emb], dim=1)
        ef = self.head(fused).squeeze(1)

        return ef, z