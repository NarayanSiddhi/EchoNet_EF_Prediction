import torch
import torch.nn as nn
import torchvision.models as models


class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):
        # x: [B, T, D]
        w = torch.softmax(self.fc(x), dim=1)  # [B, T, 1]
        return (x * w).sum(dim=1)  # [B, D]


class PTEFNet(nn.Module):
    """
    Physio-Temporal EF Prediction Network

    Input : (B, 1, T, H, W)
    Output: (B,) EF in [0, 1]
    """

    def __init__(self, hidden_dim=256):
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.attn = TemporalAttention(hidden_dim * 2)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, video):
        """
        video: (B, 1, T, H, W)
        """
        assert video.dim() == 5, f"Expected 5D input, got {video.shape}"

        B, C, T, H, W = video.shape

        if C == 1:
            video = video.repeat(1, 3, 1, 1, 1)

        video = video.permute(0, 2, 1, 3, 4)
        video = video.reshape(B * T, 3, H, W)

        feats = self.cnn(video).squeeze(-1).squeeze(-1)
        feats = feats.view(B, T, 512)

        lstm_out, _ = self.lstm(feats)
        pooled = self.attn(lstm_out)

        ef = self.head(pooled).squeeze(1)
        return ef
