import torch
import torch.nn as nn

class SiameseCNN(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),   # 105 -> 52
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 52 -> 26
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 26 -> 13
            nn.AdaptiveAvgPool2d(1),  # -> 128 x 1 x 1
        )
        self.fc = nn.Linear(128, embedding_dim)

    def forward_once(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, img_a, img_b):
        emb_a = self.forward_once(img_a)
        emb_b = self.forward_once(img_b)
        return emb_a, emb_b