import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionBranch(nn.Module):
    def __init__(self):
        super(RegressionBranch, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  # Input: (3, H, W)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # downsample by 2

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(512, 1, 1),  # Output: density map
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        return self.output_layer(x)


class AttentionBranch(nn.Module):
    def __init__(self):
        super(AttentionBranch, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()  # Values in [0, 1] for attention weights
        )

    def forward(self, x):
        x = self.features(x)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        return x
