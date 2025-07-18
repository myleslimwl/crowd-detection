import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class RegressionBranch(nn.Module):
    """
    A simple convolutional regression branch to estimate crowd density maps.
    Based on a truncated VGG16 backbone.
    """
    def __init__(self):
        super(RegressionBranch, self).__init__()
        vgg = models.vgg16_bn(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features.children())[:33])  # up to conv5_3

        self.regressor = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1)  # output 1-channel density map
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        x = F.interpolate(x, size=(x.shape[2] * 8, x.shape[3] * 8), mode='bilinear', align_corners=False)  # upscale
        return x


class AttentionBranch(nn.Module):
    """
    An attention module to generate spatial attention weights.
    """
    def __init__(self):
        super(AttentionBranch, self).__init__()
        vgg = models.vgg16_bn(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features.children())[:33])  # same as regression branch

        self.attention = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()  # attention mask between 0 and 1
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = F.interpolate(x, size=(x.shape[2] * 8, x.shape[3] * 8), mode='bilinear', align_corners=False)  # upscale
        return x
