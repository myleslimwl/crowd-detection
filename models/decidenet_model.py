import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class RegressionBranch(nn.Module):
    def __init__(self):
        super(RegressionBranch, self).__init__()
        resnet = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # remove avgpool & fc

        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)  # output density map
        )

    def forward(self, x):
        features = self.backbone(x)
        density_map = self.decoder(features)
        return density_map


class AttentionBranch(nn.Module):
    def __init__(self):
        super(AttentionBranch, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, reg_map, det_map):
        x = torch.cat([reg_map, det_map], dim=1)
        attention = self.attention(x)
        return attention


class DetectionBranch(nn.Module):
    def __init__(self):
        super(DetectionBranch, self).__init__()
        self.detector = fasterrcnn_resnet50_fpn(pretrained=True)
        self.detector.train()

    def forward(self, images, targets=None):
        if self.training:
            return self.detector(images, targets)
        else:
            return self.detector(images)


class DecideNet(nn.Module):
    def __init__(self):
        super(DecideNet, self).__init__()
        self.reg_branch = RegressionBranch()
        self.att_branch = AttentionBranch()
        self.det_branch = DetectionBranch()

    def forward(self, images, targets=None):
        # Regression output
        reg_output = self.reg_branch(images)

        # Detection output: expects list of dicts if training
        if self.training and targets is not None:
            det_output = self.det_branch(images, targets)
        else:
            det_output = self.det_branch(images)

        # Create a detection density map (simulated here)
        # Note: You should define a function to convert detection boxes to heatmaps
        det_density_map = torch.zeros_like(reg_output)

        # Attention output
        att_map = self.att_branch(reg_output, det_density_map)

        # Final output using attention map
        final_output = att_map * reg_output + (1 - att_map) * det_density_map

        return final_output, reg_output, det_output, att_map
