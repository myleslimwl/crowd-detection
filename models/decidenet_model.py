import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from models.modules import RegressionBranch, AttentionBranch

class DecideNet(nn.Module):
    def __init__(self):
        super(DecideNet, self).__init__()
        self.reg_branch = RegressionBranch()
        self.det_branch = DetectionBranch()
        self.att_branch = AttentionBranch()

    def forward(self, images, targets=None):
        reg_output = self.reg_branch(images)
        det_output = self.det_branch(images, targets)
        attention_map = self.att_branch(images)

        final_output = attention_map * det_output + (1 - attention_map) * reg_output
        return final_output, reg_output, det_output, attention_map


class DetectionBranch(nn.Module):
    def __init__(self):
        super(DetectionBranch, self).__init__()
        self.detector = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = nn.Sequential(
            nn.Linear(in_features, 2)  # 1 class + background
        )

    def forward(self, images, targets=None):
        return self.detector(images, targets)
