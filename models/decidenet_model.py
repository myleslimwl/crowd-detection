import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from models.modules import RegressionBranch, AttentionBranch


class DetectionWrapper(nn.Module):
    def __init__(self):
        super(DetectionWrapper, self).__init__()
        self.detector = fasterrcnn_resnet50_fpn(pretrained=True)

        # Replace box predictor with FastRCNNPredictor (returns cls_logits, box_regression)
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

    def forward(self, images, targets=None):
        """
        Returns:
            - Training mode: dict of losses
            - Eval mode: list of detections
        """
        return self.detector(images, targets)


class DecideNet(nn.Module):
    def __init__(self):
        super(DecideNet, self).__init__()
        self.reg_branch = RegressionBranch()
        self.det_branch = DetectionWrapper()
        self.att_branch = AttentionBranch()

    def forward(self, images, targets=None):
        """
        Args:
            images (Tensor): [B, 3, H, W]
            targets (List[Dict]): Detection targets in training mode

        Returns:
            Tuple of:
                final_output (Tensor)
                reg_output (Tensor)
                det_output (List[Dict] or None)
                att_map (Tensor)
                det_losses (dict or empty)
        """
        reg_output = self.reg_branch(images)
        att_map = self.att_branch(images)

        if self.training:
            det_losses = self.det_branch(images, targets)
            det_output = None
        else:
            det_output = self.det_branch(images)
            det_losses = {}

        final_output = att_map * reg_output  # element-wise attention weighting

        return final_output, reg_output, det_output, att_map, det_losses
