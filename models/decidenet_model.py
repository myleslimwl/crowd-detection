import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from models.modules import RegressionBranch, AttentionBranch

class DecideNet(nn.Module):
    def __init__(self):
        super(DecideNet, self).__init__()
        self.reg_branch = RegressionBranch()
        self.att_branch = AttentionBranch()

        self.det_branch = DetectionWrapper()

    def forward(self, images, targets=None):
        reg_output = self.reg_branch(images)
        att_map = self.att_branch(images)

        if self.training:
            det_output, det_losses = self.det_branch(images, targets)
            # Apply attention weighting (Hadamard product)
            final_output = att_map * det_output + (1 - att_map) * reg_output
            return final_output, reg_output, det_output, att_map, det_losses
        else:
            with torch.no_grad():
                det_output = self.det_branch(images)  # inference only
                final_output = att_map * det_output + (1 - att_map) * reg_output
                return final_output

class DetectionWrapper(nn.Module):
    def __init__(self):
        super(DetectionWrapper, self).__init__()
        self.detector = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)  # assume 2 classes: background + person

    def forward(self, images, targets=None):
        if self.training:
            loss_dict = self.detector(images, targets)
            # You could sum the losses or pass them along
            det_output = self.fake_density_from_detections(targets)  # optionally used for fusion
            return det_output, loss_dict
        else:
            predictions = self.detector(images)
            det_output = self.fake_density_from_detections(predictions)
            return det_output

    def fake_density_from_detections(self, detections):
        """
        This creates a dummy density map from bounding boxes, required to produce the final_output.
        In a real DecideNet, you'd convert detections to density maps properly.
        """
        batch_size = len(detections)
        return torch.zeros(batch_size, 1, 256, 256).to(next(self.parameters()).device)
