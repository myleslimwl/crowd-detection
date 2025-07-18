import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2

# ----------------------- RegNet -----------------------
class RegNet(nn.Module):
    def __init__(self):
        super(RegNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.output_layer = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        return self.output_layer(x)  # (N, 1, H', W')


# ----------------------- DetNet -----------------------
class DetNet(nn.Module):
    def __init__(self):
        super(DetNet, self).__init__()
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        # Freeze detection parameters
        for param in self.detector.parameters():
            param.requires_grad = False

    def forward(self, x):
        self.detector.eval()  # Ensure eval mode on each forward call
        with torch.no_grad():
            preds = self.detector(x)
        boxes = preds[0]['boxes']
        scores = preds[0]['scores']
        return boxes, scores


def convert_boxes_to_density(boxes, image_shape, sigma=4):
    """
    Convert detection boxes to density map.
    boxes: tensor of shape (N, 4)
    image_shape: (height, width)
    """
    density_map = np.zeros(image_shape[:2], dtype=np.float32)
    for box in boxes:
        x_center = int((box[0] + box[2]) / 2)
        y_center = int((box[1] + box[3]) / 2)
        cv2.circle(density_map, (x_center, y_center), sigma, 1, -1)
    return density_map


# ----------------------- AttNet -----------------------
class AttNet(nn.Module):
    def __init__(self):
        super(AttNet, self).__init__()
        self.att_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()  # Attention map between 0 and 1
        )

    def forward(self, x):
        return self.att_layers(x)  # (N, 1, H, W)


# ----------------------- DecideNet -----------------------
class DecideNet(nn.Module):
    def __init__(self):
        super(DecideNet, self).__init__()
        self.reg_branch = RegNet()
        self.det_branch = DetNet()
        self.att_branch = AttNet()

    def forward(self, x):
        N, C, H, W = x.shape
        device = x.device

        # Regression branch output
        reg_output = self.reg_branch(x)  # (N, 1, H', W')

        # Detection → density map (batch-agnostic for now)
        boxes, _ = self.det_branch(x)
        det_maps = []
        for i in range(N):
            det_map = convert_boxes_to_density(boxes, (H, W))
            det_map = torch.from_numpy(det_map).unsqueeze(0).float().to(device)  # (1, H, W)
            det_maps.append(det_map)
        det_output = torch.stack(det_maps).unsqueeze(1)  # (N, 1, H, W)

        # Attention map
        att_map = self.att_branch(x)  # (N, 1, H, W)

        # Resize regression output to match input size
        reg_output = F.interpolate(reg_output, size=(H, W), mode="bilinear", align_corners=False)

        # Fuse the two maps using the attention map
        final_output = att_map * det_output + (1 - att_map) * reg_output

        return final_output, reg_output, det_output, att_map