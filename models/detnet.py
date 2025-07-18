import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

class DetNet(nn.Module):
    def __init__(self, score_thresh=0.5):
        super(DetNet, self).__init__()

        # ✅ Load model with explicit weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.detector = fasterrcnn_resnet50_fpn(weights=weights)

        # Keep score threshold configurable
        self.score_thresh = score_thresh

    def forward(self, x):
        """
        Args:
            x (Tensor): batch of images [B, 3, H, W]
        Returns:
            density_maps: list of [H, W] np.arrays, one per image
        """
        # Assume eval mode is already set externally
        outputs = self.detector(x)

        # Convert detection boxes to density maps
        batch_density = []
        for output, img in zip(outputs, x):
            boxes = output["boxes"]
            scores = output["scores"]
            # Filter boxes by confidence threshold
            keep = scores > self.score_thresh
            kept_boxes = boxes[keep].cpu().numpy()
            image_shape = img.shape[1:]  # [C, H, W] → (H, W)
            density = convert_boxes_to_density(kept_boxes, image_shape)
            batch_density.append(density)

        return batch_density  # list of np.arrays

def convert_boxes_to_density(boxes, image_shape, sigma=4):
    """
    Converts detection bounding boxes into a synthetic density map.

    Args:
        boxes: Numpy array of shape [N, 4]
        image_shape: tuple (H, W)
        sigma: radius of Gaussian/filled circle

    Returns:
        density_map: np.array of shape (H, W)
    """
    density_map = np.zeros(image_shape, dtype=np.float32)
    for box in boxes:
        x_center = int((box[0] + box[2]) / 2)
        y_center = int((box[1] + box[3]) / 2)
        cv2.circle(density_map, (x_center, y_center), sigma, 1, -1)
    return density_map