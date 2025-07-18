import os
import scipy.io as sio
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from utils.generate_density_map import generate_density_map


class ShanghaiTechDataset(Dataset):
    def __init__(self, root, split="train", part="B", transform=None):
        assert part in ["A", "B"], "part must be 'A' or 'B'"
        self.transform = transform
        self.split = split
        self.root = root
        self.part = part

        base_dir = os.path.join(root, f"Part_{part}")
        self.image_dir = os.path.join(base_dir, f"{split}_data", "images")
        self.gt_dir = os.path.join(base_dir, f"{split}_data", "ground_truth")

        self.samples = []
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.exists(self.gt_dir):
            raise FileNotFoundError(f"Ground truth directory not found: {self.gt_dir}")

        for img_name in os.listdir(self.image_dir):
            if img_name.endswith(".jpg"):
                img_path = os.path.join(self.image_dir, img_name)
                mat_path = os.path.join(
                    self.gt_dir, f"GT_{os.path.splitext(img_name)[0]}.mat"
                )
                if os.path.exists(mat_path):
                    self.samples.append((img_path, mat_path))

        print(f"✅ Loaded {len(self.samples)} samples from {self.image_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mat_path = self.samples[idx]

        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to read image at: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()

        mat = sio.loadmat(mat_path)
        points = mat["image_info"][0][0][0][0][0]  # shape: (N, 2)

        density_map = generate_density_map(img.shape[1:], points)
        density_map = torch.from_numpy(density_map).unsqueeze(0).float()

        return img, density_map, points
