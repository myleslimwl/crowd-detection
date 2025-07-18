import os
import glob
import numpy as np
import scipy.io as sio
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils.generate_density_map import generate_density_map


class ShanghaiTechDataset(Dataset):
    def __init__(self, root='data/shanghaitech/Part_B', split='train'):
        self.root = root
        self.split = split
        self.img_dir = os.path.join(root, f'{split}_data', 'images')
        self.gt_dir = os.path.join(root, f'{split}_data', 'ground_truth')  # ✅ make sure this matches renamed folder

        image_paths = sorted(glob.glob(os.path.join(self.img_dir, '*.jpg')))
        self.image_paths = []
        self.gt_paths = []

        for img_path in image_paths:
            base_name = os.path.basename(img_path).replace('.jpg', '')
            gt_name = f'GT_{base_name}.mat'
            gt_path = os.path.join(self.gt_dir, gt_name)
            if os.path.exists(gt_path):
                self.image_paths.append(img_path)
                self.gt_paths.append(gt_path)
            else:
                print(f"⚠️ Missing GT file for {img_path}, skipping...")

        assert len(self.image_paths) > 0, "❌ No valid image-GT pairs found!"

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # ✅ Resize for training compatibility
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        gt_path = self.gt_paths[idx]

        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)

        mat = sio.loadmat(gt_path)
        points = mat["image_info"][0][0][0][0][0]  # shape (N, 2)

        img_h, img_w = img.size[1], img.size[0]  # height, width
        density = generate_density_map((img_h, img_w), points)
        density = cv2.resize(density, (256, 256))  # ✅ Matches final output size of model
        density_tensor = torch.from_numpy(density).unsqueeze(0).float()

        return img_tensor, density_tensor, points
