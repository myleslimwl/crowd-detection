import os
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.generate_density_map import generate_density_map

class MallDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = os.path.join(root, split)
        self.img_dir = os.path.join(self.root, 'images')
        self.gt_dir = os.path.join(self.root, 'ground_truth')
        self.transform = transform

        self.image_files = [
            f for f in os.listdir(self.img_dir)
            if f.endswith('.jpg') or f.endswith('.png')
        ]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        gt_path = os.path.join(
            self.gt_dir,
            os.path.splitext(self.image_files[idx])[0] + '.h5'
        )

        img = Image.open(img_path).convert('RGB')
        with h5py.File(gt_path, 'r') as f:
            points = f['points'][:]

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(np.array(img).transpose((2, 0, 1)), dtype=torch.float32) / 255.

        density = generate_density_map(img.shape[1:], points)

        return img, density, points
