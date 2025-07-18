import numpy as np
from scipy.ndimage import gaussian_filter
import torch


def generate_density_map(image_shape, points, sigma=15):
    """
    Generate a density map using Gaussian kernels for the given points.
    """
    # Ensure image_shape is a tuple of 2 integers
    if isinstance(image_shape, torch.Size):
        image_shape = tuple(image_shape)
    elif isinstance(image_shape, int) or len(image_shape) != 2:
        raise ValueError(f"Expected image_shape to be (H, W), but got: {image_shape}")

    h, w = image_shape
    density = np.zeros((h, w), dtype=np.float32)

    for point in points:
        x, y = int(min(w - 1, max(0, point[0]))), int(min(h - 1, max(0, point[1])))
        density[y, x] += 1

    density = gaussian_filter(density, sigma=sigma, mode='constant')
    return density