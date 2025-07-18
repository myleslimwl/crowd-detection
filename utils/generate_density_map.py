import numpy as np
import cv2

def generate_density_map(image_shape, points, sigma=4):
    """
    Create a density map using 2D Gaussian kernels centered on annotated head points.

    Args:
        image_shape: (H, W) of the output density map
        points: List of (x, y) coordinates (float or int)
        sigma: Standard deviation for the Gaussian blob

    Returns:
        A density map as a NumPy array of shape (H, W)
    """
    h, w = image_shape
    density = np.zeros((h, w), dtype=np.float32)

    for point in points:
        x, y = int(point[0]), int(point[1])
        if x >= w or y >= h or x < 0 or y < 0:
            continue
        # Create a 2D Gaussian
        cv2.circle(density, (x, y), sigma, 1, -1)

    return density