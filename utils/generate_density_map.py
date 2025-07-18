import numpy as np
import scipy.spatial


def generate_density_map(image_shape, points, leafsize=2048):
    """
    Generate a density map using Gaussian kernels around annotated points.

    Args:
        image_shape (tuple): Shape of the image (height, width)
        points (ndarray): Array of (x, y) coordinates of people
        leafsize (int): Leafsize for KDTree (performance tuning)

    Returns:
        density (ndarray): Density map of shape (height, width)
    """
    h, w = image_shape
    density = np.zeros((h, w), dtype=np.float32)

    if len(points) == 0:
        return density

    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    distances, _ = tree.query(points, k=4)

    for i, point in enumerate(points):
        x = min(w - 1, max(0, int(round(point[0]))))
        y = min(h - 1, max(0, int(round(point[1]))))

        if len(points) > 3:
            sigma = np.mean(distances[i][1:4]) * 0.1
        else:
            sigma = np.average([h, w]) / 50.0  # fallback for sparse crowd

        sigma = max(1, sigma)
        gaussian_radius = int(3 * sigma)
        gaussian_size = 2 * gaussian_radius + 1

        x1 = max(0, x - gaussian_radius)
        y1 = max(0, y - gaussian_radius)
        x2 = min(w, x + gaussian_radius + 1)
        y2 = min(h, y + gaussian_radius + 1)

        dx1 = gaussian_radius - (x - x1)
        dy1 = gaussian_radius - (y - y1)
        dx2 = dx1 + (x2 - x1)
        dy2 = dy1 + (y2 - y1)

        gauss = generate_2d_gaussian(gaussian_size, sigma)
        density[y1:y2, x1:x2] += gauss[dy1:dy2, dx1:dx2]

    return density


def generate_2d_gaussian(size, sigma):
    """
    Create a 2D Gaussian kernel.

    Args:
        size (int): Kernel size (must be odd)
        sigma (float): Standard deviation of the Gaussian

    Returns:
        gaussian (ndarray): 2D Gaussian kernel
    """
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    gaussian = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return gaussian / np.sum(gaussian)
