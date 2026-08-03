"""Edge detection for dithering control."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from converter.types import EdgeMask, RGBImage


def _luminance(image: RGBImage) -> NDArray[np.float64]:
    return (
        0.2126 * image[:, :, 0].astype(np.float64)
        + 0.7152 * image[:, :, 1].astype(np.float64)
        + 0.0722 * image[:, :, 2].astype(np.float64)
    )


def sobel_edges(image: RGBImage) -> EdgeMask:
    """Sobel edge magnitude normalized to 0-1."""
    lum = _luminance(image)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    padded = np.pad(lum, 1, mode="edge")
    gx = np.zeros_like(lum)
    gy = np.zeros_like(lum)
    for i in range(3):
        for j in range(3):
            gx += sobel_x[i, j] * padded[i : i + lum.shape[0], j : j + lum.shape[1]]
            gy += sobel_y[i, j] * padded[i : i + lum.shape[0], j : j + lum.shape[1]]

    magnitude = np.sqrt(gx * gx + gy * gy)
    max_val = magnitude.max()
    if max_val > 0:
        magnitude /= max_val
    return magnitude


def laplacian_edges(image: RGBImage) -> EdgeMask:
    """Laplacian edge magnitude normalized to 0-1."""
    lum = _luminance(image)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    padded = np.pad(lum, 1, mode="edge")
    result = np.zeros_like(lum)
    for i in range(3):
        for j in range(3):
            result += kernel[i, j] * padded[i : i + lum.shape[0], j : j + lum.shape[1]]
    magnitude = np.abs(result)
    max_val = magnitude.max()
    if max_val > 0:
        magnitude /= max_val
    return magnitude


def detect_edges(image: RGBImage, method: str = "sobel") -> EdgeMask:
    """Detect edges; returns normalized edge strength map."""
    if method == "laplacian":
        return laplacian_edges(image)
    return sobel_edges(image)


def stub_edge_map(image: RGBImage) -> EdgeMask:
    """Return zero edge map (MVP stub)."""
    return np.zeros((image.shape[0], image.shape[1]), dtype=np.float64)
