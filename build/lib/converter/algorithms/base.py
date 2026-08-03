"""Base dithering algorithm interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from converter.types import EdgeMask, RGBImage


class DitherAlgorithm(ABC):
    """Base class for dithering algorithms."""

    name: str = "base"

    @abstractmethod
    def apply(
        self,
        image: RGBImage,
        palette_rgb: NDArray[np.uint8],
        edge_mask: EdgeMask | None = None,
    ) -> NDArray[np.uint8]:
        """Return indexed image (values 0..palette_size-1)."""


def find_nearest(pixel: NDArray[np.float64], palette: NDArray[np.float64]) -> int:
    """Find nearest palette index for a single pixel."""
    diff = palette - pixel
    dist = np.sum(diff * diff, axis=1)
    return int(np.argmin(dist))


def clamp_image(img: NDArray[np.float64]) -> NDArray[np.float64]:
    """Clamp float image to 0-255."""
    return np.clip(img, 0, 255)


def diffuse_error(
    working: NDArray[np.float64],
    y: int,
    x: int,
    error: NDArray[np.float64],
    offsets: list[tuple[int, int, float]],
    edge_mask: EdgeMask | None,
    edge_threshold: float = 0.7,
) -> None:
    """Spread quantization error to neighbors."""
    h, w = working.shape[:2]
    for dy, dx, weight in offsets:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            if edge_mask is not None and edge_mask[ny, nx] > edge_threshold:
                continue
            working[ny, nx] += error * weight
