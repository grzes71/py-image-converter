"""Floyd-Steinberg error diffusion dithering."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from converter.algorithms.base import DitherAlgorithm, clamp_image, find_nearest
from converter.types import EdgeMask, RGBImage


class FloydSteinberg(DitherAlgorithm):
    """Floyd-Steinberg error diffusion."""

    name = "floyd"

    def apply(
        self,
        image: RGBImage,
        palette_rgb: NDArray[np.uint8],
        edge_mask: EdgeMask | None = None,
    ) -> NDArray[np.uint8]:
        h, w = image.shape[:2]
        working = image.astype(np.float64).copy()
        palette = palette_rgb.astype(np.float64)
        result = np.zeros((h, w), dtype=np.uint8)
        edge_threshold = 0.7

        for y in range(h):
            for x in range(w):
                if edge_mask is not None and edge_mask[y, x] > edge_threshold:
                    idx = find_nearest(working[y, x], palette)
                    result[y, x] = idx
                    working[y, x] = palette[idx]
                    continue

                old = working[y, x].copy()
                idx = find_nearest(old, palette)
                result[y, x] = idx
                new = palette[idx]
                working[y, x] = new
                error = old - new

                if x + 1 < w:
                    working[y, x + 1] += error * (7 / 16)
                if y + 1 < h:
                    if x > 0:
                        working[y + 1, x - 1] += error * (3 / 16)
                    working[y + 1, x] += error * (5 / 16)
                    if x + 1 < w:
                        working[y + 1, x + 1] += error * (1 / 16)

        clamp_image(working)
        return result
