"""Atkinson error diffusion."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from converter.algorithms.base import DitherAlgorithm, find_nearest
from converter.types import EdgeMask, RGBImage

_OFFSETS = [
    (0, 1, 1 / 8),
    (0, 2, 1 / 8),
    (1, -1, 1 / 8),
    (1, 0, 1 / 8),
    (1, 1, 1 / 8),
    (2, 0, 1 / 8),
]


class Atkinson(DitherAlgorithm):
    name = "atkinson"

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
                old = working[y, x].copy()
                idx = find_nearest(old, palette)
                result[y, x] = idx
                new = palette[idx]
                error = (old - new) / 8.0
                working[y, x] = new

                for dy, dx, _ in _OFFSETS:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if edge_mask is not None and edge_mask[ny, nx] > edge_threshold:
                            continue
                        working[ny, nx] += error

        return result
