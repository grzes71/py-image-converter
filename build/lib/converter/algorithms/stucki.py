"""Stucki error diffusion."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from converter.algorithms.base import DitherAlgorithm, diffuse_error, find_nearest
from converter.types import EdgeMask, RGBImage

_OFFSETS = [
    (0, 1, 8 / 42),
    (0, 2, 4 / 42),
    (1, -2, 2 / 42),
    (1, -1, 4 / 42),
    (1, 0, 8 / 42),
    (1, 1, 4 / 42),
    (1, 2, 2 / 42),
    (2, -2, 1 / 42),
    (2, -1, 2 / 42),
    (2, 0, 4 / 42),
    (2, 1, 2 / 42),
    (2, 2, 1 / 42),
]


class Stucki(DitherAlgorithm):
    name = "stucki"

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

        for y in range(h):
            for x in range(w):
                old = working[y, x].copy()
                idx = find_nearest(old, palette)
                result[y, x] = idx
                new = palette[idx]
                working[y, x] = new
                diffuse_error(working, y, x, old - new, _OFFSETS, edge_mask)

        return result
