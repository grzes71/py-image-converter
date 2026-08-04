"""Ordered Bayer 8x8 dithering."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from converter.algorithms.base import DitherAlgorithm, find_nearest
from converter.types import EdgeMask, RGBImage

_MATRIX = (
    np.array(
        [
            [0, 32, 8, 40, 2, 34, 10, 42],
            [48, 16, 56, 24, 50, 18, 58, 26],
            [12, 44, 4, 36, 14, 46, 6, 38],
            [60, 28, 52, 20, 62, 30, 54, 22],
            [3, 35, 11, 43, 1, 33, 9, 41],
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47, 7, 39, 13, 45, 5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21],
        ],
        dtype=np.float64,
    )
    / 64.0
)


class Bayer8(DitherAlgorithm):
    name = "bayer8"

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
        mh, mw = _MATRIX.shape

        for y in range(h):
            for x in range(w):
                threshold = (_MATRIX[y % mh, x % mw] - 0.5) * 64
                adjusted = working[y, x] + threshold
                idx = find_nearest(adjusted, palette)
                result[y, x] = idx
                working[y, x] = palette[idx]

        return result
