"""Ordered Bayer 4x4 dithering."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from converter.algorithms.base import DitherAlgorithm, find_nearest
from converter.types import EdgeMask, RGBImage

_MATRIX = (
    np.array(
        [
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5],
        ],
        dtype=np.float64,
    )
    / 16.0
)


class Bayer4(DitherAlgorithm):
    name = "bayer4"

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
