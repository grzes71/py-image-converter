"""Non-rectangular 8x8 ordered dithering (from dither-lab).

An 8x8 threshold matrix where each row is a shifted version
of the same 8-value sequence rather than being independent.
This produces a distinctive diagonal pattern.
Thresholds are normalized to 0-1 range from the original 0-255.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from converter.algorithms.base import DitherAlgorithm, find_nearest
from converter.types import EdgeMask, RGBImage

_MATRIX = np.array(
    [
        [113, 141, 85, 226, 56, 198, 28, 170],
        [198, 28, 170, 113, 141, 85, 226, 56],
        [85, 226, 56, 198, 28, 170, 113, 141],
        [170, 113, 141, 85, 226, 56, 198, 28],
        [56, 198, 28, 170, 113, 141, 85, 226],
        [141, 85, 226, 56, 198, 28, 170, 113],
        [28, 170, 113, 141, 85, 226, 56, 198],
        [226, 56, 198, 28, 170, 113, 141, 85],
    ],
    dtype=np.float64,
) / 255.0


class NonRectangular8(DitherAlgorithm):
    """Non-rectangular 8x8 ordered dithering — shifted-row pattern."""

    name = "atari_non_rectangular8"

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
