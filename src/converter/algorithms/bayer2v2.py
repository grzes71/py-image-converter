"""Bayer 2x2 variant 2 ordered dithering (from dither-lab).

Alternative threshold arrangement for the 2x2 Bayer matrix.
Thresholds are normalized to 0-1 range from the original 0-255.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from converter.algorithms.base import DitherAlgorithm, find_nearest
from converter.types import EdgeMask, RGBImage

_MATRIX = np.array(
    [
        [51, 102],
        [204, 153],
    ],
    dtype=np.float64,
) / 255.0


class Bayer2v2(DitherAlgorithm):
    """Bayer 2x2 variant 2 — alternative threshold arrangement."""

    name = "atari_bayer2v2"

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
