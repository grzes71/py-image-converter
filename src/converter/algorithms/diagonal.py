"""Diagonal error diffusion dithering (from dither-lab).

Unique kernel that sends the largest share (6/16) to the
diagonal neighbour (x+1, y+1), producing a distinctive
directional texture. Divisor = 16.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from converter.algorithms.base import DitherAlgorithm, diffuse_error, find_nearest
from converter.types import EdgeMask, RGBImage

_OFFSETS = [
    (0, 1, 5 / 16),
    (1, -1, 2 / 16),
    (1, 0, 3 / 16),
    (1, 1, 6 / 16),
]


class Diagonal(DitherAlgorithm):
    """Diagonal error diffusion — strongest weight goes to lower-right diagonal."""

    name = "atari_diagonal"

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
