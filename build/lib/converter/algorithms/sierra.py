"""Sierra error diffusion."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from converter.algorithms.base import DitherAlgorithm, diffuse_error, find_nearest
from converter.types import EdgeMask, RGBImage

_OFFSETS = [
    (0, 1, 5 / 32),
    (0, 2, 3 / 32),
    (1, -2, 2 / 32),
    (1, -1, 4 / 32),
    (1, 0, 5 / 32),
    (1, 1, 4 / 32),
    (1, 2, 2 / 32),
    (2, -1, 2 / 32),
    (2, 0, 3 / 32),
    (2, 1, 2 / 32),
]


class Sierra(DitherAlgorithm):
    name = "sierra"

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
