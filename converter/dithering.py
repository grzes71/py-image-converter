"""Dithering factory and adaptive dithering."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from converter.algorithms.atkinson import Atkinson
from converter.algorithms.bayer2 import Bayer2
from converter.algorithms.bayer4 import Bayer4
from converter.algorithms.bayer8 import Bayer8
from converter.algorithms.base import DitherAlgorithm
from converter.algorithms.burkes import Burkes
from converter.algorithms.floyd import FloydSteinberg
from converter.algorithms.jarvis import Jarvis
from converter.algorithms.sierra import Sierra
from converter.algorithms.sierra_lite import SierraLite
from converter.algorithms.stucki import Stucki
from converter.histogram import compute_gradient_magnitude, compute_local_variance
from converter.quantizer import quantize_nearest
from converter.types import DitherMethod, EdgeMask, RGBImage


class NoDither(DitherAlgorithm):
    """Quantize without dithering."""

    name = "none"

    def apply(
        self,
        image: RGBImage,
        palette_rgb: NDArray[np.uint8],
        edge_mask: EdgeMask | None = None,
    ) -> NDArray[np.uint8]:
        palette_indices = list(range(len(palette_rgb)))
        return quantize_nearest(image, palette_indices)


_DITHER_REGISTRY: dict[DitherMethod, DitherAlgorithm] = {
    DitherMethod.NONE: NoDither(),
    DitherMethod.FLOYD: FloydSteinberg(),
    DitherMethod.JARVIS: Jarvis(),
    DitherMethod.STUCKI: Stucki(),
    DitherMethod.SIERRA: Sierra(),
    DitherMethod.SIERRA_LITE: SierraLite(),
    DitherMethod.BURKES: Burkes(),
    DitherMethod.ATKINSON: Atkinson(),
    DitherMethod.BAYER2: Bayer2(),
    DitherMethod.BAYER4: Bayer4(),
    DitherMethod.BAYER8: Bayer8(),
}


def get_dither_algorithm(method: DitherMethod) -> DitherAlgorithm:
    """Return dithering algorithm for the given method."""
    if method == DitherMethod.ADAPTIVE:
        return AdaptiveDither()
    return _DITHER_REGISTRY.get(method, FloydSteinberg())


class AdaptiveDither(DitherAlgorithm):
    """Tile-based adaptive dithering based on local image properties."""

    name = "adaptive"
    tile_size = 16

    def _classify_region(self, variance: float, gradient: float) -> DitherMethod:
        if gradient > 0.35:
            return DitherMethod.NONE
        if variance < 100:
            return DitherMethod.BAYER4
        if variance > 800:
            return DitherMethod.FLOYD
        return DitherMethod.ATKINSON

    def apply(
        self,
        image: RGBImage,
        palette_rgb: NDArray[np.uint8],
        edge_mask: EdgeMask | None = None,
    ) -> NDArray[np.uint8]:
        h, w = image.shape[:2]
        variance_map = compute_local_variance(image, self.tile_size)
        gradient_map = compute_gradient_magnitude(image)
        result = np.zeros((h, w), dtype=np.uint8)

        for y in range(0, h, self.tile_size):
            for x in range(0, w, self.tile_size):
                tile = image[y : y + self.tile_size, x : x + self.tile_size]
                var_tile = variance_map[y : y + self.tile_size, x : x + self.tile_size]
                grad_tile = gradient_map[y : y + self.tile_size, x : x + self.tile_size]
                method = self._classify_region(float(np.mean(var_tile)), float(np.mean(grad_tile)))
                algo = _DITHER_REGISTRY.get(method, FloydSteinberg())
                mask_tile = edge_mask[y : y + self.tile_size, x : x + self.tile_size] if edge_mask is not None else None
                tile_result = algo.apply(tile, palette_rgb, mask_tile)
                result[y : y + self.tile_size, x : x + self.tile_size] = tile_result

        return result


def apply_dithering(
    image: RGBImage,
    palette_rgb: NDArray[np.uint8],
    method: DitherMethod,
    edge_mask: EdgeMask | None = None,
) -> NDArray[np.uint8]:
    """Apply dithering and return indexed image."""
    algo = get_dither_algorithm(method)
    return algo.apply(image, palette_rgb, edge_mask)
