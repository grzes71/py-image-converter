"""Color quantization strategies."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from converter import atari_palette
from converter.types import QuantizerMethod, RGBImage


def _palette_array(indices: list[int]) -> NDArray[np.float64]:
    return np.array([atari_palette.get_color(i).rgb for i in indices], dtype=np.float64)


def _luminance(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def quantize_nearest(image: RGBImage, palette_indices: list[int]) -> NDArray[np.uint8]:
    """Map each pixel to nearest palette color index (0..N-1)."""
    palette = _palette_array(palette_indices)
    flat = image.reshape(-1, 3).astype(np.float64)
    diff = flat[:, np.newaxis, :] - palette[np.newaxis, :, :]
    dist = np.sum(diff * diff, axis=2)
    return np.argmin(dist, axis=1).reshape(image.shape[:2]).astype(np.uint8)


def quantize_perceptual(image: RGBImage, palette_indices: list[int]) -> NDArray[np.uint8]:
    """Perceptual nearest with luminance weighting."""
    palette = _palette_array(palette_indices)
    pal_lum = _luminance(palette)
    flat = image.reshape(-1, 3).astype(np.float64)
    lum = _luminance(flat)
    diff = flat[:, np.newaxis, :] - palette[np.newaxis, :, :]
    rgb_dist = np.sum(diff * diff, axis=2)
    lum_diff = (lum[:, np.newaxis] - pal_lum[np.newaxis, :]) ** 2
    dist = rgb_dist + 2.0 * lum_diff
    return np.argmin(dist, axis=1).reshape(image.shape[:2]).astype(np.uint8)


def quantize_weighted(image: RGBImage, palette_indices: list[int]) -> NDArray[np.uint8]:
    """Weighted nearest emphasizing green channel (human vision)."""
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    palette = _palette_array(palette_indices)
    flat = image.reshape(-1, 3).astype(np.float64)
    diff = (flat[:, np.newaxis, :] - palette[np.newaxis, :, :]) * weights
    dist = np.sum(diff * diff, axis=2)
    return np.argmin(dist, axis=1).reshape(image.shape[:2]).astype(np.uint8)


def quantize_adaptive(
    image: RGBImage,
    palette_indices: list[int],
    variance_map: NDArray[np.float64] | None = None,
) -> NDArray[np.uint8]:
    """Adaptive nearest: perceptual in high-variance regions, nearest elsewhere."""
    if variance_map is None:
        return quantize_nearest(image, palette_indices)

    nearest = quantize_nearest(image, palette_indices)
    perceptual = quantize_perceptual(image, palette_indices)
    threshold = float(np.median(variance_map))
    mask = variance_map > threshold
    result = nearest.copy()
    result[mask] = perceptual[mask]
    return result


def quantize(
    image: RGBImage,
    palette_indices: list[int],
    method: QuantizerMethod,
    variance_map: NDArray[np.float64] | None = None,
) -> NDArray[np.uint8]:
    """Quantize image to palette indices."""
    if method == QuantizerMethod.PERCEPTUAL:
        return quantize_perceptual(image, palette_indices)
    if method == QuantizerMethod.WEIGHTED:
        return quantize_weighted(image, palette_indices)
    if method == QuantizerMethod.ADAPTIVE:
        return quantize_adaptive(image, palette_indices, variance_map)
    return quantize_nearest(image, palette_indices)


def indexed_to_rgb(indexed: NDArray[np.uint8], palette_indices: list[int]) -> RGBImage:
    """Convert indexed image back to RGB using palette."""
    palette = np.array([atari_palette.get_color(i).rgb for i in palette_indices], dtype=np.uint8)
    return palette[indexed]
