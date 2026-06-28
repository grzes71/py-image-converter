"""Preview generation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from converter.quantizer import indexed_to_rgb
from converter.resize import resize_nearest
from converter.types import IndexedImage, RGBImage


def upscale_nearest(image: RGBImage, factor: int) -> RGBImage:
    """Upscale image by integer factor using nearest neighbor."""
    h, w = image.shape[:2]
    return resize_nearest(image, w * factor, h * factor)


def side_by_side(left: RGBImage, right: RGBImage) -> RGBImage:
    """Place two equal-height images side by side."""
    if left.shape[0] != right.shape[0]:
        scale = left.shape[0] / right.shape[0]
        new_w = int(right.shape[1] * scale)
        right = resize_nearest(right, new_w, left.shape[0])
    gap = np.zeros((left.shape[0], 2, 3), dtype=np.uint8)
    return np.concatenate([left, gap, right], axis=1)


def error_heatmap(original: RGBImage, converted: RGBImage) -> RGBImage:
    """Generate per-pixel error heatmap (red = high error)."""
    diff = np.abs(original.astype(np.float64) - converted.astype(np.float64))
    error = np.mean(diff, axis=2)
    max_err = error.max() or 1.0
    normalized = (error / max_err * 255).astype(np.uint8)
    heatmap = np.zeros((*error.shape, 3), dtype=np.uint8)
    heatmap[:, :, 0] = normalized
    heatmap[:, :, 2] = 255 - normalized
    return heatmap


def generate_previews(
    original: RGBImage,
    indexed: IndexedImage,
    palette_indices: list[int],
    output_dir: str | Path,
    stem: str,
) -> list[Path]:
    """Generate preview images and return paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = indexed_to_rgb(indexed, palette_indices)
    paths: list[Path] = []

    x2 = upscale_nearest(converted, 2)
    p = output_dir / f"{stem}_x2.png"
    Image.fromarray(x2).save(p)
    paths.append(p)

    x4 = upscale_nearest(converted, 4)
    p = output_dir / f"{stem}_x4.png"
    Image.fromarray(x4).save(p)
    paths.append(p)

    comparison = side_by_side(original, converted)
    p = output_dir / f"{stem}_compare.png"
    Image.fromarray(comparison).save(p)
    paths.append(p)

    heatmap = error_heatmap(original, converted)
    p = output_dir / f"{stem}_error.png"
    Image.fromarray(heatmap).save(p)
    paths.append(p)

    return paths
