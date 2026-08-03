"""Indexed PNG export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from converter.types import IndexedImage, RGBImage


def export_png(indexed: IndexedImage, palette_rgb: RGBImage, path: str | Path) -> None:
    """Export indexed image as PNG with embedded palette."""
    path = Path(path)
    img = Image.fromarray(indexed, mode="P")
    palette = np.zeros(256 * 3, dtype=np.uint8)
    flat_palette = palette_rgb.reshape(-1)
    palette[: len(flat_palette)] = flat_palette
    img.putpalette(palette.tolist())
    img.save(path)
