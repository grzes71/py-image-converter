"""Load input images via Pillow (I/O only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from converter.types import ImageBuffer, RGBImage

SUPPORTED_EXTENSIONS = {".png", ".bmp", ".gif", ".jpg", ".jpeg"}


def load_image(path: str | Path) -> ImageBuffer:
    """Load PNG/BMP/GIF/JPG and return RGB uint8 array."""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{path.suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with Image.open(path) as img:
        rgb = img.convert("RGB")
        data: RGBImage = np.asarray(rgb, dtype=np.uint8)
    return ImageBuffer(data=data)


def save_rgb_image(path: str | Path, data: RGBImage) -> None:
    """Save an RGB image using Pillow."""
    Image.fromarray(data, mode="RGB").save(path)
