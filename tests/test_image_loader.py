"""Tests for image loader."""

from pathlib import Path

import numpy as np
from PIL import Image

from converter.image_loader import load_image


def test_load_png(tmp_path: Path) -> None:
    path = tmp_path / "test.png"
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    img.save(path)

    buffer = load_image(path)
    assert buffer.width == 10
    assert buffer.height == 10
    assert buffer.data[0, 0, 0] == 255
    assert buffer.data[0, 0, 1] == 0
