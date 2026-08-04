"""Tests for resize module."""

import numpy as np

from converter.resize import resize_bilinear, resize_image, resize_lanczos, resize_nearest
from converter.types import ResizeMethod


def _make_gradient(width: int, height: int) -> np.ndarray:
    row = np.linspace(0, 255, width, dtype=np.uint8)
    return np.tile(row, (height, 1))[:, :, np.newaxis].repeat(3, axis=2)


def test_resize_nearest_dimensions() -> None:
    img = _make_gradient(100, 50)
    result = resize_nearest(img, 160, 192)
    assert result.shape == (192, 160, 3)


def test_resize_lanczos_dimensions() -> None:
    img = _make_gradient(100, 50)
    result = resize_lanczos(img, 160, 192)
    assert result.shape == (192, 160, 3)


def test_resize_registry() -> None:
    img = _make_gradient(20, 20)
    result = resize_image(img, 10, 10, ResizeMethod.BILINEAR)
    assert result.shape == (10, 10, 3)
