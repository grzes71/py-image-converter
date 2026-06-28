"""Tests for Floyd-Steinberg dithering."""

import numpy as np

from converter.algorithms.floyd import FloydSteinberg


def test_floyd_produces_valid_indices() -> None:
    img = np.full((16, 16, 3), 128, dtype=np.uint8)
    palette = np.array([[0, 0, 0], [255, 255, 255], [128, 128, 128], [64, 64, 64]], dtype=np.uint8)
    result = FloydSteinberg().apply(img, palette)
    assert result.shape == (16, 16)
    assert result.max() < 4
    assert result.min() >= 0
