"""Tests for adaptive dithering and edge detection."""

import numpy as np

from converter.dithering import AdaptiveDither, apply_dithering
from converter.edge_detect import detect_edges, laplacian_edges, sobel_edges
from converter.types import DitherMethod


def test_edge_detection() -> None:
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, 16:, :] = 255
    sobel = sobel_edges(img)
    lap = laplacian_edges(img)
    assert sobel.shape == (32, 32)
    assert lap.shape == (32, 32)
    assert sobel.max() <= 1.0
    assert detect_edges(img).shape == (32, 32)


def test_adaptive_dither() -> None:
    img = np.linspace(0, 255, 32 * 32, dtype=np.uint8).reshape(32, 32)
    img = np.stack([img, img, img], axis=2)
    palette = np.array([[0, 0, 0], [128, 128, 128], [192, 192, 192], [255, 255, 255]], dtype=np.uint8)
    result = AdaptiveDither().apply(img, palette)
    assert result.shape == (32, 32)
    assert result.max() < 4


def test_apply_dithering_adaptive() -> None:
    img = np.full((16, 16, 3), 100, dtype=np.uint8)
    palette = np.array([[0, 0, 0], [255, 255, 255], [100, 100, 100], [200, 200, 200]], dtype=np.uint8)
    result = apply_dithering(img, palette, DitherMethod.ADAPTIVE)
    assert result.shape == (16, 16)
