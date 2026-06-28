"""Tests for dithering algorithms."""

import numpy as np

from converter.algorithms import (
    Atkinson,
    Bayer2,
    Bayer4,
    Bayer8,
    Burkes,
    Jarvis,
    Sierra,
    SierraLite,
    Stucki,
)


def _run_algo(algo, size: int = 8) -> None:
    img = np.linspace(0, 255, size * size, dtype=np.uint8).reshape(size, size)
    img = np.stack([img, img, img], axis=2)
    palette = np.array([[0, 0, 0], [85, 85, 85], [170, 170, 170], [255, 255, 255]], dtype=np.uint8)
    result = algo.apply(img, palette)
    assert result.shape == (size, size)
    assert result.max() < 4


def test_all_dither_algorithms() -> None:
    for algo in [Atkinson(), Bayer2(), Bayer4(), Bayer8(), Burkes(), Jarvis(), Sierra(), SierraLite(), Stucki()]:
        _run_algo(algo)
