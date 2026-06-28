"""Tests for palette selection."""

import numpy as np

from converter.palette import PopularitySelector, select_palette
from converter.types import PaletteMethod


def test_popularity_selector_returns_four_colors() -> None:
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:16, :, 0] = 255
    img[16:, :, 2] = 255
    selector = PopularitySelector()
    indices = selector.select(img, 4, background_index=0)
    assert len(indices) == 4
    assert indices[0] == 0
    assert len(set(indices)) == 4


def test_select_palette_kmeans() -> None:
    img = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    indices = select_palette(img, 4, PaletteMethod.KMEANS, background_index=0)
    assert len(indices) == 4
