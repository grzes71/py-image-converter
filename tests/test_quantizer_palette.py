"""Tests for quantizer modes and additional palette selectors."""

import numpy as np

from converter.palette import MedianCutSelector, OctreeSelector, PerceptualSelector
from converter.quantizer import quantize, quantize_adaptive, quantize_perceptual
from converter.types import QuantizerMethod


def test_quantizer_modes() -> None:
    img = np.random.default_rng(1).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    palette = list(range(4))
    nearest = quantize(img, palette, QuantizerMethod.NEAREST)
    perceptual = quantize(img, palette, QuantizerMethod.PERCEPTUAL)
    weighted = quantize(img, palette, QuantizerMethod.WEIGHTED)
    assert nearest.shape == (32, 32)
    assert perceptual.shape == (32, 32)
    assert weighted.shape == (32, 32)


def test_adaptive_quantizer() -> None:
    img = np.random.default_rng(2).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    variance = np.random.default_rng(3).random((16, 16))
    result = quantize_adaptive(img, list(range(4)), variance)
    assert result.shape == (16, 16)


def test_palette_selectors() -> None:
    img = np.random.default_rng(4).integers(0, 256, (48, 48, 3), dtype=np.uint8)
    for selector in [MedianCutSelector(), OctreeSelector(), PerceptualSelector()]:
        indices = selector.select(img, 4, background_index=0)
        assert len(indices) == 4
        assert indices[0] == 0
