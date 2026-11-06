"""Tests for metrics and preview."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from converter.metrics import compute_mse, compute_metrics, compute_ssim
from converter.preview import error_heatmap, generate_previews, side_by_side, upscale_nearest
from converter.quantizer import indexed_to_rgb


def test_metrics_computation() -> None:
    original = np.full((32, 32, 3), 128, dtype=np.uint8)
    indexed = np.zeros((32, 32), dtype=np.uint8)
    palette = [0, 14, 28, 254]
    metrics = compute_metrics(original, indexed, list(range(len(palette))))
    assert metrics.mse >= 0
    assert metrics.colors_used >= 1
    assert 0 <= metrics.ssim <= 1


def test_ssim_identical() -> None:
    img = np.full((16, 16, 3), 100, dtype=np.uint8)
    assert compute_ssim(img, img) == 1.0


def test_preview_helpers() -> None:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert upscale_nearest(img, 2).shape == (20, 20, 3)
    combined = side_by_side(img, img)
    assert combined.shape[1] == 22
    heat = error_heatmap(img, img)
    assert heat.shape == (10, 10, 3)


def test_metrics_use_the_given_palette() -> None:
    """The palette argument must select the colours the error is measured against (B2)."""
    original = np.full((8, 8, 3), 200, dtype=np.uint8)
    indexed = np.ones((8, 8), dtype=np.uint8)

    dark = compute_metrics(original, indexed, [0, 1])
    light = compute_metrics(original, indexed, [0, 15])
    assert dark.mse != light.mse

    reconstructed = indexed_to_rgb(indexed, [0, 15])
    assert light.mse == pytest.approx(compute_mse(original, reconstructed))


def test_previews_use_the_given_palette(tmp_path: Path) -> None:
    """Preview colours must come from the palette argument (B3)."""
    source = np.zeros((4, 4, 3), dtype=np.uint8)
    indexed = np.zeros((4, 4), dtype=np.uint8)

    black = generate_previews(source, indexed, [0], tmp_path / "black", "screen")
    purple = generate_previews(source, indexed, [0x65], tmp_path / "purple", "screen")
    assert len(black) == 4
    assert len(purple) == 4

    for dark_path, light_path in zip(black, purple, strict=True):
        assert dark_path.read_bytes() != light_path.read_bytes()
