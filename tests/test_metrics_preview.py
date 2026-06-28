"""Tests for metrics and preview."""

import numpy as np

from converter.metrics import compute_metrics, compute_ssim
from converter.preview import error_heatmap, side_by_side, upscale_nearest


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
