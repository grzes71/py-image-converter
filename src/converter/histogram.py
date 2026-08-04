"""Histogram analysis for palette selection and adaptive dithering."""

from __future__ import annotations

import colorsys

import numpy as np
from numpy.typing import NDArray

from converter.types import HistogramReport, RGBImage


def _luminance_channel(image: RGBImage) -> NDArray[np.float64]:
    return (
        0.2126 * image[:, :, 0].astype(np.float64)
        + 0.7152 * image[:, :, 1].astype(np.float64)
        + 0.0722 * image[:, :, 2].astype(np.float64)
    )


def compute_rgb_histogram(image: RGBImage) -> dict[str, NDArray[np.int64]]:
    """Compute per-channel RGB histograms (256 bins)."""
    return {
        "r": np.bincount(image[:, :, 0].ravel(), minlength=256),
        "g": np.bincount(image[:, :, 1].ravel(), minlength=256),
        "b": np.bincount(image[:, :, 2].ravel(), minlength=256),
    }


def compute_hsv_histogram(image: RGBImage) -> dict[str, NDArray[np.int64]]:
    """Compute HSV histograms."""
    h_bins = np.zeros(36, dtype=np.int64)
    s_bins = np.zeros(101, dtype=np.int64)
    v_bins = np.zeros(101, dtype=np.int64)
    flat = image.reshape(-1, 3)
    for r, g, b in flat:
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        h_bins[min(int(h * 36), 35)] += 1
        s_bins[min(int(s * 100), 100)] += 1
        v_bins[min(int(v * 100), 100)] += 1
    return {"h": h_bins, "s": s_bins, "v": v_bins}


def compute_luminance_histogram(image: RGBImage) -> NDArray[np.int64]:
    """Compute luminance histogram (256 bins)."""
    lum = _luminance_channel(image).astype(np.uint8)
    return np.bincount(lum.ravel(), minlength=256)


def compute_local_variance(image: RGBImage, tile_size: int = 8) -> NDArray[np.float64]:
    """Compute local variance map using tile-based analysis."""
    lum = _luminance_channel(image)
    h, w = lum.shape
    variance = np.zeros((h, w), dtype=np.float64)
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = lum[y : y + tile_size, x : x + tile_size]
            var = float(np.var(tile))
            variance[y : y + tile_size, x : x + tile_size] = var
    return variance


def compute_gradient_magnitude(image: RGBImage) -> NDArray[np.float64]:
    """Compute gradient magnitude using Sobel-like operators."""
    lum = _luminance_channel(image)
    gx = np.zeros_like(lum)
    gy = np.zeros_like(lum)
    gx[:, 1:-1] = lum[:, 2:] - lum[:, :-2]
    gy[1:-1, :] = lum[2:, :] - lum[:-2, :]
    return np.sqrt(gx * gx + gy * gy)


def compute_detail_score(image: RGBImage) -> float:
    """Overall detail score based on gradient energy."""
    grad = compute_gradient_magnitude(image)
    return float(np.mean(grad))


def analyze_histogram(image: RGBImage, full: bool = False) -> HistogramReport:
    """Analyze image histograms and local properties."""
    report = HistogramReport(
        rgb_histogram=compute_rgb_histogram(image),
        luminance_histogram=compute_luminance_histogram(image),
        detail_score=compute_detail_score(image),
    )
    if full:
        report.hsv_histogram = compute_hsv_histogram(image)
        report.local_variance = compute_local_variance(image)
        report.gradient_magnitude = compute_gradient_magnitude(image)
    return report
