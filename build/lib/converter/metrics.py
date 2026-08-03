"""Quality metrics for converted images."""

from __future__ import annotations

import numpy as np

from converter.quantizer import indexed_to_rgb
from converter.types import IndexedImage, MetricResult, RGBImage


def _luminance(image: RGBImage) -> np.ndarray:
    return (
        0.2126 * image[:, :, 0].astype(np.float64)
        + 0.7152 * image[:, :, 1].astype(np.float64)
        + 0.0722 * image[:, :, 2].astype(np.float64)
    )


def compute_mse(original: RGBImage, converted: RGBImage) -> float:
    diff = original.astype(np.float64) - converted.astype(np.float64)
    return float(np.mean(diff * diff))


def compute_psnr(original: RGBImage, converted: RGBImage) -> float:
    mse = compute_mse(original, converted)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10((255.0**2) / mse))


def compute_ssim(original: RGBImage, converted: RGBImage) -> float:
    """Structural similarity index (simplified global SSIM)."""
    orig = original.astype(np.float64)
    conv = converted.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_x = orig.mean()
    mu_y = conv.mean()
    sigma_x = orig.var()
    sigma_y = conv.var()
    sigma_xy = float(np.mean((orig - mu_x) * (conv - mu_y)))

    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return float(num / den) if den != 0 else 1.0


def compute_mean_rgb_error(original: RGBImage, converted: RGBImage) -> float:
    diff = np.abs(original.astype(np.float64) - converted.astype(np.float64))
    return float(np.mean(diff))


def compute_mean_luminance_error(original: RGBImage, converted: RGBImage) -> float:
    return float(np.mean(np.abs(_luminance(original) - _luminance(converted))))


def count_colors_used(indexed: IndexedImage) -> int:
    return int(len(np.unique(indexed)))


def compute_histogram_coverage(original: RGBImage, converted: RGBImage) -> float:
    """Fraction of original luminance range represented in output."""
    orig_lum = _luminance(original)
    conv_lum = _luminance(converted)
    orig_range = orig_lum.max() - orig_lum.min()
    if orig_range == 0:
        return 1.0
    conv_range = conv_lum.max() - conv_lum.min()
    return float(min(1.0, conv_range / orig_range))


def compute_metrics(
    original: RGBImage,
    indexed: IndexedImage,
    palette_indices: list[int],
) -> MetricResult:
    """Compute all quality metrics."""
    converted = indexed_to_rgb(indexed, palette_indices)
    return MetricResult(
        mse=compute_mse(original, converted),
        psnr=compute_psnr(original, converted),
        ssim=compute_ssim(original, converted),
        mean_rgb_error=compute_mean_rgb_error(original, converted),
        mean_luminance_error=compute_mean_luminance_error(original, converted),
        colors_used=count_colors_used(indexed),
        histogram_coverage=compute_histogram_coverage(original, converted),
    )


def format_metrics(metrics: MetricResult) -> str:
    """Format metrics for CLI output."""
    lines = [
        f"MSE: {metrics.mse:.2f}",
        f"PSNR: {metrics.psnr:.2f} dB",
        f"SSIM: {metrics.ssim:.4f}",
        f"Mean RGB error: {metrics.mean_rgb_error:.2f}",
        f"Mean luminance error: {metrics.mean_luminance_error:.2f}",
        f"Colors used: {metrics.colors_used}",
        f"Histogram coverage: {metrics.histogram_coverage:.2%}",
    ]
    return "\n".join(lines)
