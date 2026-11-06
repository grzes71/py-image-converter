"""Tests for the individual pipeline stages.

Every stage is exercised on its own with a small synthetic image, and `convert_image` is verified
to be nothing more than their composition. This keeps the stages independently testable before the
`ConversionEngine` (T1.3) starts caching them.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from converter.dithering import apply_dithering
from converter.histogram import compute_local_variance
from converter.image_loader import load_image
from converter.metrics import compute_metrics
from converter.palette import palette_rgb
from converter.pipeline import (
    build_metrics,
    choose_palette,
    compute_edges,
    convert_image,
    prepare_image,
    render_indexed,
)
from converter.quantizer import quantize, quantize_adaptive
from converter.resize import resize_image
from converter.sharpen import apply_sharpen
from converter.types import ConversionConfig, DitherMethod, PaletteMethod, QuantizerMethod

REFERENCE_IMAGE = Path(__file__).parent / "fixtures" / "reference.png"


def _gradient(width: int = 64, height: int = 48) -> np.ndarray:
    """Build a small deterministic RGB gradient (uint8, H x W x 3)."""
    columns = np.arange(width, dtype=np.uint16)
    rows = np.arange(height, dtype=np.uint16)
    grid_x, grid_y = np.meshgrid(columns, rows)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = (grid_x * 4) % 256
    image[:, :, 1] = (grid_y * 5) % 256
    image[:, :, 2] = ((grid_x + grid_y) * 3) % 256
    return image


def _prepared(config: ConversionConfig) -> np.ndarray:
    """Run only the preparation stage on the synthetic gradient."""
    return prepare_image(_gradient(), config)


def test_prepare_image_resizes_to_the_configured_size() -> None:
    prepared = _prepared(ConversionConfig(width=32, height=24))
    assert prepared.shape == (24, 32, 3)
    assert prepared.dtype == np.uint8


def test_prepare_image_matches_resize_then_sharpen() -> None:
    source = _gradient()
    config = ConversionConfig(width=32, height=24, sharpen=True, sharpen_strength=1.5)
    expected = apply_sharpen(
        resize_image(source, config.width, config.height, config.resize),
        config.sharpen,
        config.sharpen_strength,
    )
    np.testing.assert_array_equal(prepare_image(source, config), expected)


def test_prepare_image_sharpening_changes_the_result() -> None:
    source = _gradient()
    plain = prepare_image(source, ConversionConfig(width=32, height=24))
    sharpened = prepare_image(
        source,
        ConversionConfig(width=32, height=24, sharpen=True, sharpen_strength=2.0),
    )
    assert not np.array_equal(plain, sharpened)


def test_prepare_image_does_not_modify_the_source() -> None:
    source = _gradient()
    original = source.copy()
    prepare_image(source, ConversionConfig(width=32, height=24, sharpen=True))
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("method", [PaletteMethod.MEDIAN_CUT, PaletteMethod.POPULARITY])
def test_choose_palette_returns_unique_indices(method: PaletteMethod) -> None:
    prepared = _prepared(ConversionConfig(width=32, height=24))
    palette = choose_palette(prepared, ConversionConfig(colors=4, palette=method))
    assert len(palette) == 4
    assert len(set(palette)) == 4
    assert all(0 <= index <= 255 for index in palette)


def test_choose_palette_puts_the_background_index_first() -> None:
    prepared = _prepared(ConversionConfig(width=32, height=24))
    palette = choose_palette(prepared, ConversionConfig(colors=4, background_index=6))
    assert palette[0] == 6
    assert len(set(palette)) == 4


def test_choose_palette_is_deterministic() -> None:
    prepared = _prepared(ConversionConfig(width=32, height=24))
    config = ConversionConfig(colors=3, palette=PaletteMethod.MEDIAN_CUT)
    assert choose_palette(prepared, config) == choose_palette(prepared.copy(), config)


def test_compute_edges_returns_a_non_negative_mask() -> None:
    prepared = _prepared(ConversionConfig(width=32, height=24))
    mask = compute_edges(prepared)
    assert mask.shape == (24, 32)
    assert mask.dtype == np.float64
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) > 0.0


def test_compute_edges_of_a_flat_image_is_zero() -> None:
    flat = np.full((16, 16, 3), 128, dtype=np.uint8)
    np.testing.assert_array_equal(compute_edges(flat), np.zeros((16, 16)))


def test_render_indexed_without_dither_matches_quantize() -> None:
    config = ConversionConfig(
        width=32,
        height=24,
        dither=DitherMethod.NONE,
        palette=PaletteMethod.MEDIAN_CUT,
    )
    image = _prepared(config)
    palette = choose_palette(image, config)
    indexed = render_indexed(image, palette, compute_edges(image), config)

    np.testing.assert_array_equal(indexed, quantize(image, palette, config.quantizer))
    assert indexed.shape == (24, 32)
    assert int(indexed.max()) < len(palette)


def test_render_indexed_adaptive_quantizer_receives_the_variance_map() -> None:
    config = ConversionConfig(
        width=32,
        height=24,
        dither=DitherMethod.NONE,
        quantizer=QuantizerMethod.ADAPTIVE,
        palette=PaletteMethod.MEDIAN_CUT,
    )
    image = _prepared(config)
    palette = choose_palette(image, config)
    edge_mask = compute_edges(image)

    indexed = render_indexed(image, palette, edge_mask, config)
    expected = quantize_adaptive(image, palette, compute_local_variance(image))
    np.testing.assert_array_equal(indexed, expected)

    nearest = render_indexed(
        image, palette, edge_mask, replace(config, quantizer=QuantizerMethod.NEAREST)
    )
    assert not np.array_equal(indexed, nearest)


def test_render_indexed_with_a_dither_matches_apply_dithering() -> None:
    config = ConversionConfig(
        width=32,
        height=24,
        dither=DitherMethod.BAYER2,
        palette=PaletteMethod.MEDIAN_CUT,
    )
    image = _prepared(config)
    palette = choose_palette(image, config)
    edge_mask = compute_edges(image)

    indexed = render_indexed(image, palette, edge_mask, config)
    expected = apply_dithering(image, palette_rgb(palette), config.dither, edge_mask)
    np.testing.assert_array_equal(indexed, expected)


def test_render_indexed_ignores_the_quantizer_when_dithering() -> None:
    config = ConversionConfig(
        width=32,
        height=24,
        dither=DitherMethod.BAYER2,
        palette=PaletteMethod.MEDIAN_CUT,
    )
    image = _prepared(config)
    palette = choose_palette(image, config)
    edge_mask = compute_edges(image)

    plain = render_indexed(image, palette, edge_mask, config)
    adaptive = render_indexed(
        image,
        palette,
        edge_mask,
        replace(config, quantizer=QuantizerMethod.ADAPTIVE),
    )
    np.testing.assert_array_equal(plain, adaptive)


def test_build_metrics_matches_compute_metrics() -> None:
    config = ConversionConfig(
        width=32,
        height=24,
        dither=DitherMethod.NONE,
        palette=PaletteMethod.MEDIAN_CUT,
    )
    image = _prepared(config)
    palette = choose_palette(image, config)
    indexed = render_indexed(image, palette, compute_edges(image), config)

    expected = compute_metrics(image, indexed, palette)
    metrics = build_metrics(image, indexed, palette)
    assert metrics.mse == pytest.approx(expected.mse)
    assert metrics.psnr == pytest.approx(expected.psnr)
    assert metrics.ssim == pytest.approx(expected.ssim)
    assert metrics.mean_rgb_error == pytest.approx(expected.mean_rgb_error)
    assert metrics.mean_luminance_error == pytest.approx(expected.mean_luminance_error)
    assert metrics.colors_used == expected.colors_used
    assert metrics.histogram_coverage == pytest.approx(expected.histogram_coverage)


def test_convert_image_is_the_composition_of_the_stages() -> None:
    """`convert_image` must do nothing beyond chaining the stage functions."""
    config = ConversionConfig(
        dither=DitherMethod.BAYER2,
        palette=PaletteMethod.MEDIAN_CUT,
        metrics=True,
    )
    result = convert_image(REFERENCE_IMAGE, config)

    prepared = prepare_image(load_image(REFERENCE_IMAGE).data, config)
    palette = choose_palette(prepared, config)
    edge_mask = compute_edges(prepared)

    np.testing.assert_array_equal(result.source_resized, prepared)
    assert result.palette_indices == palette
    np.testing.assert_array_equal(result.palette_rgb, palette_rgb(palette))
    expected_indexed = render_indexed(prepared, palette, edge_mask, config)
    np.testing.assert_array_equal(result.indexed, expected_indexed)

    assert result.metrics is not None
    expected_metrics = build_metrics(prepared, expected_indexed, palette)
    assert result.metrics.mse == pytest.approx(expected_metrics.mse)
    assert result.metrics.ssim == pytest.approx(expected_metrics.ssim)


def test_convert_image_without_metrics_leaves_the_metrics_empty() -> None:
    config = ConversionConfig(dither=DitherMethod.NONE, palette=PaletteMethod.MEDIAN_CUT)
    result = convert_image(REFERENCE_IMAGE, config)
    assert result.metrics is None
    assert result.indexed.shape == (192, 160)
    assert len(result.palette_indices) == 4
