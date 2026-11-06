"""Regression tests for the B1-B4 pipeline fixes.

Each test pins one defect from the implementation plan to the behaviour that was wrong:

- B1: ``dither=none`` quantized against ``range(N)`` (Atari colours 0..3) instead of the palette.
- B2: metrics were computed against ``range(N)`` instead of the palette.
- B3: preview images were rendered with ``range(N)`` instead of the palette.
- B4: the adaptive quantizer never received a variance map, so it behaved like ``nearest``.

The assertions compare the pipeline result against an independent composition of the same
public stages, so they keep working after the planned stage refactoring (T1.1).
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from converter.histogram import compute_local_variance
from converter.metrics import compute_metrics
from converter.pipeline import convert_image, export_result
from converter.preview import generate_previews
from converter.quantizer import quantize, quantize_adaptive
from converter.types import ConversionConfig, DitherMethod, PaletteMethod, QuantizerMethod

REFERENCE_IMAGE = Path(__file__).parent / "fixtures" / "reference.png"


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dither_none_uses_the_selected_palette() -> None:
    """B1: the quantizer must receive the selected palette indices, not ``range(N)``."""
    config = ConversionConfig(dither=DitherMethod.NONE, palette=PaletteMethod.MEDIAN_CUT)
    result = convert_image(REFERENCE_IMAGE, config)

    palette = result.palette_indices
    grey_ramp = list(range(len(palette)))
    assert palette != grey_ramp, "fixture precondition: the palette must differ from range(N)"

    with_palette = quantize(result.source_resized, palette, config.quantizer)
    with_grey_ramp = quantize(result.source_resized, grey_ramp, config.quantizer)

    np.testing.assert_array_equal(result.indexed, with_palette)
    assert not np.array_equal(with_palette, with_grey_ramp), "the two quantizations must differ"
    assert not np.array_equal(result.indexed, with_grey_ramp)


def test_metrics_use_the_selected_palette() -> None:
    """B2: metrics must be computed against the palette that produced the image."""
    config = ConversionConfig(metrics=True)
    result = convert_image(REFERENCE_IMAGE, config)
    assert result.metrics is not None

    palette = result.palette_indices
    grey_ramp = list(range(len(palette)))
    expected = compute_metrics(result.source_resized, result.indexed, palette)
    wrong = compute_metrics(result.source_resized, result.indexed, grey_ramp)

    assert result.metrics.mse == pytest.approx(expected.mse)
    assert result.metrics.psnr == pytest.approx(expected.psnr)
    assert result.metrics.ssim == pytest.approx(expected.ssim)
    assert result.metrics.histogram_coverage == pytest.approx(expected.histogram_coverage)
    assert result.metrics.mse != pytest.approx(wrong.mse)
    assert result.metrics.mse < wrong.mse, "the real palette must reconstruct better than grey"


def test_previews_use_the_selected_palette(tmp_path: Path) -> None:
    """B3: preview images must be rendered with the palette that produced the image."""
    config = ConversionConfig(
        dither=DitherMethod.NONE,
        palette=PaletteMethod.MEDIAN_CUT,
        preview=True,
    )
    result = convert_image(REFERENCE_IMAGE, config)

    export_result(result, tmp_path / "screen.bin", config)
    actual = sorted((tmp_path / "screen_preview").glob("*.png"))
    assert len(actual) == 4

    expected = generate_previews(
        result.source_resized,
        result.indexed,
        result.palette_indices,
        tmp_path / "expected",
        "screen",
    )
    grey_ramp = generate_previews(
        result.source_resized,
        result.indexed,
        list(range(len(result.palette_indices))),
        tmp_path / "grey_ramp",
        "screen",
    )

    for produced, wanted, wrong in zip(actual, sorted(expected), sorted(grey_ramp), strict=True):
        assert _sha256(produced) == _sha256(wanted)
        assert _sha256(wanted) != _sha256(wrong)


def test_preview_png_size_matches_the_upscale_factor(tmp_path: Path) -> None:
    """B3: the preview files still have the documented x2/x4 dimensions after the fix."""
    config = ConversionConfig(
        dither=DitherMethod.NONE,
        palette=PaletteMethod.MEDIAN_CUT,
        preview=True,
    )
    result = convert_image(REFERENCE_IMAGE, config)
    export_result(result, tmp_path / "screen.bin", config)

    preview_dir = tmp_path / "screen_preview"
    with Image.open(preview_dir / "screen_x2.png") as image:
        assert image.size == (result.indexed.shape[1] * 2, result.indexed.shape[0] * 2)
    with Image.open(preview_dir / "screen_x4.png") as image:
        assert image.size == (result.indexed.shape[1] * 4, result.indexed.shape[0] * 4)


def test_adaptive_quantizer_receives_the_variance_map() -> None:
    """B4: the adaptive quantizer must use a local variance map instead of acting like nearest."""
    adaptive_config = ConversionConfig(
        dither=DitherMethod.NONE,
        quantizer=QuantizerMethod.ADAPTIVE,
        palette=PaletteMethod.MEDIAN_CUT,
    )
    nearest_config = replace(adaptive_config, quantizer=QuantizerMethod.NEAREST)

    adaptive = convert_image(REFERENCE_IMAGE, adaptive_config)
    nearest = convert_image(REFERENCE_IMAGE, nearest_config)

    assert not np.array_equal(adaptive.indexed, nearest.indexed), (
        "adaptive must differ from nearest"
    )

    variance_map = compute_local_variance(adaptive.source_resized)
    expected = quantize_adaptive(adaptive.source_resized, adaptive.palette_indices, variance_map)
    np.testing.assert_array_equal(adaptive.indexed, expected)


def test_adaptive_quantizer_reports_different_metrics() -> None:
    """B4: because the output differs, the reported metrics must differ as well."""
    adaptive_config = ConversionConfig(
        dither=DitherMethod.NONE,
        quantizer=QuantizerMethod.ADAPTIVE,
        palette=PaletteMethod.MEDIAN_CUT,
        metrics=True,
    )
    nearest_config = replace(adaptive_config, quantizer=QuantizerMethod.NEAREST)

    adaptive = convert_image(REFERENCE_IMAGE, adaptive_config)
    nearest = convert_image(REFERENCE_IMAGE, nearest_config)

    assert adaptive.metrics is not None
    assert nearest.metrics is not None
    assert adaptive.metrics.mse != pytest.approx(nearest.metrics.mse)


def test_dithered_output_ignores_the_quantizer() -> None:
    """Guard: the B1/B4 fix must not leak the quantizer into dithered conversions."""
    dithered = ConversionConfig(dither=DitherMethod.FLOYD, palette=PaletteMethod.MEDIAN_CUT)
    with_adaptive = replace(dithered, quantizer=QuantizerMethod.ADAPTIVE)

    first = convert_image(REFERENCE_IMAGE, dithered)
    second = convert_image(REFERENCE_IMAGE, with_adaptive)
    np.testing.assert_array_equal(first.indexed, second.indexed)
