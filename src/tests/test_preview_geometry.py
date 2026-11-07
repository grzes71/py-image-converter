"""Geometry of the ``Atari result`` preview.

Two mechanisms have to work together and must not be confused:

* **A - the conversion geometry** (:func:`converter.aspect.plan_geometry`): how the source is fitted
  onto the target grid. With ``fit`` the source is squeezed horizontally by the pixel aspect so that it
  looks right once displayed.
* **B - the display aspect** (:func:`converter.aspect.preview_pixel_aspect`): how wide one hardware
  pixel is drawn.

A square in the source must therefore look square on screen *after* B is applied, and the bitmap must
look exactly like it does in Altirra. The reference numbers come from measuring the emulator (see the
``converter.aspect`` docstring).
"""

from __future__ import annotations

import numpy as np
import pytest

from converter.aspect import (
    PAL_DISPLAY_ASPECT,
    PAL_RASTER,
    apply_geometry,
    pal_screen,
    preview_pixel_aspect,
)
from converter.types import AspectMode, ConversionConfig, ConversionResult

ANTIC_D = (160, 96)
ANTIC_E = (160, 192)
ANTIC_F = (320, 192)

COLUMN = PAL_DISPLAY_ASPECT / (PAL_RASTER[0] / PAL_RASTER[1])
"""One raster column relative to one raster line on screen - the unit the table is built from."""


def _square_source(size: int = 200, block: int = 100) -> np.ndarray:
    """A square-pixel image with one bright square in the middle (100x100 inside 200x200)."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - block) // 2
    image[top : top + block, top : top + block] = (255, 255, 255)
    return image


def _square_in_bitmap(config: ConversionConfig) -> tuple[int, int]:
    """Width x height of the bright square inside the converted bitmap."""
    prepared = apply_geometry(_square_source(), config)
    lit = prepared.max(axis=2) > 127
    rows, columns = np.nonzero(lit)
    assert len(rows), "the square is missing from the bitmap"
    return (
        int(columns.max() - columns.min() + 1),
        int(rows.max() - rows.min() + 1),
    )


def _result(indexed: np.ndarray) -> ConversionResult:
    """A two-colour result around an indexed bitmap, for the renderer and the exporters."""
    height, width = indexed.shape
    return ConversionResult(
        indexed=indexed,
        palette_indices=[0, 1],
        palette_rgb=np.array([(0, 0, 0), (255, 255, 255)], dtype=np.uint8),
        source_resized=np.zeros((height, width, 3), dtype=np.uint8),
    )


# --- the table ----------------------------------------------------------


def test_the_par_values_follow_the_measured_altirra_model() -> None:
    """160-wide modes sample two raster columns per pixel, ANTIC F one; rows double in mode D."""
    assert pal_screen(*ANTIC_D).pixel_aspect == pytest.approx(COLUMN)
    assert pal_screen(*ANTIC_E).pixel_aspect == pytest.approx(2 * COLUMN)
    assert pal_screen(*ANTIC_F).pixel_aspect == pytest.approx(COLUMN)


def test_the_numbers_are_the_documented_ones() -> None:
    assert pal_screen(*ANTIC_D).pixel_aspect == pytest.approx(0.93, abs=0.005)
    assert pal_screen(*ANTIC_E).pixel_aspect == pytest.approx(1.86, abs=0.01)
    assert pal_screen(*ANTIC_F).pixel_aspect == pytest.approx(0.93, abs=0.005)


@pytest.mark.parametrize("size", [ANTIC_D, ANTIC_E, ANTIC_F])
def test_every_mode_covers_the_same_physical_area(size: tuple[int, int]) -> None:
    """D is half as tall, F has twice the columns - all three show the same screen area."""
    assert pal_screen(*size).display_aspect == pytest.approx(1.55, abs=0.01)


def test_the_nineteen_hundred_row_variant_keeps_the_e_factor() -> None:
    assert pal_screen(160, 190).pixel_aspect == pytest.approx(2 * COLUMN)


# --- a square has to stay square ---------------------------------------


def test_a_square_source_stays_square_with_fit() -> None:
    """Mechanism A squeezes it by the PAR, mechanism B stretches it back - once, not twice."""
    config = ConversionConfig(aspect=AspectMode.FIT)
    width, height = _square_in_bitmap(config)

    par = preview_pixel_aspect(config, size=ANTIC_E)
    on_screen = (width * par, height)
    assert on_screen[0] == pytest.approx(on_screen[1], rel=0.03), (
        f"the square is {on_screen[0]:.1f} x {on_screen[1]} on screen"
    )


def test_the_bitmap_is_pre_squeezed_so_the_display_restores_the_proportions() -> None:
    config = ConversionConfig(aspect=AspectMode.FIT)
    width, height = _square_in_bitmap(config)

    assert width < height, "the bitmap content is narrower than tall"
    assert width / height == pytest.approx(1 / preview_pixel_aspect(config, size=ANTIC_E), rel=0.03)


def test_a_double_correction_would_not_be_square() -> None:
    """The same square with the aspect applied twice would be ~86% too wide - the tests catch it."""
    config = ConversionConfig(aspect=AspectMode.FIT)
    width, height = _square_in_bitmap(config)
    par = preview_pixel_aspect(config, size=ANTIC_E)

    assert (width * par) / height == pytest.approx(1.0, rel=0.03), "applied once: square"
    assert (width * par * par) / height > 1.7, "applied twice: clearly not square"


def test_disabled_keeps_the_bitmap_and_shows_the_hardware_shape() -> None:
    """Disabled changes nothing about the conversion, and the preview still uses hardware pixels."""
    config = ConversionConfig(aspect=AspectMode.DISABLED)
    width, height = _square_in_bitmap(config)

    assert (width, height) == (80, 96), "a 200x200 source is stretched onto the whole 160x192 grid"
    par = preview_pixel_aspect(config, size=ANTIC_E)
    assert (width * par) / height == pytest.approx(pal_screen(*ANTIC_E).display_aspect, rel=0.01)


# --- the display factor -------------------------------------------------


@pytest.mark.parametrize(
    ("size", "expected"),
    [(ANTIC_D, 0.93), (ANTIC_E, 1.86), (ANTIC_F, 0.93)],
)
def test_the_display_factor_follows_the_buffers_mode(
    size: tuple[int, int], expected: float
) -> None:
    config = ConversionConfig(width=size[0], height=size[1])

    assert preview_pixel_aspect(config, size=size) == pytest.approx(expected, abs=0.01)


def test_the_correction_mode_does_not_change_the_display_factor() -> None:
    """Mechanism A must not switch mechanism B off - the bug this module guards."""
    for mode in AspectMode:
        config = ConversionConfig(width=160, height=192, aspect=mode)
        assert preview_pixel_aspect(config, size=ANTIC_E) == pytest.approx(2 * COLUMN)
        assert preview_pixel_aspect(config, size=ANTIC_E) != pytest.approx(1.0)


def test_an_unknown_result_size_is_displayed_with_square_pixels() -> None:
    """No ANTIC mode means no hardware display: nothing to correct."""
    config = ConversionConfig(width=160, height=192)

    assert preview_pixel_aspect(config, size=(200, 100)) == pytest.approx(1.0)
    assert preview_pixel_aspect(ConversionConfig(width=200, height=100)) == pytest.approx(1.0)


def test_the_explicit_factor_still_wins_for_a_custom_grid() -> None:
    config = ConversionConfig(width=200, height=100, pixel_aspect=1.5)

    assert preview_pixel_aspect(config, size=(200, 100)) == pytest.approx(1.5)


# --- the export never sees any of this ---------------------------------


def test_the_conversion_is_side_effect_free() -> None:
    """Rendering a preview must not touch the bitmap the exporters write."""
    from converter.hardware_colors import HardwareFrameRenderer

    config = ConversionConfig(aspect=AspectMode.FIT)
    source = _square_source()
    prepared = apply_geometry(source, config)
    result = _result((prepared.max(axis=2) > 127).astype(np.uint8))

    frame = HardwareFrameRenderer().frame(result, ConversionConfig())
    again = apply_geometry(source, config)

    assert frame is not None, "the frame renders"
    assert np.array_equal(prepared, again), "the conversion is unchanged by looking at it"
    assert np.array_equal(source, _square_source()), "the source is untouched"
    assert result.indexed.shape == (192, 160), "the exported bitmap is the target grid"


def test_repeated_exports_are_byte_identical(tmp_path) -> None:
    from converter.pipeline import export_result
    from converter.types import ExportFormat

    config = ConversionConfig(export_formats=[ExportFormat.XEX])
    prepared = apply_geometry(_square_source(), config)
    result = _result((prepared.max(axis=2) > 127).astype(np.uint8))

    export_result(result, tmp_path / "first.xex", config)
    export_result(result, tmp_path / "second.xex", config)

    assert (tmp_path / "first.xex").read_bytes() == (tmp_path / "second.xex").read_bytes()
