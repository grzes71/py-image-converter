"""GUI side of the preview geometry: the same proportions as the hardware, at any zoom.

The core suite checks the numbers; these tests check that the window really applies them to the view
transform, that zooming does not change the proportions, and that a square in the source is drawn as a
square in ``Atari result``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="PySide6 is not installed (install the 'gui' extra)")

from PySide6.QtWidgets import QApplication  # noqa: E402

from converter.aspect import apply_geometry, preview_pixel_aspect  # noqa: E402
from converter.types import AspectMode, ConversionConfig, ConversionResult  # noqa: E402
from converter_gui.controllers.conversion_controller import DisplayedConversion  # noqa: E402

from gui_fakes import build_window  # noqa: E402

WIDTH, HEIGHT = 160, 192
ZOOM_LEVELS = (1.0, 2.0, 4.0, 8.0)


def _result(indexed: np.ndarray) -> ConversionResult:
    height, width = indexed.shape
    return ConversionResult(
        indexed=indexed,
        palette_indices=[0, 1],
        palette_rgb=np.array([(0, 0, 0), (255, 255, 255)], dtype=np.uint8),
        source_resized=np.zeros((height, width, 3), dtype=np.uint8),
    )


def _square_bitmap(config: ConversionConfig) -> np.ndarray:
    """The bitmap a 200x200 source with a centred 100x100 square produces."""
    source = np.zeros((200, 200, 3), dtype=np.uint8)
    source[50:150, 50:150] = (255, 255, 255)
    prepared = apply_geometry(source, config)
    return (prepared.max(axis=2) > 127).astype(np.uint8)


def _deliver(window, config: ConversionConfig, indexed: np.ndarray) -> None:
    window._controller.result_ready.emit(
        DisplayedConversion(
            generation=1,
            config=config,
            result=_result(indexed),
            timings={},
            produced_at=0.0,
        )
    )
    QApplication.processEvents()


def _pixel_shape(window) -> tuple[float, float]:
    """Size of one bitmap pixel on screen, in device pixels (width, height)."""
    transform = window.preview.views[1].transform()
    return (transform.m11(), transform.m22())


def _square_shape(window, indexed: np.ndarray) -> tuple[float, float]:
    """Size of the bright block of the bitmap on screen, in device pixels."""
    rows, columns = np.nonzero(indexed)
    scale_x, scale_y = _pixel_shape(window)
    width = (columns.max() - columns.min() + 1) * scale_x
    height = (rows.max() - rows.min() + 1) * scale_y
    return (float(width), float(height))


# --- the proportions ----------------------------------------------------


def test_the_result_view_draws_hardware_pixels(qtbot) -> None:
    window, _backend, _boxes = build_window(qtbot)
    config = ConversionConfig(width=WIDTH, height=HEIGHT)

    _deliver(window, config, np.zeros((HEIGHT, WIDTH), dtype=np.uint8))

    scale_x, scale_y = _pixel_shape(window)
    assert scale_x / scale_y == pytest.approx(preview_pixel_aspect(config, size=(WIDTH, HEIGHT)))


def test_the_source_view_keeps_square_pixels(qtbot) -> None:
    window, _backend, _boxes = build_window(qtbot)

    _deliver(
        window, ConversionConfig(width=WIDTH, height=HEIGHT), np.zeros((HEIGHT, WIDTH), np.uint8)
    )

    transform = window.preview.views[0].transform()
    assert transform.m11() == pytest.approx(transform.m22())


@pytest.mark.parametrize("level", ZOOM_LEVELS)
def test_zooming_never_changes_the_proportions(level: float, qtbot) -> None:
    window, _backend, _boxes = build_window(qtbot)
    config = ConversionConfig(width=WIDTH, height=HEIGHT)
    _deliver(window, config, np.zeros((HEIGHT, WIDTH), dtype=np.uint8))

    window.preview.set_zoom(level)

    scale_x, scale_y = _pixel_shape(window)
    assert scale_x == pytest.approx(level * preview_pixel_aspect(config, size=(WIDTH, HEIGHT)))
    assert scale_y == pytest.approx(level)


@pytest.mark.parametrize("mode", list(AspectMode))
def test_a_square_source_is_drawn_as_a_square(mode: AspectMode, qtbot) -> None:
    """Fit squeezes the bitmap, the display stretches it back: the square stays square."""
    window, _backend, _boxes = build_window(qtbot)
    config = ConversionConfig(width=WIDTH, height=HEIGHT, aspect=mode)
    indexed = _square_bitmap(config)

    _deliver(window, config, indexed)

    width, height = _square_shape(window, indexed)
    if mode is AspectMode.DISABLED:
        # Disabled keeps the stretched bitmap, so the square is stretched by the pixel aspect.
        expected = WIDTH * preview_pixel_aspect(config, size=(WIDTH, HEIGHT)) / HEIGHT
        assert width / height == pytest.approx(expected, rel=0.02)
    else:
        assert width == pytest.approx(height, rel=0.03), f"{mode.value}: square on screen"


def test_zooming_a_square_keeps_it_square(qtbot) -> None:
    window, _backend, _boxes = build_window(qtbot)
    config = ConversionConfig(width=WIDTH, height=HEIGHT, aspect=AspectMode.FIT)
    indexed = _square_bitmap(config)
    _deliver(window, config, indexed)

    for level in ZOOM_LEVELS:
        window.preview.set_zoom(level)
        width, height = _square_shape(window, indexed)
        assert width == pytest.approx(height, rel=0.03), f"zoom {level}"


# --- zoom and pan stay synchronised ------------------------------------


def test_the_two_views_keep_one_zoom_and_one_scene_position(qtbot) -> None:
    window, _backend, _boxes = build_window(qtbot)
    _deliver(
        window, ConversionConfig(width=WIDTH, height=HEIGHT), np.zeros((HEIGHT, WIDTH), np.uint8)
    )
    primary, secondary = window.preview.views

    window.preview.set_zoom(4.0)
    primary.verticalScrollBar().setValue(120)
    QApplication.processEvents()

    assert window.preview.zoom == pytest.approx(4.0)
    assert abs(secondary.mapToScene(0, 0).y() - primary.mapToScene(0, 0).y()) <= 1.0 / 4.0
    assert secondary.transform().m22() == pytest.approx(primary.transform().m22())
