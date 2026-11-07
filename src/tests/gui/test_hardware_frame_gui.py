"""GUI tests for the hardware ``Atari result`` preview.

The window has to show the frame the hardware would display - the bitmap interpreted through the
registers the XEX export writes - and it has to keep that promise when the registers change by hand
(no reconversion) and when the automatic mapping changes with a new conversion.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="PySide6 is not installed (install the 'gui' extra)")

from PySide6.QtWidgets import QApplication  # noqa: E402

import converter.hardware_colors as hardware_colors  # noqa: E402
from converter.atari_match import PAL_RGB  # noqa: E402
from converter.hardware_colors import (  # noqa: E402
    render_hardware_frame,
    resolved_registers,
)
from converter.types import AspectMode, ConversionConfig, ConversionResult, ExportFormat  # noqa: E402
from converter_gui.controllers.conversion_controller import DisplayedConversion  # noqa: E402
from converter_gui.exporting import ExportPayload, export_displayed  # noqa: E402

from gui_fakes import build_window, make_result, open_window  # noqa: E402

WIDTH, HEIGHT = 160, 192

#: Four easy-to-tell-apart register values (COLOR0..COLOR4) for the manual tests.
MANUAL = (0x0F, 0xF0, 0x2A, 0x84, 0x00)


def _screen_result(palette=None) -> ConversionResult:
    """A 160x192 result whose bitmap uses all four pixel values."""
    colours = (
        palette
        if palette is not None
        else [(0, 0, 0), (255, 255, 255), (44, 142, 80), (29, 129, 68)]
    )
    indexed = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    indexed[:, :] = np.arange(WIDTH, dtype=np.uint8) % 4
    return ConversionResult(
        indexed=indexed,
        palette_indices=list(range(len(colours))),
        palette_rgb=np.array(colours, dtype=np.uint8).reshape(len(colours), 3),
        source_resized=np.tile(indexed[:, :, np.newaxis], (1, 1, 3)).astype(np.uint8),
    )


def _shown_colours(window) -> np.ndarray:
    """The RGB image ``Atari result`` currently shows."""
    image = window.preview.views[1].pixmap_item.pixmap().toImage()
    return np.array(
        [
            [image.pixelColor(x, y).getRgb()[:3] for x in range(image.width())]
            for y in range(image.height())
        ],
        dtype=np.uint8,
    )


def _converted_window(qtbot, result=None):
    """A window with one finished conversion of a full-size result."""
    window, backend, boxes = open_window(qtbot)
    backend.finish(result if result is not None else _screen_result())
    QApplication.processEvents()
    return window, backend, boxes


def _toggle_manual(panel, registers=MANUAL) -> None:
    panel.atari_palette_check.setChecked(False)
    for spin, value in zip(panel.atari_color_spins, registers, strict=True):
        spin.setValue(value)
    QApplication.processEvents()


# --- what the result view shows ----------------------------------------


def test_the_result_view_shows_the_hardware_frame(qtbot) -> None:
    window, _backend, _boxes = _converted_window(qtbot)
    result = window.displayed.result
    frame = window._frame

    assert frame is not None
    assert np.array_equal(_shown_colours(window), frame.rgb)
    assert frame.registers == tuple(resolved_registers(result, window.requested_config)), (
        "the same registers the export would write"
    )


def test_the_preview_shows_pal_palette_colours(qtbot) -> None:
    window, _backend, _boxes = _converted_window(qtbot)

    shown = {tuple(int(channel) for channel in pixel) for pixel in window._frame.rgb.reshape(-1, 3)}
    assert shown <= set(PAL_RGB)


def test_the_preview_does_not_scale_the_bitmap(qtbot) -> None:
    """The frame is the bitmap size; zooming is the view's job, not the renderer's."""
    window, _backend, _boxes = _converted_window(qtbot)
    pixmap = window.preview.views[1].pixmap_item.pixmap()

    assert (pixmap.width(), pixmap.height()) == (WIDTH, HEIGHT)


def test_a_size_without_a_mode_falls_back_to_the_converted_image(qtbot) -> None:
    """A result no ANTIC mode can display has no hardware frame (and cannot be exported as XEX)."""
    window, backend, _boxes = open_window(qtbot)
    backend.finish(make_result())
    QApplication.processEvents()

    assert window._frame is None
    assert np.array_equal(_shown_colours(window), make_result().converted_rgb)


# --- manual registers ---------------------------------------------------


def test_a_manual_register_redraws_without_a_conversion(qtbot) -> None:
    window, backend, _boxes = _converted_window(qtbot)
    before = _shown_colours(window)
    conversions = len(backend.conversions())

    _toggle_manual(window.parameter_panel)

    assert len(backend.conversions()) == conversions, "export settings never start a conversion"
    after = _shown_colours(window)
    assert not np.array_equal(before, after), "the picture followed the new registers"
    assert window._frame.registers == MANUAL


def test_the_manual_registers_are_interpreted_per_pixel(qtbot) -> None:
    window, _backend, _boxes = _converted_window(qtbot)

    _toggle_manual(window.parameter_panel)

    expected = render_hardware_frame(window.displayed.result.indexed, MANUAL, "E")
    assert np.array_equal(_shown_colours(window), expected)
    for value, register in enumerate((4, 0, 1, 2)):
        column = window.displayed.result.indexed[0]
        column_value = int(column[value])
        assert column_value == value, "the test bitmap shows one value per column"
        assert tuple(int(channel) for channel in expected[0, value]) == PAL_RGB[MANUAL[register]]


def test_toggling_back_restores_the_automatic_colours(qtbot) -> None:
    window, _backend, _boxes = _converted_window(qtbot)
    automatic = _shown_colours(window)

    _toggle_manual(window.parameter_panel)
    assert not np.array_equal(_shown_colours(window), automatic)

    window.parameter_panel.atari_palette_check.setChecked(True)
    QApplication.processEvents()

    assert np.array_equal(_shown_colours(window), automatic)
    assert window._frame.registers == tuple(
        resolved_registers(window.displayed.result, window.requested_config)
    )


def test_a_new_conversion_updates_the_automatic_colours(qtbot) -> None:
    window, backend, _boxes = _converted_window(qtbot)
    before = _shown_colours(window)
    other = _screen_result([(255, 255, 255), (0, 0, 0), (0x62, 0x0E, 0x71), (0x1F, 0x2F, 0x3F)])

    # A new conversion needs a new generation: change an image parameter, let the debounce fire.
    panel = window.parameter_panel
    panel.dither_combo.setCurrentIndex(panel.dither_combo.count() - 1)
    qtbot.waitUntil(lambda: backend.running is not None, timeout=2000)
    backend.finish(other)
    QApplication.processEvents()

    assert not np.array_equal(_shown_colours(window), before)
    assert np.array_equal(_shown_colours(window), window._frame.rgb)
    assert window._frame.registers == tuple(resolved_registers(other, window.requested_config))


def test_an_unrelated_parameter_change_does_not_render_again(qtbot, monkeypatch) -> None:
    window, _backend, _boxes = _converted_window(qtbot)
    calls = {"renders": 0}
    original = hardware_colors.render_hardware_frame

    def counting(*args, **kwargs):
        calls["renders"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(hardware_colors, "render_hardware_frame", counting)
    panel = window.parameter_panel

    panel.sharpen_check.setChecked(not panel.sharpen_check.isChecked())
    QApplication.processEvents()

    assert calls["renders"] == 0, "a change that keeps the colours must hit the cache"


def test_one_register_edit_renders_once(qtbot, monkeypatch) -> None:
    window, _backend, _boxes = _converted_window(qtbot)
    panel = window.parameter_panel
    panel.atari_palette_check.setChecked(False)
    QApplication.processEvents()

    calls = {"renders": 0}
    original = hardware_colors.render_hardware_frame

    def counting(*args, **kwargs):
        calls["renders"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(hardware_colors, "render_hardware_frame", counting)
    panel.atari_color_spins[0].setValue(0x2A)
    QApplication.processEvents()

    assert calls["renders"] == 1, "a new register set is rendered exactly once"
    assert window._frame.registers[0] == 0x2A


# --- geometry -----------------------------------------------------------


def _aspect_ratio(window) -> float:
    """How much wider than tall the result view draws the bitmap (the pixel aspect ratio)."""
    transform = window.preview.views[1].transform()
    return transform.m11() / transform.m22()


def _deliver(window, config: ConversionConfig, result=None) -> None:
    """Hand the window a finished conversion with exactly this configuration."""
    window._controller.result_ready.emit(
        DisplayedConversion(
            generation=1,
            config=config,
            result=result if result is not None else _screen_result(),
            timings={},
            produced_at=0.0,
        )
    )
    QApplication.processEvents()


def test_the_hardware_frame_is_shown_with_the_pal_aspect(qtbot) -> None:
    """The frame brings the colours, the view brings the pixel shape - once, not twice."""
    window, _backend, _boxes = build_window(qtbot)
    _deliver(window, ConversionConfig(width=WIDTH, height=HEIGHT, aspect=AspectMode.FIT))

    assert window._frame.rgb.shape == (HEIGHT, WIDTH, 3), "the renderer never scales"
    assert _aspect_ratio(window) == pytest.approx(1.86, abs=0.01)  # ANTIC E, PAL
    source, _result = window.preview.views
    assert source.transform().m11() / source.transform().m22() == pytest.approx(1.0)


@pytest.mark.parametrize("level", [1.0, 2.0, 4.0, 8.0])
def test_the_aspect_survives_every_zoom_level(level: float, qtbot) -> None:
    window, _backend, _boxes = build_window(qtbot)
    _deliver(window, ConversionConfig(width=WIDTH, height=HEIGHT, aspect=AspectMode.FIT))

    window.preview.set_zoom(level)

    assert _aspect_ratio(window) == pytest.approx(1.86, abs=0.01)


def test_the_geometry_does_not_depend_on_the_colours(qtbot) -> None:
    """Manual registers change the pixels, never the shape the view draws them with."""
    window, _backend, _boxes = build_window(qtbot)
    _deliver(window, ConversionConfig(width=WIDTH, height=HEIGHT, aspect=AspectMode.FIT))
    before = _aspect_ratio(window)

    _toggle_manual(window.parameter_panel)

    assert _aspect_ratio(window) == pytest.approx(before)


# --- the exported file ------------------------------------------------


def test_the_exported_xex_uses_the_registers_of_the_preview(qtbot, tmp_path) -> None:
    """Acceptance criterion: the file and the picture come from one set of register values."""
    window, _backend, _boxes = _converted_window(qtbot)
    payload = ExportPayload(displayed=window.displayed, export_config=window.requested_config)

    export_displayed(payload, ExportFormat.XEX, tmp_path / "gui.xex")

    sequence = bytearray()
    for offset, value in enumerate(window._frame.registers):
        address = 0x02C4 + offset
        sequence.extend([0xA9, value & 0xFF, 0x8D, address & 0xFF, (address >> 8) & 0xFF])
    assert bytes(sequence) in (tmp_path / "gui.xex").read_bytes()


def test_antic_d_uses_its_own_aspect(qtbot) -> None:
    window, backend, _boxes = _converted_window(qtbot)
    panel = window.parameter_panel
    panel.aspect_combo.setCurrentIndex(panel.aspect_combo.findData("fit"))
    panel.target_combo.setCurrentIndex(panel.target_combo.findData("antic_d_96"))
    qtbot.waitUntil(lambda: backend.running is not None, timeout=2000)

    d_result = ConversionResult(
        indexed=np.zeros((96, 160), dtype=np.uint8),
        palette_indices=[0, 1],
        palette_rgb=np.array([(0, 0, 0), (255, 255, 255)], dtype=np.uint8),
        source_resized=np.zeros((96, 160, 3), dtype=np.uint8),
    )
    backend.finish(d_result)
    QApplication.processEvents()

    assert window._frame.mode == "D"
    assert window._frame.rgb.shape == (96, 160, 3)
    assert _aspect_ratio(window) == pytest.approx(0.93, abs=0.01), (
        "ANTIC D doubles every line, so its pixels are half as wide as ANTIC E's"
    )
