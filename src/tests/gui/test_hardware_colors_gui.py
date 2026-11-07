"""GUI tests for the hardware colour registers.

What the section promises: with *Use converter palette* on, the fields show the values the export
will really write (mapped from the converted palette), with it off, the values typed by hand are used
and kept. The last test exports through the GUI's own path and decodes the resulting XEX, which is the
acceptance criterion of the task.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6", reason="PySide6 is not installed (install the 'gui' extra)")

from PySide6.QtWidgets import QApplication  # noqa: E402

import numpy as np  # noqa: E402

from converter.hardware_colors import derived_registers, resolved_registers  # noqa: E402
from converter.types import (  # noqa: E402
    ConversionConfig,
    ConversionResult,
    ExportFormat,
)
from converter_gui.controllers.conversion_controller import DisplayedConversion  # noqa: E402
from converter_gui.exporting import ExportPayload, export_displayed  # noqa: E402
from converter_gui.widgets.parameter_panel import ParameterPanel  # noqa: E402

from gui_fakes import build_window  # noqa: E402

#: A palette whose measured Atari colours are known: dark grey, white, green, purple.
KNOWN_PALETTE = [(0x32, 0x31, 0x32), (0xFF, 0xFF, 0xFF), (0x2C, 0x8E, 0x50), (0x62, 0x0E, 0x71)]
KNOWN_REGISTERS = [0x0F, 0xA4, 0x40, 0x00, 0x00]


def _panel(qtbot, colors=KNOWN_PALETTE, size=(160, 192)) -> ParameterPanel:
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    if colors is not None:
        panel.set_palette_colors(colors, size=size)
    QApplication.processEvents()
    return panel


def _values(panel: ParameterPanel) -> list[int]:
    return [spin.value() for spin in panel.atari_color_spins]


# --- automatic values ---------------------------------------------------


def test_the_derived_values_are_visible_in_the_section(qtbot) -> None:
    panel = _panel(qtbot)

    assert panel.atari_palette_check.isChecked() is True
    assert _values(panel) == KNOWN_REGISTERS, "what the export will write is on screen"
    assert _values(panel) == derived_registers(KNOWN_PALETTE, "E")


def test_the_values_follow_the_palette(qtbot) -> None:
    panel = _panel(qtbot)
    assert _values(panel) == KNOWN_REGISTERS

    panel.set_palette_colors([(0x32, 0x31, 0x32), (0x6C, 0x24, 0x00)], size=(160, 192))
    QApplication.processEvents()

    assert _values(panel) == derived_registers([(0x32, 0x31, 0x32), (0x6C, 0x24, 0x00)], "E")
    assert _values(panel)[0] != KNOWN_REGISTERS[0] or _values(panel)[4] == 0x00


def test_the_mode_of_the_result_decides_the_arrangement(qtbot) -> None:
    palette = [(0xFF, 0xFF, 0xFF), (0xB8, 0x5E, 0x60)]
    panel = _panel(qtbot, colors=palette)
    ant_e = _values(panel)

    panel.set_palette_colors(palette, size=(320, 192))
    QApplication.processEvents()
    ant_f = _values(panel)

    assert ant_e[4] == 0x0F, "mode E keeps the background in COLBK"
    assert ant_f[1] != 0x00 and ant_f[2] != 0x00, "mode F uses COLPF1/COLPF2"
    assert ant_f[2] & 0xF0 == ant_f[1] & 0xF0, "mode F shares one hue"


def test_without_a_palette_nothing_is_invented(qtbot) -> None:
    panel = _panel(qtbot, colors=None)

    assert _values(panel) == [0, 0, 0, 0, 0]
    assert "after the first conversion" in panel.atari_status.text()


def test_the_status_says_the_values_are_automatic(qtbot) -> None:
    panel = _panel(qtbot)

    text = panel.atari_status.text()
    assert text.startswith("Automatic:")
    assert "CIE94" in text
    assert "COLBK" in text, "the section also explains where colour 0 lands"


def test_the_automatic_mode_reports_no_manual_colours(qtbot) -> None:
    panel = _panel(qtbot)

    assert panel.config().atari_colors is None, (
        "the export derives them, the panel does not freeze them"
    )


# --- manual values ------------------------------------------------------


def test_manual_values_survive_the_checkbox_round_trip(qtbot) -> None:
    panel = _panel(qtbot)

    panel.atari_palette_check.setChecked(False)
    for spin, value in zip(panel.atari_color_spins, (0x11, 0x22, 0x33, 0x44, 0x55), strict=True):
        spin.setValue(value)
    QApplication.processEvents()

    assert panel.config().atari_colors == [0x11, 0x22, 0x33, 0x44, 0x55]

    panel.atari_palette_check.setChecked(True)
    QApplication.processEvents()
    assert _values(panel) == KNOWN_REGISTERS, "automatic values replace them on screen"

    panel.atari_palette_check.setChecked(False)
    QApplication.processEvents()

    assert _values(panel) == [0x11, 0x22, 0x33, 0x44, 0x55], "the manual values came back"
    assert panel.config().atari_colors == [0x11, 0x22, 0x33, 0x44, 0x55]


def test_manual_values_are_not_overwritten_by_a_new_palette(qtbot) -> None:
    panel = _panel(qtbot)
    panel.atari_palette_check.setChecked(False)
    for spin, value in zip(panel.atari_color_spins, (1, 2, 3, 4, 5), strict=True):
        spin.setValue(value)

    panel.set_palette_colors([(0x32, 0x31, 0x32), (0xFF, 0xFF, 0xFF)], size=(160, 192))
    QApplication.processEvents()

    assert _values(panel) == [1, 2, 3, 4, 5]
    assert panel.config().atari_colors == [1, 2, 3, 4, 5]


def test_the_status_says_the_values_are_manual(qtbot) -> None:
    panel = _panel(qtbot)

    panel.atari_palette_check.setChecked(False)
    QApplication.processEvents()

    assert panel.atari_status.text().startswith("Manual:")


def test_the_checkbox_does_not_change_the_image_configuration(qtbot) -> None:
    """Toggling is export-only: it must not touch a single image-relevant field."""
    panel = _panel(qtbot)
    before = panel.config()

    panel.atari_palette_check.setChecked(False)
    for spin in panel.atari_color_spins:
        spin.setValue(0x2A)
    QApplication.processEvents()
    after = panel.config()

    image_fields = (
        "width",
        "height",
        "palette",
        "colors",
        "dither",
        "quantizer",
        "resize",
        "aspect",
    )
    for name in image_fields:
        assert getattr(after, name) == getattr(before, name)


# --- the export path ----------------------------------------------------


def _expected_writes(registers) -> bytes:
    """The init routine's ``LDA #value / STA $02Cx`` sequence for five register values.

    Decoding the whole XEX is not needed here: the five writes are contiguous, so their exact bytes
    can be looked up in the file - and they can only be there if the exporter wrote precisely these
    values to precisely these shadows.
    """
    sequence = bytearray()
    for offset, value in enumerate(registers):
        address = 0x02C4 + offset
        sequence.extend([0xA9, value & 0xFF, 0x8D, address & 0xFF, (address >> 8) & 0xFF])
    return bytes(sequence)


def _payload(panel: ParameterPanel, result=None) -> ExportPayload:
    """An export payload for a result with a real screen size (10x8 could not be exported at all)."""
    result = _conversion_result() if result is None else result
    height, width = result.indexed.shape
    displayed = DisplayedConversion(
        generation=1,
        config=ConversionConfig(width=width, height=height),
        result=result,
        timings={},
        produced_at=0.0,
    )
    return ExportPayload(displayed=displayed, export_config=panel.config())


def _conversion_result(
    palette=KNOWN_PALETTE, width: int = 160, height: int = 192
) -> ConversionResult:
    """A minimal result with a given palette and an exportable shape."""
    return ConversionResult(
        indexed=np.zeros((height, width), dtype=np.uint8),
        palette_indices=list(range(len(palette))),
        palette_rgb=np.array(palette, dtype=np.uint8).reshape(len(palette), 3),
        source_resized=np.zeros((height, width, 3), dtype=np.uint8),
    )


def test_exporting_from_the_gui_writes_the_mapped_registers(qtbot, tmp_path) -> None:
    """The acceptance criterion: *Use converter palette* -> the XEX holds the mapped colours."""
    panel = _panel(qtbot)
    payload = _payload(panel)

    export_displayed(payload, ExportFormat.XEX, tmp_path / "gui.xex")

    expected = resolved_registers(payload.displayed.result, payload.export_config, "E")
    assert _expected_writes(expected) in (tmp_path / "gui.xex").read_bytes()
    assert expected == resolved_registers(payload.displayed.result, ConversionConfig(), "E"), (
        "the payload carries no manual colours, so the mapping decides"
    )


def test_exporting_from_the_gui_uses_manual_values_when_they_are_set(qtbot, tmp_path) -> None:
    panel = _panel(qtbot)
    panel.atari_palette_check.setChecked(False)
    for spin, value in zip(panel.atari_color_spins, (0x21, 0x22, 0x23, 0x24, 0x25), strict=True):
        spin.setValue(value)
    payload = _payload(panel)

    export_displayed(payload, ExportFormat.XEX, tmp_path / "manual.xex")

    assert (
        _expected_writes([0x21, 0x22, 0x23, 0x24, 0x25]) in (tmp_path / "manual.xex").read_bytes()
    )


def test_the_window_shows_the_values_of_the_displayed_result(qtbot) -> None:
    window, _backend, _boxes = build_window(qtbot)
    result = _conversion_result()

    window._controller.result_ready.emit(
        DisplayedConversion(
            generation=1,
            config=ConversionConfig(width=160, height=192),
            result=result,
            timings={},
            produced_at=0.0,
        )
    )
    QApplication.processEvents()

    expected = derived_registers(KNOWN_PALETTE, "E")
    assert [spin.value() for spin in window.parameter_panel.atari_color_spins] == expected
    assert expected == KNOWN_REGISTERS
