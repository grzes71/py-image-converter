"""Export from the GUI (T4.4).

The window owns only the user interaction: the save dialog asks for a folder, a name and a format, and
``converter_gui.exporting`` decides what may be written and calls ``converter.pipeline.export_result``.
The tests therefore replace ``QFileDialog.getSaveFileName`` with a canned answer and inspect the files
that really appear on disk, the status bar and the error boxes.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="PySide6 is not installed (install the 'gui' extra)")

from PySide6.QtWidgets import QFileDialog  # noqa: E402

from converter.exporters.atari_bin_export import (  # noqa: E402
    ANTIC_E_BYTES,
    ANTIC_E_HEIGHT,
    ANTIC_E_WIDTH,
)
from converter.types import ConversionResult, ExportFormat  # noqa: E402
from converter_gui.exporting import (  # noqa: E402
    EXPORT_FORMATS,
    EXPORT_SUFFIX,
    dialog_filters,
)
from converter_gui.main_window import MainWindow  # noqa: E402

from gui_fakes import (  # noqa: E402
    CREATED,
    DEBOUNCE_MS,
    REFERENCE_IMAGE,
    build_window,
    converted_window,
    make_metrics,
    open_window,
)

PALETTE = np.array([[0, 0, 0], [80, 80, 80], [160, 160, 160], [255, 255, 255]], dtype=np.uint8)


def _antic_e_result() -> ConversionResult:
    """A 160x192 four-colour result: every exporter accepts it, and it is cheap to build."""
    indexed = np.zeros((ANTIC_E_HEIGHT, ANTIC_E_WIDTH), dtype=np.uint8)
    indexed[::2, ::2] = 1
    indexed[1::2, 1::2] = 2
    indexed[0::4] = 3
    return ConversionResult(
        indexed=indexed,
        palette_indices=[0, 1, 2, 3],
        palette_rgb=PALETTE.copy(),
        source_resized=np.zeros((ANTIC_E_HEIGHT, ANTIC_E_WIDTH, 3), dtype=np.uint8),
        metrics=make_metrics(),
        timings={"prepare": 0.001, "palette": 0.002, "edges": 0.0, "render": 0.003, "metrics": 0.0},
    )


def _exporting_window(
    qtbot,
) -> tuple[MainWindow, object, list[tuple[str, str]]]:
    """A window showing a result every format can hold."""
    window, backend, boxes = open_window(qtbot)
    backend.finish(_antic_e_result())
    assert window.displayed is not None
    return window, backend, boxes


def _save_dialog(
    monkeypatch,
    *,
    path: Path | str,
    selected_filter: str = "",
    seen: dict[str, object] | None = None,
) -> None:
    """Answer ``QFileDialog.getSaveFileName`` with ``path`` instead of showing it."""

    def fake_dialog(parent, title, directory, filters):
        if seen is not None:
            seen.update(title=title, directory=directory, filters=filters)
        return str(path), selected_filter

    monkeypatch.setattr(QFileDialog, "getSaveFileName", fake_dialog)


def _forbid_dialog(monkeypatch) -> None:
    """Fail loudly if the save dialog is reached at all."""

    def fake_dialog(*args, **kwargs):
        raise AssertionError("the save dialog must not open")

    monkeypatch.setattr(QFileDialog, "getSaveFileName", fake_dialog)


def _filter_for(fmt: ExportFormat) -> str:
    return dialog_filters([fmt])[0][0]


# --- the save dialog ----------------------------------------------------


def test_export_asks_for_a_path_with_the_available_formats(qtbot, monkeypatch) -> None:
    window, _backend, boxes = _exporting_window(qtbot)
    seen: dict[str, object] = {}
    _save_dialog(monkeypatch, path="", seen=seen)  # the user cancels

    assert window.export_dialog() is False

    filters = str(seen["filters"])
    assert filters.count("(*.") == 5
    for suffix in (".png", ".bin", ".s", ".dta", ".xex"):
        assert f"*{suffix}" in filters
    assert "*.asm" in filters, "ASM also accepts .asm, like the CLI"
    suggested = Path(str(seen["directory"]))
    assert suggested.parent == REFERENCE_IMAGE.parent, "the source folder is the default"
    assert suggested.name == f"{REFERENCE_IMAGE.stem}.png", "the first format's suffix is proposed"
    assert boxes == [], "cancelling is not an error"
    assert "cancel" in window.statusBar().currentMessage().lower()


@pytest.mark.parametrize("fmt", EXPORT_FORMATS)
def test_each_format_is_written(qtbot, tmp_path: Path, monkeypatch, fmt: ExportFormat) -> None:
    window, _backend, boxes = _exporting_window(qtbot)
    _save_dialog(monkeypatch, path=tmp_path / "screen", selected_filter=_filter_for(fmt))

    assert window.export_dialog() is True

    written = sorted(tmp_path.iterdir())
    assert [path.name for path in written] == [f"screen{EXPORT_SUFFIX[fmt]}"]
    assert written[0].stat().st_size > 0
    if fmt is ExportFormat.BIN:
        assert written[0].stat().st_size == ANTIC_E_BYTES
    assert boxes == []
    assert "exported" in window.statusBar().currentMessage().lower()


def test_a_typed_suffix_wins_over_the_selected_filter(qtbot, tmp_path: Path, monkeypatch) -> None:
    """The engine writes ASM as ``.s``; typing ``.asm`` must not produce a second file."""
    window, _backend, _ = _exporting_window(qtbot)
    _save_dialog(
        monkeypatch,
        path=tmp_path / "screen.asm",
        selected_filter=_filter_for(ExportFormat.PNG),
    )

    assert window.export_dialog() is True

    assert [path.name for path in sorted(tmp_path.iterdir())] == ["screen.s"]


def test_the_last_directory_is_remembered(qtbot, tmp_path: Path, monkeypatch) -> None:
    window, _backend, _ = _exporting_window(qtbot)
    first = tmp_path / "first"
    first.mkdir()
    _save_dialog(monkeypatch, path=first / "screen", selected_filter=_filter_for(ExportFormat.BIN))
    assert window.export_dialog() is True

    seen: dict[str, object] = {}
    _save_dialog(monkeypatch, path="", seen=seen)
    assert window.export_dialog() is False

    assert Path(str(seen["directory"])).parent == first


# --- compatibility ------------------------------------------------------


def test_incompatible_formats_are_not_offered(qtbot, monkeypatch) -> None:
    """A 10x8 result cannot become BIN/ASM/DTA/XEX; only PNG is offered."""
    window, _backend, _ = converted_window(qtbot)
    seen: dict[str, object] = {}
    _save_dialog(monkeypatch, path="", seen=seen)

    assert window.export_dialog() is False

    filters = str(seen["filters"])
    assert "*.png" in filters
    for suffix in (".bin", ".s", ".dta", ".xex"):
        assert f"*{suffix}" not in filters, f"{suffix} cannot hold this result"


def test_an_incompatible_path_is_refused_with_a_reason(qtbot, tmp_path: Path, monkeypatch) -> None:
    window, _backend, boxes = converted_window(qtbot)
    _save_dialog(
        monkeypatch,
        path=tmp_path / "screen.bin",
        selected_filter=_filter_for(ExportFormat.PNG),
    )

    assert window.export_dialog() is False

    assert list(tmp_path.iterdir()) == [], "nothing is written for an incompatible format"
    assert len(boxes) == 1
    title, message = boxes[0]
    assert title == "Cannot export"
    assert "ANTIC E requires" in message
    assert "failed" in window.statusBar().currentMessage().lower()


def test_a_result_no_format_can_hold_never_reaches_the_dialog(qtbot, monkeypatch) -> None:
    window, backend, boxes = build_window(qtbot)
    window.open_image(REFERENCE_IMAGE)
    backend.finish(
        ConversionResult(
            indexed=np.zeros((8, 10), dtype=np.uint8),
            palette_indices=[0, 1, 2, 3],
            # PNG stores 256 colours at most and the BIN-like formats need 160x190/192: nothing fits.
            palette_rgb=np.zeros((300, 3), dtype=np.uint8),
            source_resized=np.zeros((8, 10, 3), dtype=np.uint8),
            metrics=None,
            timings={},
        )
    )
    assert window.displayed is not None
    _forbid_dialog(monkeypatch)

    assert window.export_dialog() is False

    assert len(boxes) == 1
    assert boxes[0][0] == "Cannot export"
    assert "any supported format" in boxes[0][1]


# --- the action ---------------------------------------------------------


def test_the_export_action_follows_the_state(qtbot) -> None:
    window, backend, _ = build_window(qtbot)
    assert window.export_action.isEnabled() is False, "nothing to export yet"

    window.open_image(REFERENCE_IMAGE)
    assert window.export_action.isEnabled() is False, "a conversion is running"

    backend.finish(_antic_e_result())
    assert window.export_action.isEnabled() is True
    assert "Save the displayed result" in window.export_action.toolTip()

    window.parameter_panel.colors_spin.setValue(2)
    assert window.is_stale is True
    assert window.export_action.isEnabled() is False, "a stale result is not exportable"
    assert "matches the settings" in window.export_action.toolTip()


# --- refusal ------------------------------------------------------------


def test_export_is_refused_without_a_result(qtbot, monkeypatch) -> None:
    window, _backend, boxes = build_window(qtbot)
    _forbid_dialog(monkeypatch)

    assert window.export_dialog() is False

    assert "nothing to export yet" in window.statusBar().currentMessage().lower()
    assert boxes == []


def test_export_is_refused_while_the_conversion_runs(qtbot, monkeypatch) -> None:
    window, backend, _ = open_window(qtbot)  # the conversion is in flight
    _forbid_dialog(monkeypatch)

    assert window.export_dialog() is False

    assert "still running" in window.statusBar().currentMessage()
    backend.finish(_antic_e_result())


def test_export_is_refused_while_the_result_is_stale(qtbot, monkeypatch) -> None:
    window, _backend, _ = _exporting_window(qtbot)
    window.parameter_panel.colors_spin.setValue(2)  # auto-update stays on
    _forbid_dialog(monkeypatch)

    assert window.export_dialog() is False

    assert "no longer matches the settings" in window.statusBar().currentMessage()


# --- failures and the current export settings ---------------------------


def test_a_write_error_is_reported(qtbot, tmp_path: Path, monkeypatch) -> None:
    window, _backend, boxes = _exporting_window(qtbot)
    _save_dialog(
        monkeypatch, path=tmp_path / "screen", selected_filter=_filter_for(ExportFormat.BIN)
    )

    def exploding_export(*args, **kwargs):
        raise OSError("disk is full")

    monkeypatch.setattr("converter_gui.exporting.export_result", exploding_export)

    assert window.export_dialog() is False

    assert len(boxes) == 1
    title, message = boxes[0]
    assert title == "Export failed"
    assert "screen.bin" in message
    assert "disk is full" in window.statusBar().currentMessage()
    assert list(tmp_path.iterdir()) == []


def test_the_current_export_settings_travel_with_the_displayed_image(
    qtbot, tmp_path: Path, monkeypatch
) -> None:
    """The image config produced the picture; the label and the colours come from the live settings."""
    window, _backend, _ = _exporting_window(qtbot)
    window.controller.set_config(
        dataclasses.replace(
            window.requested_config,
            label_name="MENU",
            atari_colors=[0x11, 0x22, 0x33, 0x44, 0x55],
        )
    )
    assert window.displayed is not None
    assert window.displayed.config.label_name is None, "the image configuration is untouched"
    assert window.is_stale is False, "export-only settings do not invalidate the image"
    _save_dialog(
        monkeypatch, path=tmp_path / "screen", selected_filter=_filter_for(ExportFormat.ASM)
    )

    assert window.export_dialog() is True

    text = (tmp_path / "screen.s").read_text(encoding="utf-8")
    assert "MENU" in text, "the label name of the current settings is used"
    assert "$11" in text or "$11," in text, "the hardware colours of the current settings are used"
    assert "COLORS" in text


# --- with the real engine ----------------------------------------------


def test_export_of_a_real_conversion_writes_the_screen(qtbot, tmp_path: Path, monkeypatch) -> None:
    """Window, controller, QThreadBackend and the real engine: the exported BIN is a real screen dump."""
    window = MainWindow(debounce_ms=DEBOUNCE_MS)
    qtbot.addWidget(window)
    CREATED.append(window)
    assert window.open_image(REFERENCE_IMAGE) is True
    qtbot.waitUntil(lambda: window.displayed is not None, timeout=60000)
    assert window.export_action.isEnabled() is True
    _save_dialog(
        monkeypatch, path=tmp_path / "screen", selected_filter=_filter_for(ExportFormat.BIN)
    )

    assert window.export_dialog() is True

    written = tmp_path / "screen.bin"
    assert written.stat().st_size == ANTIC_E_BYTES
    assert window.backend.is_running is True  # the export never touches the worker thread
