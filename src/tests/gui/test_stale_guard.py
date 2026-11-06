"""Display rules of plan §7.5 and §7.6 (T3.4).

Two pieces of state must never be confused: ``DisplayedConversion`` (the image on screen together with
the configuration that produced it) and the controller's *requested* configuration, which may already
carry newer export-only settings. These tests drive the real :class:`ConversionController` through a
fake backend and check the rules at the window: a stale result is never presented or exported as
current, an export intent waits for the fresh result, a failure keeps the last good preview together
with its metrics and palette, and only the configuration that produced the image travels with the
image.

The export dialog itself belongs to T4.4 - there is no action and no window yet - so the gate is
exercised through ``MainWindow.request_export()`` and the ``export_requested`` signal it publishes,
exactly the interface T4.4 will use. Skipped when PySide6 is missing (see ``conftest.py``).
"""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("PySide6", reason="PySide6 is not installed (install the 'gui' extra)")

from converter.types import ExportFormat  # noqa: E402
from converter_gui.controllers.conversion_controller import (  # noqa: E402
    DisplayedConversion,
)
from converter_gui.main_window import ExportPayload  # noqa: E402

from gui_fakes import (  # noqa: E402
    REFERENCE_IMAGE,
    build_window,
    converted_window,
    make_result,
)


# --- the export gate (§7.5 rules 2 and 3) -------------------------------


def test_export_is_blocked_without_a_current_result(qtbot) -> None:
    """No result, a running conversion and a stale result all refuse the export."""
    window, backend, _ = build_window(qtbot)
    emitted: list[ExportPayload] = []
    window.export_requested.connect(emitted.append)

    assert window.request_export() is False, "nothing is loaded"

    window.open_image(REFERENCE_IMAGE)
    assert window.request_export() is False, "a conversion is running"
    backend.finish()
    assert len(emitted) == 1, "the controller honours the waiting intent once the result lands"
    assert emitted[0].displayed is window.displayed
    assert window.request_export() is True
    assert len(emitted) == 2

    window.parameter_panel.colors_spin.setValue(2)
    assert window.request_export() is False, "the result no longer matches the settings"
    assert len(emitted) == 2, "a stale request emits nothing"


def test_the_export_payload_keeps_the_image_config_and_the_export_settings_apart(qtbot) -> None:
    """Export-only settings never convert, so they can be newer than the displayed result."""
    window, backend, _ = converted_window(qtbot)
    emitted: list[ExportPayload] = []
    window.export_requested.connect(emitted.append)
    generation_before = window.latest_generation

    window.controller.set_config(
        dataclasses.replace(
            window.requested_config,
            export_formats=[ExportFormat.PNG, ExportFormat.ASM],
            atari_colors=[0x11, 0x22],
            label_name="menu",
        )
    )

    assert window.latest_generation == generation_before, "export-only changes do not convert"
    assert window.is_stale is False, "the image still matches the image settings"
    assert window.state_badge.text().startswith("ready")
    assert window.request_export() is True

    payload = emitted[0]
    assert payload.displayed is window.displayed
    assert payload.displayed.config.export_formats == [ExportFormat.BIN], "the image config"
    assert payload.displayed.config == backend.configs()[-1], (
        "exactly what the engine was told to do"
    )
    assert payload.export_config.export_formats == [ExportFormat.PNG, ExportFormat.ASM]
    assert payload.export_config.atari_colors == [0x11, 0x22]
    assert payload.export_config.label_name == "menu"
    assert backend.conversions() == [1], "no conversion was needed"


def test_the_export_signal_is_the_isolated_hook_for_the_export_window(qtbot) -> None:
    """T4.4 will connect to ``MainWindow.export_requested``; until then nothing else happens."""
    window, backend, _ = converted_window(qtbot)
    emitted: list[ExportPayload] = []
    window.export_requested.connect(emitted.append)

    window.convert_action.trigger()  # a no-op: gen 1 is already displayed
    assert emitted == []

    window.request_export()
    assert len(emitted) == 1
    assert isinstance(emitted[0].displayed, DisplayedConversion)


# --- generations decide what is current (§7.5 rules 1 and 4) ------------


def test_a_superseded_result_is_never_shown(qtbot) -> None:
    """A late answer for an old generation must not replace the current one."""
    window, backend, _ = converted_window(qtbot)
    window.parameter_panel.colors_spin.setValue(2)  # gen 2
    qtbot.waitUntil(lambda: window.is_updating is True, timeout=2000)
    backend.finish(make_result(width=20, height=16))  # the current generation
    current = window.displayed
    assert current is not None and current.generation == 2

    backend.finish_late(1, make_result())
    backend.fail_late(1, "late failure")

    assert window.displayed is current, "the old answer is ignored"
    assert window.preview.views[0].pixmap_item.pixmap().width() == 20
    assert window.error_banner.isHidden() is True, "a late failure shows no error"
    assert window.last_error is None
    assert window.request_export() is True, "the current result stays exportable"


def test_staleness_comes_from_generations_not_config_equality(qtbot) -> None:
    window, _, _ = converted_window(qtbot)
    original_colors = window.requested_config.colors

    window.parameter_panel.colors_spin.setValue(2)
    window.parameter_panel.colors_spin.setValue(original_colors)

    assert window.displayed is not None
    assert window.requested_config == window.displayed.config, "the configurations match again"
    assert window.is_stale is True, "but the generation moved on, so the display is stale"


def test_the_shown_times_belong_to_the_displayed_generation(qtbot) -> None:
    """Rule 5: the stage times describe the image on screen, never the pending conversion."""
    window, backend, _ = converted_window(qtbot)
    displayed = window.displayed
    assert displayed is not None
    expected_ms = sum(displayed.timings.values()) * 1000
    timings_before = window.metrics_panel.timing_values

    window.parameter_panel.colors_spin.setValue(2)  # gen 2 starts
    qtbot.waitUntil(lambda: window.is_updating is True, timeout=2000)

    message = window.statusBar().currentMessage()
    assert f"updating… (gen {window.latest_generation})" in message
    assert f"stages {expected_ms:.1f} ms" in message, (
        "the times still belong to the displayed gen 1"
    )
    assert window.metrics_panel.timing_values == timings_before


# --- failures keep the last good preview (§7.6) -------------------------


def test_a_failed_conversion_keeps_the_previous_result(qtbot) -> None:
    window, backend, boxes = converted_window(qtbot)
    displayed = window.displayed
    metrics_before = window.metrics_panel.metric_values
    timings_before = window.metrics_panel.timing_values
    palette_before = window.parameter_panel.background_color
    views_before = [view.pixmap_item.pixmap().cacheKey() for view in window.preview.views]
    window.parameter_panel.colors_spin.setValue(2)  # gen 2
    qtbot.waitUntil(lambda: window.is_updating is True, timeout=2000)

    backend.fail("boom")

    assert boxes == [], "a conversion error is not modal (plan §7.6)"
    assert window.error_banner.isHidden() is False
    assert "The conversion failed" in window.error_label.text()
    assert "boom" in window.error_label.text()
    assert window.displayed is displayed, "the last good result must survive a failure"
    assert displayed is not None and displayed.generation == 1
    assert [view.pixmap_item.pixmap().cacheKey() for view in window.preview.views] == views_before
    assert window.metrics_panel.metric_values == metrics_before
    assert window.metrics_panel.timing_values == timings_before
    assert window.parameter_panel.background_color == palette_before, "the palette is never cleared"
    assert window.last_error == "boom"
    assert window.state_badge.text() == "error - still showing gen 1"
    assert "error - still showing gen 1" in window.statusBar().currentMessage()
    assert window.metrics_panel.outdated_generation == 1, "the numbers are from gen 1"
    assert window.convert_action.isEnabled() is True, "the user can retry"
    assert window.request_export() is False, "a stale result is not exportable"


def test_a_cancelled_conversion_shows_no_error(qtbot) -> None:
    window, backend, boxes = converted_window(qtbot)
    window.parameter_panel.colors_spin.setValue(2)
    qtbot.waitUntil(lambda: window.is_updating is True, timeout=2000)

    backend.emit_cancelled()

    assert boxes == []
    assert window.error_banner.isHidden() is True
    assert window.last_error is None
    assert window.displayed is not None and window.displayed.generation == 1
    assert window.state_badge.text() == "stale - showing gen 1", "no error, just a stale result"


def test_the_metrics_of_an_old_generation_are_marked_outdated(qtbot) -> None:
    window, backend, _ = converted_window(qtbot)
    assert window.metrics_panel.outdated_generation is None

    window.parameter_panel.colors_spin.setValue(2)

    assert window.metrics_panel.outdated_generation == 1
    assert window.metrics_panel.outdated_label.isHidden() is False
    assert window.metrics_panel.outdated_label.text() == "outdated (gen 1)"
    assert window.metrics_panel.metrics_form.rowCount() == 7, "the numbers are still readable"

    qtbot.waitUntil(lambda: window.is_updating is True, timeout=2000)
    backend.finish()

    assert window.metrics_panel.outdated_generation is None
    assert window.metrics_panel.outdated_label.isHidden() is True
