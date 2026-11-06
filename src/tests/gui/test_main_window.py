"""Tests for the main window on the asynchronous conversion flow (T3.3).

The window is exercised through the real :class:`ConversionController` on top of a fake backend, so the
wiring, the generation bookkeeping and the rendering are all covered without threads; the fake answers
only when a test says so, which keeps the assertions free of timing assumptions. One integration test
runs the whole stack - window, controller, ``QThreadBackend`` and the real engine - on a fixture image.
Skipped when PySide6 is missing (see ``conftest.py``).
"""

from __future__ import annotations

import dataclasses
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="PySide6 is not installed (install the 'gui' extra)")

from PySide6.QtCore import QMimeData, QObject, QPointF, Qt, QUrl, Signal  # noqa: E402
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QKeySequence  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QFileDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QSplitter,
    QToolBar,
)

import converter  # noqa: E402
import converter_gui  # noqa: E402
import converter_gui.app  # noqa: E402
from converter.types import (  # noqa: E402
    ConversionConfig,
    ConversionResult,
    ExportFormat,
    MetricResult,
    SourceIdentity,
)
from converter_gui.controllers.conversion_controller import (  # noqa: E402
    ConversionController,
    DisplayedConversion,
)
from converter_gui.main_window import (  # noqa: E402
    INITIAL_SIZE,
    MINIMUM_SIZE,
    NO_SOURCE_TEXT,
    RESULT_CAPTION,
    SHUTDOWN_TIMEOUT_MS,
    SOURCE_CAPTION,
    ExportPayload,
    MainWindow,
)
from converter_gui.widgets.image_preview import ImagePreview  # noqa: E402
from converter_gui.widgets.metrics_panel import MetricsPanel  # noqa: E402
from converter_gui.widgets.parameter_panel import ParameterPanel  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
REFERENCE_IMAGE = FIXTURES / "reference.png"

DEBOUNCE_MS = 5
"""Short on purpose: the window accepts a debounce override, the production default stays 300 ms."""

CREATED: list[MainWindow] = []
"""Every window built by a test; the fixture below never leaves a running worker behind."""


@pytest.fixture(autouse=True)
def _always_stop_the_workers():
    """Shut every window down, so no test can leak a ``QThread`` into the next one."""
    yield
    while CREATED:
        window = CREATED.pop()
        if window._shutdown_result is not True:
            window.shutdown(50)


def _metrics() -> MetricResult:
    return MetricResult(
        mse=12.5,
        psnr=27.25,
        ssim=0.9375,
        mean_rgb_error=5.0,
        mean_luminance_error=4.25,
        colors_used=4,
        histogram_coverage=0.625,
    )


def _palette(count: int = 4) -> np.ndarray:
    colours = [(10, 20, 30), (40, 50, 60), (70, 80, 90), (1, 2, 3)]
    return np.array(colours[:count], dtype=np.uint8).reshape(count, 3)


def _result(colors: int = 4, height: int = 8, width: int = 10) -> ConversionResult:
    """A tiny but complete result: two RGB images, a palette, metrics and stage timings."""
    return ConversionResult(
        indexed=np.zeros((height, width), dtype=np.uint8),
        palette_indices=list(range(colors)),
        palette_rgb=_palette(colors),
        source_resized=np.zeros((height, width, 3), dtype=np.uint8),
        metrics=_metrics(),
        timings={
            "prepare": 0.010,
            "palette": 0.002,
            "edges": 0.001,
            "render": 0.100,
            "metrics": 0.002,
        },
    )


class FakeBackend(QObject):
    """A backend that records requests and answers only when the test asks it to."""

    source_loaded = Signal(int, object)
    finished = Signal(int, object)
    failed = Signal(int, str)
    cancelled = Signal(int)

    def __init__(
        self, *, auto_load: bool = True, auto_convert: bool = False, shutdown_result: bool = True
    ) -> None:
        super().__init__()
        self.calls: list[tuple] = []
        self.auto_load = auto_load
        self.auto_convert = auto_convert
        self.shutdown_result = shutdown_result
        self.fail_loads = False
        self.running: int | None = None

    # --- the ConversionBackend protocol --------------------------------

    def load_source(self, gen: int, path: str, verify_content: bool = False) -> None:
        self.calls.append(("load", gen, str(path), verify_content))
        if not self.auto_load:
            return
        if self.fail_loads:
            self.failed.emit(gen, "unreadable image")
            return
        self.source_loaded.emit(gen, SourceIdentity(path=str(path), size=3, mtime_ns=4))

    def convert(self, gen: int, config: ConversionConfig) -> None:
        self.calls.append(("convert", gen, config))
        self.running = gen
        if self.auto_convert:
            self.finish()

    def cancel(self, gen: int) -> None:
        self.calls.append(("cancel", gen))

    def shutdown(self, timeout_ms: int) -> bool:
        self.calls.append(("shutdown", timeout_ms))
        if self.shutdown_result:
            self.running = None
        return self.shutdown_result

    # --- test controls -------------------------------------------------

    def kinds(self) -> list[str]:
        """The recorded request kinds, in order."""
        return [call[0] for call in self.calls]

    def conversions(self) -> list[int]:
        """Generations of the recorded conversions."""
        return [call[1] for call in self.calls if call[0] == "convert"]

    def finish(self, result: ConversionResult | None = None) -> None:
        gen, self.running = self.running, None
        assert gen is not None, "no conversion is running"
        self.finished.emit(gen, result if result is not None else _result())

    def fail(self, message: str = "boom") -> None:
        gen, self.running = self.running, None
        assert gen is not None, "no conversion is running"
        self.failed.emit(gen, message)

    def emit_cancelled(self) -> None:
        gen, self.running = self.running, None
        assert gen is not None, "no conversion is running"
        self.cancelled.emit(gen)

    def finish_late(self, gen: int, result: ConversionResult) -> None:
        """Answer for a generation the controller has already superseded."""
        self.finished.emit(gen, result)

    def fail_late(self, gen: int, message: str) -> None:
        self.failed.emit(gen, message)


def _window(
    qtbot, backend: FakeBackend | None = None, **kwargs
) -> tuple[MainWindow, FakeBackend, list[tuple[str, str]]]:
    """A window plus its fake backend and the modal boxes it would have opened."""
    fake = FakeBackend() if backend is None else backend
    kwargs.setdefault("debounce_ms", DEBOUNCE_MS)
    window = MainWindow(backend=fake, **kwargs)
    qtbot.addWidget(window)
    CREATED.append(window)
    boxes: list[tuple[str, str]] = []
    window._report_error = lambda title, message, **rest: boxes.append((title, message))
    window._warn_shutdown_timeout = lambda: boxes.append(("Still closing", ""))
    return window, fake, boxes


def _open(
    qtbot, path: Path = REFERENCE_IMAGE
) -> tuple[MainWindow, FakeBackend, list[tuple[str, str]]]:
    """A window with ``path`` handed to the controller (and, by default, one conversion running)."""
    window, backend, boxes = _window(qtbot)
    assert window.open_image(path) is True
    return window, backend, boxes


def _converted(
    qtbot, path: Path = REFERENCE_IMAGE
) -> tuple[MainWindow, FakeBackend, list[tuple[str, str]]]:
    """A window with one finished conversion of ``path`` on screen."""
    window, backend, boxes = _open(qtbot, path)
    backend.finish()
    return window, backend, boxes


def _drop_event(path: Path) -> tuple[QDropEvent, QMimeData]:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    event = QDropEvent(
        QPointF(10.0, 10.0),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    return event, mime


def _drag_enter_event(path: Path) -> tuple[QDragEnterEvent, QMimeData]:
    """The mime object must outlive the event: Qt keeps a pointer to it."""
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    event = QDragEnterEvent(
        QPointF(1.0, 1.0),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    return event, mime


# --- construction and wiring --------------------------------------------


def test_starts(qtbot) -> None:
    """The window can be created, registered, shown and is really visible."""
    window, _, _ = _window(qtbot)

    window.show()

    assert window.isVisible()
    assert window.windowTitle() == "PyImageConverter"
    assert (window.width(), window.height()) == INITIAL_SIZE
    assert (window.minimumWidth(), window.minimumHeight()) == MINIMUM_SIZE


def test_the_window_contains_the_components(qtbot) -> None:
    window, backend, _ = _window(qtbot)

    assert isinstance(window.preview, ImagePreview)
    assert isinstance(window.parameter_panel, ParameterPanel)
    assert isinstance(window.metrics_panel, MetricsPanel)
    assert len(window.preview.views) == 1
    assert window.centralWidget().findChildren(QSplitter) != [], "the layout is resizable"
    assert window.file_label.text() == NO_SOURCE_TEXT
    assert window.state_badge.text() == "no image"
    assert window.convert_action.isEnabled() is False
    assert window.latest_generation == 0
    assert window.displayed is None
    assert window.backend is backend


def test_the_window_owns_a_controller_on_top_of_the_backend(qtbot) -> None:
    """One backend, one controller - and the window is not an engine host any more."""
    window, backend, _ = _window(qtbot)

    assert isinstance(window.controller, ConversionController)
    assert window.controller.backend is backend
    assert window.controller.auto_update is True, "auto-update is on by default"
    assert not hasattr(window, "engine"), "the window must not own an engine (plan §7.1)"
    assert window.auto_update_action.isChecked() is True
    assert window.auto_update_action.isCheckable() is True


def test_the_menus_and_the_toolbar_expose_the_actions(qtbot) -> None:
    window, _, _ = _window(qtbot)

    menus = {menu.title().replace("&", ""): menu for menu in window.menuBar().findChildren(QMenu)}
    toolbar = window.findChild(QToolBar, "mainToolBar")

    assert set(menus) >= {"File", "Edit", "View", "Help"}
    file_actions = [
        a.text().replace("&", "") for a in menus["File"].actions() if not a.isSeparator()
    ]
    assert file_actions == ["Open…", "Exit"]
    edit_actions = [
        a.text().replace("&", "") for a in menus["Edit"].actions() if not a.isSeparator()
    ]
    assert edit_actions == ["Convert", "Auto-update"]
    view_actions = [
        a.text().replace("&", "") for a in menus["View"].actions() if not a.isSeparator()
    ]
    assert "Auto-update" in view_actions, "the toggle is reachable from the View menu"
    assert "Fit" in view_actions
    assert [a.text().replace("&", "") for a in menus["Help"].actions()] == [
        "About PyImageConverter"
    ]
    assert window.open_action.shortcut() == QKeySequence.StandardKey.Open
    assert toolbar is not None
    assert window.open_action in toolbar.actions()
    assert window.convert_action in toolbar.actions()
    assert window.auto_update_action in toolbar.actions()
    assert all(action.isEnabled() is False for action in window.zoom_actions.values())
    assert window.fit_action.isEnabled() is False


def test_the_previews_are_labelled(qtbot) -> None:
    window, _, _ = _window(qtbot)

    captions = [
        label.text()
        for label in window.centralWidget().findChildren(QLabel)
        if label.text() in (SOURCE_CAPTION, RESULT_CAPTION)
    ]

    assert captions == [SOURCE_CAPTION, RESULT_CAPTION]


# --- opening images -----------------------------------------------------


def test_opening_an_image_is_delegated_to_the_controller(qtbot) -> None:
    """The window asks the controller; the load itself never happens on the GUI thread."""
    window, backend, boxes = _window(qtbot, FakeBackend(auto_load=False))

    assert window.open_image(REFERENCE_IMAGE) is True

    assert boxes == []
    assert backend.calls == [("load", 1, str(REFERENCE_IMAGE), False)]
    assert window.latest_generation == 1, "the controller owns the generation counter"
    assert window.source_path is None, "nothing is loaded until the backend answers"
    assert window.file_label.text() == f"{REFERENCE_IMAGE.name} (loading)"
    assert window.state_badge.text() == f"loading {REFERENCE_IMAGE.name}…"
    assert window.convert_action.isEnabled() is False, "there is no source to convert yet"


def test_a_loaded_source_is_recorded_and_converted(qtbot) -> None:
    window, backend, boxes = _window(qtbot)

    window.open_image(REFERENCE_IMAGE)

    assert boxes == []
    assert window.source_path == REFERENCE_IMAGE
    assert REFERENCE_IMAGE.name in window.file_label.text()
    assert backend.kinds() == ["load", "convert"], "the controller starts the conversion"
    assert backend.conversions() == [1]
    assert window.is_updating is True
    assert window.state_badge.text() == "updating… (gen 1)"
    assert window.convert_action.isEnabled() is False, "Convert is pointless while converting"
    assert window.controller.source_identity is not None


def test_a_failed_load_changes_nothing_and_shows_the_banner(qtbot) -> None:
    """A source that cannot be read keeps the previous one and never becomes the current one."""
    window, backend, _ = _converted(qtbot)
    views_before = [view.pixmap_item.pixmap().cacheKey() for view in window.preview.views]
    backend.fail_loads = True
    broken = FIXTURES / "does-not-exist.png"

    assert window.open_image(broken) is True, "the request itself is accepted"

    assert window.error_banner.isHidden() is False
    assert "could not be loaded" in window.error_label.text()
    assert "unreadable image" in window.error_label.text()
    assert window.source_path == REFERENCE_IMAGE, "the previous source stays current"
    assert window.last_error == "unreadable image"
    assert window.displayed is not None and window.displayed.generation == 1
    assert [view.pixmap_item.pixmap().cacheKey() for view in window.preview.views] == views_before
    assert window.state_badge.text().startswith("error")


def test_the_dialog_filter_uses_the_supported_extensions(qtbot, monkeypatch) -> None:
    window, _, _ = _window(qtbot)
    seen: dict[str, str] = {}

    def fake_dialog(parent, title, directory, filters):
        seen["filters"] = filters
        return str(REFERENCE_IMAGE), ""

    monkeypatch.setattr(QFileDialog, "getOpenFileName", fake_dialog)

    window.open_dialog()

    assert "*.png" in seen["filters"]
    assert "*.bmp" in seen["filters"]
    assert "*.jpg" in seen["filters"]
    assert window.source_path == REFERENCE_IMAGE


def test_cancelling_the_open_dialog_changes_nothing(qtbot, monkeypatch) -> None:
    window, backend, _ = _window(qtbot)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("", ""))
    before = (window.latest_generation, window.displayed, window.source_path)

    window.open_dialog()

    assert (window.latest_generation, window.displayed, window.source_path) == before
    assert backend.calls == []
    assert window.convert_action.isEnabled() is False


def test_an_unsupported_extension_is_refused(qtbot, tmp_path: Path) -> None:
    window, backend, boxes = _window(qtbot)
    bogus = tmp_path / "notes.txt"
    bogus.write_text("not an image", encoding="utf-8")

    assert window.open_image(bogus) is False

    assert len(boxes) == 1
    assert boxes[0][0] == "Unsupported file"
    assert window.latest_generation == 0
    assert window.source_path is None
    assert backend.calls == [], "an unsupported file never reaches the backend"


def test_drag_and_drop_opens_a_supported_file(qtbot) -> None:
    window, _, _ = _window(qtbot)
    event, _mime = _drop_event(REFERENCE_IMAGE)

    window.dropEvent(event)

    assert window.source_path == REFERENCE_IMAGE
    assert window.latest_generation == 1


def test_drag_and_drop_ignores_other_files(qtbot, tmp_path: Path) -> None:
    window, backend, _ = _window(qtbot)
    event, _mime = _drop_event(tmp_path / "notes.txt")

    window.dropEvent(event)

    assert window.source_path is None
    assert window.latest_generation == 0
    assert backend.calls == []


def test_drag_enter_accepts_only_supported_files(qtbot, tmp_path: Path) -> None:
    window, _, _ = _window(qtbot)
    accepted, accepted_mime = _drag_enter_event(REFERENCE_IMAGE)
    rejected, rejected_mime = _drag_enter_event(tmp_path / "notes.txt")

    window.dragEnterEvent(accepted)
    window.dragEnterEvent(rejected)

    assert accepted.isAccepted() is True
    assert rejected.isAccepted() is False
    assert accepted_mime is not None and rejected_mime is not None


# --- parameters, auto-update and manual conversion ----------------------


def test_a_parameter_change_is_forwarded_to_the_controller(qtbot) -> None:
    """The panel feeds the controller directly; the window adds no comparison and no timer."""
    window, backend, _ = _converted(qtbot)
    conversions_before = backend.conversions()

    window.parameter_panel.colors_spin.setValue(2)

    assert window.requested_config.colors == 2
    assert window.latest_generation == 2, "an image setting bumps the generation"
    assert window.is_stale is True
    assert backend.conversions() == conversions_before, "the debounce has not elapsed yet"
    assert window.displayed is not None and window.displayed.generation == 1
    qtbot.waitUntil(lambda: backend.conversions() == conversions_before + [2], timeout=2000)


def test_auto_update_off_keeps_parameter_changes_from_converting(qtbot) -> None:
    window, backend, _ = _converted(qtbot)
    conversions_before = backend.conversions()

    window.auto_update_action.setChecked(False)

    assert window.controller.auto_update is False
    window.parameter_panel.colors_spin.setValue(2)
    qtbot.wait(4 * DEBOUNCE_MS)
    assert backend.conversions() == conversions_before, "nothing converts on its own"
    assert window.is_stale is True
    assert window.state_badge.text() == "stale - showing gen 1"
    assert window.convert_action.isEnabled() is True, "the user can still convert by hand"


def test_auto_update_comes_back_on(qtbot) -> None:
    window, backend, _ = _converted(qtbot)
    window.auto_update_action.setChecked(False)

    window.auto_update_action.setChecked(True)
    assert window.controller.auto_update is True
    window.parameter_panel.colors_spin.setValue(2)  # supersedes the re-enable request

    qtbot.waitUntil(lambda: 2 in backend.conversions(), timeout=2000)
    assert window.is_updating is True


def test_the_toggle_follows_the_controller(qtbot) -> None:
    """The action is a view of ``controller.auto_update``, not a second source of truth."""
    window, _, _ = _window(qtbot)

    window.controller.set_auto_update(False)

    assert window.auto_update_action.isChecked() is False


def test_manual_convert_uses_convert_now(qtbot) -> None:
    window, backend, _ = _converted(qtbot)
    window.auto_update_action.setChecked(False)
    window.parameter_panel.colors_spin.setValue(2)
    qtbot.wait(4 * DEBOUNCE_MS)
    conversions_before = backend.conversions()

    window.convert_action.trigger()

    assert backend.conversions() == conversions_before + [2]
    assert window.is_updating is True
    assert window.state_badge.text() == "updating… (gen 2)"


def test_convert_is_disabled_without_a_source(qtbot) -> None:
    window, backend, _ = _window(qtbot)

    window.convert_action.trigger()

    assert window.convert_action.isEnabled() is False
    assert backend.conversions() == []
    assert window.displayed is None
    assert window.latest_generation == 0


def test_convert_while_converting_only_keeps_one_job(qtbot) -> None:
    window, backend, _ = _converted(qtbot)
    window.parameter_panel.colors_spin.setValue(2)  # gen 2, debounced
    qtbot.waitUntil(lambda: window.is_updating is True, timeout=2000)

    window.convert_action.trigger()

    assert backend.conversions().count(2) == 1, "the controller coalesces identical requests"


# --- rendering the result -----------------------------------------------


def test_a_result_updates_the_previews_the_metrics_and_the_palette(qtbot) -> None:
    window, backend, boxes = _open(qtbot)

    backend.finish()

    displayed = window.displayed
    assert boxes == []
    assert isinstance(displayed, DisplayedConversion)
    assert displayed.generation == window.latest_generation == 1
    assert window.is_stale is False
    assert window.state_badge.text() == "ready (gen 1)"
    assert len(window.preview.views) == 2
    for view in window.preview.views:
        pixmap = view.pixmap_item.pixmap()
        assert (pixmap.width(), pixmap.height()) == (10, 8)
    assert window.metrics_panel.metrics_form.rowCount() == 7
    assert set(window.metrics_panel.timing_values) >= {"prepare", "palette", "render"}
    assert window.metrics_panel.outdated_generation is None
    assert window.parameter_panel.background_color is not None
    assert window.parameter_panel.background_color.getRgb()[:3] == (10, 20, 30), "palette[0]"
    assert "ready (gen 1)" in window.statusBar().currentMessage()


def test_the_result_arrays_are_not_modified(qtbot) -> None:
    """The engine hands out read-only arrays; the window must convert, never touch them."""
    window, backend, _ = _open(qtbot)
    result = _result()
    result.source_resized.setflags(write=False)

    backend.finish(result)

    assert result.source_resized.flags.writeable is False
    assert window.displayed is not None and window.displayed.result is result


def test_the_displayed_conversion_records_its_configuration(qtbot) -> None:
    window, _, _ = _converted(qtbot)

    displayed = window.displayed

    assert isinstance(displayed, DisplayedConversion)
    assert displayed.config == window.requested_config
    assert displayed.timings == displayed.result.timings
    assert displayed.result.metrics is not None


def test_the_status_bar_shows_the_generation_times(qtbot) -> None:
    window, _, _ = _converted(qtbot)
    displayed = window.displayed
    assert displayed is not None
    total_ms = sum(displayed.timings.values()) * 1000

    message = window.statusBar().currentMessage()

    assert window.controller.status_text() in message
    assert REFERENCE_IMAGE.name in message
    assert f"stages {total_ms:.1f} ms" in message, "the times come from the displayed result"


def test_the_badge_and_the_status_follow_the_controller(qtbot) -> None:
    window, backend, _ = _open(qtbot)
    assert window.state_badge.text() == "updating… (gen 1)"

    backend.finish()
    assert window.state_badge.text() == "ready (gen 1)"

    window.parameter_panel.colors_spin.setValue(2)
    assert window.state_badge.text() == "stale - showing gen 1"
    assert "stale - showing gen 1" in window.statusBar().currentMessage()

    qtbot.waitUntil(lambda: window.is_updating is True, timeout=2000)
    assert window.state_badge.text() == "updating… (gen 2)"
    backend.finish()
    assert window.state_badge.text() == "ready (gen 2)"


def test_the_zoom_actions_are_enabled_by_a_result(qtbot) -> None:
    window, backend, _ = _open(qtbot)
    assert all(action.isEnabled() is False for action in window.zoom_actions.values())

    backend.finish()

    assert all(action.isEnabled() for action in window.zoom_actions.values())
    assert window.fit_action.isEnabled() is True
    window.zoom_actions[4.0].trigger()
    assert window.preview.zoom == pytest.approx(4.0)
    window.fit_action.trigger()
    assert window.preview.is_fit is True


# --- generations --------------------------------------------------------


def test_a_superseded_result_is_never_shown(qtbot) -> None:
    """A late answer for an old generation must not replace the current one."""
    window, backend, _ = _converted(qtbot)
    window.parameter_panel.colors_spin.setValue(2)  # gen 2
    qtbot.waitUntil(lambda: window.is_updating is True, timeout=2000)
    backend.finish(_result(width=20, height=16))  # the current generation
    current = window.displayed
    assert current is not None and current.generation == 2

    backend.finish_late(1, _result())
    backend.fail_late(1, "late failure")

    assert window.displayed is current, "the old answer is ignored"
    assert window.preview.views[0].pixmap_item.pixmap().width() == 20
    assert window.error_banner.isHidden() is True, "a late failure shows no error"
    assert window.last_error is None


def test_the_previous_result_is_not_presented_as_the_new_source(qtbot, tmp_path: Path) -> None:
    window, backend, _ = _converted(qtbot)
    second = tmp_path / "second.png"

    assert window.open_image(second) is True

    assert window.latest_generation == 2
    assert window.displayed is not None and window.displayed.generation == 1
    assert window.is_stale is True
    assert len(window.preview.views) == 2, "the old picture stays until the new one is ready"

    backend.finish(_result(width=20, height=16))

    assert window.displayed.generation == 2
    assert window.is_stale is False
    assert window.preview.views[0].pixmap_item.pixmap().width() == 20


def test_staleness_comes_from_generations_not_config_equality(qtbot) -> None:
    window, _, _ = _converted(qtbot)
    original_colors = window.requested_config.colors

    window.parameter_panel.colors_spin.setValue(2)
    window.parameter_panel.colors_spin.setValue(original_colors)

    assert window.displayed is not None
    assert window.requested_config == window.displayed.config, "the configurations match again"
    assert window.is_stale is True, "but the generation moved on, so the display is stale"


def test_the_metrics_of_an_old_generation_are_marked_outdated(qtbot) -> None:
    window, backend, _ = _converted(qtbot)
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


# --- failures -----------------------------------------------------------


def test_a_failed_conversion_keeps_the_previous_result(qtbot) -> None:
    window, backend, boxes = _converted(qtbot)
    displayed = window.displayed
    metrics_before = window.metrics_panel.metric_values
    timings_before = window.metrics_panel.timing_values
    window.parameter_panel.colors_spin.setValue(2)  # gen 2
    qtbot.waitUntil(lambda: window.is_updating is True, timeout=2000)

    backend.fail("boom")

    assert boxes == [], "a conversion error is not modal (plan §7.6)"
    assert window.error_banner.isHidden() is False
    assert "The conversion failed" in window.error_label.text()
    assert "boom" in window.error_label.text()
    assert window.displayed is displayed, "the last good result must survive a failure"
    assert displayed is not None and displayed.generation == 1
    assert window.metrics_panel.metric_values == metrics_before
    assert window.metrics_panel.timing_values == timings_before
    assert window.last_error == "boom"
    assert window.state_badge.text() == "error - still showing gen 1"
    assert "error - still showing gen 1" in window.statusBar().currentMessage()
    assert window.metrics_panel.outdated_generation == 1, "the numbers are from gen 1"
    assert window.convert_action.isEnabled() is True, "the user can retry"
    assert window.request_export() is False, "a stale result is not exportable"


def test_the_error_banner_can_be_dismissed_and_asks_for_details(qtbot) -> None:
    window, backend, _ = _open(qtbot)
    backend.fail("boom")
    details: list[int] = []
    window.show_error_details = lambda: details.append(1)

    window.error_details_button.click()
    assert window.error_banner.isHidden() is False, "asking for details keeps the banner"

    window.error_dismiss_button.click()
    assert window.error_banner.isHidden() is True
    assert window.error_label.text() == ""
    assert details == [1]


def test_the_error_disappears_after_the_next_good_result(qtbot) -> None:
    window, backend, _ = _open(qtbot)
    backend.fail("boom")
    assert window.error_banner.isHidden() is False

    window.convert_action.trigger()
    backend.finish()

    assert window.last_error is None
    assert window.error_banner.isHidden() is True
    assert window.is_stale is False
    assert window.state_badge.text() == "ready (gen 1)"


def test_a_cancelled_conversion_shows_no_error(qtbot) -> None:
    window, backend, boxes = _converted(qtbot)
    window.parameter_panel.colors_spin.setValue(2)
    qtbot.waitUntil(lambda: window.is_updating is True, timeout=2000)

    backend.emit_cancelled()

    assert boxes == []
    assert window.error_banner.isHidden() is True
    assert window.last_error is None
    assert window.displayed is not None and window.displayed.generation == 1
    assert window.state_badge.text() == "stale - showing gen 1", "no error, just a stale result"


# --- export -------------------------------------------------------------


def test_export_is_blocked_without_a_current_result(qtbot) -> None:
    window, backend, _ = _window(qtbot)
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
    window, backend, _ = _converted(qtbot)
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
    assert window.request_export() is True

    payload = emitted[0]
    assert payload.displayed is window.displayed
    assert payload.displayed.config.export_formats == [ExportFormat.BIN], "the image config"
    assert payload.export_config.export_formats == [ExportFormat.PNG, ExportFormat.ASM]
    assert payload.export_config.atari_colors == [0x11, 0x22]
    assert payload.export_config.label_name == "menu"
    assert backend.conversions() == [1]


def test_the_export_signal_is_the_isolated_hook_for_the_export_window(qtbot) -> None:
    """T4.4 will connect to ``MainWindow.export_requested``; until then nothing else happens."""
    window, backend, _ = _converted(qtbot)
    emitted: list[ExportPayload] = []
    window.export_requested.connect(emitted.append)

    window.convert_action.trigger()  # a no-op: gen 1 is already displayed
    assert emitted == []

    window.request_export()
    assert len(emitted) == 1
    assert isinstance(emitted[0].displayed, DisplayedConversion)


# --- small actions ------------------------------------------------------


def test_the_about_dialog_reports_the_version(qtbot, monkeypatch) -> None:
    window, _, _ = _window(qtbot)
    captured: dict[str, str] = {}
    monkeypatch.setattr(
        QMessageBox,
        "about",
        lambda parent, title, text: captured.update(title=title, text=text),
    )

    window.show_about()

    assert converter.__version__ in captured["text"]
    assert "PyImageConverter" in captured["title"]


def test_the_exit_action_closes_the_window(qtbot) -> None:
    window, backend, _ = _window(qtbot)
    window.show()
    assert window.isVisible() is True

    window.exit_action.trigger()

    assert window.isVisible() is False
    assert backend.calls[-1][0] == "shutdown"


# --- shutdown -----------------------------------------------------------


def test_closing_stops_the_controller_and_the_backend(qtbot) -> None:
    window, backend, boxes = _converted(qtbot)
    window.show()

    window.close()

    assert window.isVisible() is False
    assert boxes == []
    assert backend.calls[-1] == ("shutdown", SHUTDOWN_TIMEOUT_MS)
    assert window._shutdown_result is True
    assert window.backend is backend, "the window keeps its backend until it is destroyed"
    assert window.controller.status_text() == "shut down"


def test_a_shutdown_timeout_keeps_the_window_and_the_worker_alive(qtbot) -> None:
    window, backend, boxes = _open(qtbot)
    window.show()
    backend.shutdown_result = False

    window.close()

    assert window.isVisible() is True, "a running worker must not be destroyed"
    assert window._shutdown_result is False
    assert boxes == [("Still closing", "")], "the user is told the close did not finish"
    assert window.backend is backend
    assert "shutting down" in window.statusBar().currentMessage()

    backend.shutdown_result = True
    window.close()

    assert window.isVisible() is False, "the second attempt succeeds"
    assert window._shutdown_result is True
    assert sum(1 for kind, *_ in backend.calls if kind == "shutdown") == 2


def test_late_signals_do_not_touch_a_closed_window(qtbot) -> None:
    window, backend, boxes = _open(qtbot)
    window.show()
    backend.shutdown_result = False
    window.close()
    views_before = [view.pixmap_item.pixmap().cacheKey() for view in window.preview.views]

    backend.finished.emit(1, _result(width=20, height=16))
    backend.failed.emit(1, "too late")
    backend.cancelled.emit(1)

    assert window.displayed is None
    assert [view.pixmap_item.pixmap().cacheKey() for view in window.preview.views] == views_before
    assert window.error_banner.isHidden() is True
    assert boxes == [("Still closing", "")], "nothing else was reported"


def test_requests_are_refused_once_the_shutdown_started(qtbot) -> None:
    window, backend, _ = _open(qtbot)
    window.show()
    backend.shutdown_result = False
    window.close()
    calls_before = len(backend.calls)

    assert window.open_image(REFERENCE_IMAGE) is False
    assert window.request_export() is False
    window.convert_now()

    assert len(backend.calls) == calls_before, "no new work reaches a stopping backend"


# --- the whole stack ----------------------------------------------------


def test_the_real_backend_converts_a_fixture_image(qtbot) -> None:
    """Window, controller, ``QThreadBackend`` and the real engine, end to end."""
    window = MainWindow(debounce_ms=DEBOUNCE_MS)
    qtbot.addWidget(window)
    CREATED.append(window)
    window.show()

    assert window.open_image(REFERENCE_IMAGE) is True
    qtbot.waitUntil(lambda: window.displayed is not None, timeout=60000)

    displayed = window.displayed
    assert displayed is not None
    assert displayed.generation == window.latest_generation == 1
    assert displayed.config == window.requested_config
    assert window.is_stale is False
    assert window.state_badge.text() == "ready (gen 1)"
    assert len(window.preview.views) == 2
    for view in window.preview.views:
        pixmap = view.pixmap_item.pixmap()
        assert (pixmap.width(), pixmap.height()) == (
            displayed.config.width,
            displayed.config.height,
        )
    assert window.metrics_panel.metrics_form.rowCount() == 7
    assert window.metrics_panel.outdated_generation is None
    assert "stages" in window.statusBar().currentMessage()

    window.close()

    assert window.isVisible() is False
    assert window.backend.is_running is False
    assert window._shutdown_result is True


# --- entry points and isolation -----------------------------------------


def test_create_application_sets_the_metadata(qtbot) -> None:
    application = converter_gui.app.create_application([])

    assert isinstance(application, QApplication)
    assert application.applicationName() == "PyImageConverter"
    assert application.organizationName() == "PyImageConverter"


def test_main_shows_the_window_and_returns_the_event_loop_code(qtbot, monkeypatch) -> None:
    """``main()`` builds a visible window; ``exec()`` is stubbed so no loop is entered."""
    seen: dict[str, list[MainWindow]] = {}

    def fake_exec(self) -> int:
        seen["windows"] = [
            widget
            for widget in QApplication.topLevelWidgets()
            if isinstance(widget, MainWindow) and widget.isVisible()
        ]
        return 7

    monkeypatch.setattr(QApplication, "exec", fake_exec)

    exit_code = converter_gui.app.main([])

    assert exit_code == 7
    assert len(seen["windows"]) == 1
    window = seen["windows"][0]
    qtbot.addWidget(window)
    assert window.windowTitle() == "PyImageConverter"
    window.close()
    assert window._shutdown_result is True


def test_module_entry_point_delegates_to_main(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(converter_gui.app, "main", lambda: calls.append("main") or 3)

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("converter_gui", run_name="__main__")

    assert exit_info.value.code == 3
    assert calls == ["main"]


def test_root_launcher_delegates_to_main(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(converter_gui.app, "main", lambda: calls.append("main") or 4)

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(REPO_ROOT / "convert_gui.py"), run_name="__main__")

    assert exit_info.value.code == 4
    assert calls == ["main"]


def test_importing_the_package_does_not_import_pyside6() -> None:
    """The package itself must stay importable without Qt; only the modules inside it need it."""
    probe = (
        "import json, sys\n"
        "import converter_gui\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('PySide6'))))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    assert completed.stdout.strip().endswith("[]"), completed.stdout


def test_the_window_module_imports_no_engine() -> None:
    """The window talks to the backend only; importing the engine here would invite sync calls."""
    source = (SRC / "converter_gui" / "main_window.py").read_text(encoding="utf-8")

    assert "from converter.engine" not in source
    assert "ConversionEngine(" not in source
    assert "controller.set_source" in source
    assert "controller.convert_now" in source
