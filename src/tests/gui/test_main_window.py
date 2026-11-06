"""Tests for the main window on the asynchronous conversion flow (T3.3).

The window is exercised through the real :class:`ConversionController` on top of a fake backend, so the
wiring, the generation bookkeeping and the rendering are all covered without threads; the fake answers
only when a test says so, which keeps the assertions free of timing assumptions. One integration test
runs the whole stack - window, controller, ``QThreadBackend`` and the real engine - on a fixture image.
The display rules of plan §7.5/§7.6 (stale guard, export gate, failure handling) live in
``test_stale_guard.py``. Skipped when PySide6 is missing (see ``conftest.py``).
"""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6", reason="PySide6 is not installed (install the 'gui' extra)")

from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl  # noqa: E402
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
    MainWindow,
)
from converter_gui.widgets.image_preview import ImagePreview  # noqa: E402
from converter_gui.widgets.metrics_panel import MetricsPanel  # noqa: E402
from converter_gui.widgets.parameter_panel import ParameterPanel  # noqa: E402

from gui_fakes import (  # noqa: E402
    CREATED,
    DEBOUNCE_MS,
    FIXTURES,
    REFERENCE_IMAGE,
    FakeBackend,
    build_window,
    converted_window,
    make_result,
    open_window,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"


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
    window, _, _ = build_window(qtbot)

    window.show()

    assert window.isVisible()
    assert window.windowTitle() == "PyImageConverter"
    assert (window.width(), window.height()) == INITIAL_SIZE
    assert (window.minimumWidth(), window.minimumHeight()) == MINIMUM_SIZE


def test_the_window_contains_the_components(qtbot) -> None:
    window, backend, _ = build_window(qtbot)

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
    window, backend, _ = build_window(qtbot)

    assert isinstance(window.controller, ConversionController)
    assert window.controller.backend is backend
    assert window.controller.auto_update is True, "auto-update is on by default"
    assert not hasattr(window, "engine"), "the window must not own an engine (plan §7.1)"
    assert window.auto_update_action.isChecked() is True
    assert window.auto_update_action.isCheckable() is True


def test_the_menus_and_the_toolbar_expose_the_actions(qtbot) -> None:
    window, _, _ = build_window(qtbot)

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
    window, _, _ = build_window(qtbot)

    captions = [
        label.text()
        for label in window.centralWidget().findChildren(QLabel)
        if label.text() in (SOURCE_CAPTION, RESULT_CAPTION)
    ]

    assert captions == [SOURCE_CAPTION, RESULT_CAPTION]


# --- opening images -----------------------------------------------------


def test_opening_an_image_is_delegated_to_the_controller(qtbot) -> None:
    """The window asks the controller; the load itself never happens on the GUI thread."""
    window, backend, boxes = build_window(qtbot, FakeBackend(auto_load=False))

    assert window.open_image(REFERENCE_IMAGE) is True

    assert boxes == []
    assert backend.calls == [("load", 1, str(REFERENCE_IMAGE), False)]
    assert window.latest_generation == 1, "the controller owns the generation counter"
    assert window.source_path is None, "nothing is loaded until the backend answers"
    assert window.file_label.text() == f"{REFERENCE_IMAGE.name} (loading)"
    assert window.state_badge.text() == f"loading {REFERENCE_IMAGE.name}…"
    assert window.convert_action.isEnabled() is False, "there is no source to convert yet"


def test_a_loaded_source_is_recorded_and_converted(qtbot) -> None:
    window, backend, boxes = build_window(qtbot)

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
    window, backend, _ = converted_window(qtbot)
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
    window, _, _ = build_window(qtbot)
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
    window, backend, _ = build_window(qtbot)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("", ""))
    before = (window.latest_generation, window.displayed, window.source_path)

    window.open_dialog()

    assert (window.latest_generation, window.displayed, window.source_path) == before
    assert backend.calls == []
    assert window.convert_action.isEnabled() is False


def test_an_unsupported_extension_is_refused(qtbot, tmp_path: Path) -> None:
    window, backend, boxes = build_window(qtbot)
    bogus = tmp_path / "notes.txt"
    bogus.write_text("not an image", encoding="utf-8")

    assert window.open_image(bogus) is False

    assert len(boxes) == 1
    assert boxes[0][0] == "Unsupported file"
    assert window.latest_generation == 0
    assert window.source_path is None
    assert backend.calls == [], "an unsupported file never reaches the backend"


def test_drag_and_drop_opens_a_supported_file(qtbot) -> None:
    window, _, _ = build_window(qtbot)
    event, _mime = _drop_event(REFERENCE_IMAGE)

    window.dropEvent(event)

    assert window.source_path == REFERENCE_IMAGE
    assert window.latest_generation == 1


def test_drag_and_drop_ignores_other_files(qtbot, tmp_path: Path) -> None:
    window, backend, _ = build_window(qtbot)
    event, _mime = _drop_event(tmp_path / "notes.txt")

    window.dropEvent(event)

    assert window.source_path is None
    assert window.latest_generation == 0
    assert backend.calls == []


def test_drag_enter_accepts_only_supported_files(qtbot, tmp_path: Path) -> None:
    window, _, _ = build_window(qtbot)
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
    window, backend, _ = converted_window(qtbot)
    conversions_before = backend.conversions()

    window.parameter_panel.colors_spin.setValue(2)

    assert window.requested_config.colors == 2
    assert window.latest_generation == 2, "an image setting bumps the generation"
    assert window.is_stale is True
    assert backend.conversions() == conversions_before, "the debounce has not elapsed yet"
    assert window.displayed is not None and window.displayed.generation == 1
    qtbot.waitUntil(lambda: backend.conversions() == conversions_before + [2], timeout=2000)


def test_auto_update_off_keeps_parameter_changes_from_converting(qtbot) -> None:
    window, backend, _ = converted_window(qtbot)
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
    window, backend, _ = converted_window(qtbot)
    window.auto_update_action.setChecked(False)

    window.auto_update_action.setChecked(True)
    assert window.controller.auto_update is True
    window.parameter_panel.colors_spin.setValue(2)  # supersedes the re-enable request

    qtbot.waitUntil(lambda: 2 in backend.conversions(), timeout=2000)
    assert window.is_updating is True


def test_the_toggle_follows_the_controller(qtbot) -> None:
    """The action is a view of ``controller.auto_update``, not a second source of truth."""
    window, _, _ = build_window(qtbot)

    window.controller.set_auto_update(False)

    assert window.auto_update_action.isChecked() is False


def test_manual_convert_uses_convert_now(qtbot) -> None:
    window, backend, _ = converted_window(qtbot)
    window.auto_update_action.setChecked(False)
    window.parameter_panel.colors_spin.setValue(2)
    qtbot.wait(4 * DEBOUNCE_MS)
    conversions_before = backend.conversions()

    window.convert_action.trigger()

    assert backend.conversions() == conversions_before + [2]
    assert window.is_updating is True
    assert window.state_badge.text() == "updating… (gen 2)"


def test_convert_is_disabled_without_a_source(qtbot) -> None:
    window, backend, _ = build_window(qtbot)

    window.convert_action.trigger()

    assert window.convert_action.isEnabled() is False
    assert backend.conversions() == []
    assert window.displayed is None
    assert window.latest_generation == 0


def test_convert_while_converting_only_keeps_one_job(qtbot) -> None:
    window, backend, _ = converted_window(qtbot)
    window.parameter_panel.colors_spin.setValue(2)  # gen 2, debounced
    qtbot.waitUntil(lambda: window.is_updating is True, timeout=2000)

    window.convert_action.trigger()

    assert backend.conversions().count(2) == 1, "the controller coalesces identical requests"


# --- rendering the result -----------------------------------------------


def test_a_result_updates_the_previews_the_metrics_and_the_palette(qtbot) -> None:
    window, backend, boxes = open_window(qtbot)

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
    window, backend, _ = open_window(qtbot)
    result = make_result()
    result.source_resized.setflags(write=False)

    backend.finish(result)

    assert result.source_resized.flags.writeable is False
    assert window.displayed is not None and window.displayed.result is result


def test_the_displayed_conversion_records_its_configuration(qtbot) -> None:
    window, _, _ = converted_window(qtbot)

    displayed = window.displayed

    assert isinstance(displayed, DisplayedConversion)
    assert displayed.config == window.requested_config
    assert displayed.timings == displayed.result.timings
    assert displayed.result.metrics is not None


def test_the_status_bar_shows_the_generation_times(qtbot) -> None:
    window, _, _ = converted_window(qtbot)
    displayed = window.displayed
    assert displayed is not None
    total_ms = sum(displayed.timings.values()) * 1000

    message = window.statusBar().currentMessage()

    assert window.controller.status_text() in message
    assert REFERENCE_IMAGE.name in message
    assert f"stages {total_ms:.1f} ms" in message, "the times come from the displayed result"


def test_the_badge_and_the_status_follow_the_controller(qtbot) -> None:
    window, backend, _ = open_window(qtbot)
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
    window, backend, _ = open_window(qtbot)
    assert all(action.isEnabled() is False for action in window.zoom_actions.values())

    backend.finish()

    assert all(action.isEnabled() for action in window.zoom_actions.values())
    assert window.fit_action.isEnabled() is True
    window.zoom_actions[4.0].trigger()
    assert window.preview.zoom == pytest.approx(4.0)
    window.fit_action.trigger()
    assert window.preview.is_fit is True


# --- changing the source ------------------------------------------------


def test_the_previous_result_is_not_presented_as_the_new_source(qtbot, tmp_path: Path) -> None:
    window, backend, _ = converted_window(qtbot)
    second = tmp_path / "second.png"

    assert window.open_image(second) is True

    assert window.latest_generation == 2
    assert window.displayed is not None and window.displayed.generation == 1
    assert window.is_stale is True
    assert len(window.preview.views) == 2, "the old picture stays until the new one is ready"

    backend.finish(make_result(width=20, height=16))

    assert window.displayed.generation == 2
    assert window.is_stale is False
    assert window.preview.views[0].pixmap_item.pixmap().width() == 20


# --- the error banner ---------------------------------------------------


def test_the_error_banner_can_be_dismissed_and_asks_for_details(qtbot) -> None:
    window, backend, _ = open_window(qtbot)
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
    window, backend, _ = open_window(qtbot)
    backend.fail("boom")
    assert window.error_banner.isHidden() is False

    window.convert_action.trigger()
    backend.finish()

    assert window.last_error is None
    assert window.error_banner.isHidden() is True
    assert window.is_stale is False
    assert window.state_badge.text() == "ready (gen 1)"


# --- small actions ------------------------------------------------------


def test_the_about_dialog_reports_the_version(qtbot, monkeypatch) -> None:
    window, _, _ = build_window(qtbot)
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
    window, backend, _ = build_window(qtbot)
    window.show()
    assert window.isVisible() is True

    window.exit_action.trigger()

    assert window.isVisible() is False
    assert backend.calls[-1][0] == "shutdown"


# --- shutdown -----------------------------------------------------------


def test_closing_stops_the_controller_and_the_backend(qtbot) -> None:
    window, backend, boxes = converted_window(qtbot)
    window.show()

    window.close()

    assert window.isVisible() is False
    assert boxes == []
    assert backend.calls[-1] == ("shutdown", SHUTDOWN_TIMEOUT_MS)
    assert window._shutdown_result is True
    assert window.backend is backend, "the window keeps its backend until it is destroyed"
    assert window.controller.status_text() == "shut down"


def test_a_shutdown_timeout_keeps_the_window_and_the_worker_alive(qtbot) -> None:
    window, backend, boxes = open_window(qtbot)
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
    window, backend, boxes = open_window(qtbot)
    window.show()
    backend.shutdown_result = False
    window.close()
    views_before = [view.pixmap_item.pixmap().cacheKey() for view in window.preview.views]

    backend.finished.emit(1, make_result(width=20, height=16))
    backend.failed.emit(1, "too late")
    backend.cancelled.emit(1)

    assert window.displayed is None
    assert [view.pixmap_item.pixmap().cacheKey() for view in window.preview.views] == views_before
    assert window.error_banner.isHidden() is True
    assert boxes == [("Still closing", "")], "nothing else was reported"


def test_requests_are_refused_once_the_shutdown_started(qtbot) -> None:
    window, backend, _ = open_window(qtbot)
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
