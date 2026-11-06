"""Tests for the integrated main window (T2.6).

They exercise the real flow - open a fixture through the injected path, convert with the real engine,
inspect both previews, the metrics panel and the generation/stale bookkeeping - plus the failure paths
and the entry points of the application. Skipped when PySide6 is missing (see ``conftest.py``).
"""

from __future__ import annotations

import json
import os
import runpy
import shutil
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
from converter.engine import ConversionEngine  # noqa: E402
from converter_gui.main_window import (  # noqa: E402
    INITIAL_SIZE,
    MINIMUM_SIZE,
    NO_SOURCE_TEXT,
    RESULT_CAPTION,
    SOURCE_CAPTION,
    DisplayedConversion,
    MainWindow,
)
from converter_gui.widgets.image_preview import ImagePreview  # noqa: E402
from converter_gui.widgets.metrics_panel import MetricsPanel  # noqa: E402
from converter_gui.widgets.parameter_panel import ParameterPanel  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
REFERENCE_IMAGE = Path(__file__).resolve().parents[1] / "fixtures" / "reference.png"


def _window(
    qtbot, engine: ConversionEngine | None = None
) -> tuple[MainWindow, list[tuple[str, str]]]:
    """A window with the modal error box replaced by a recorder, so tests stay headless."""
    window = MainWindow(engine=engine)
    qtbot.addWidget(window)
    errors: list[tuple[str, str]] = []
    window._report_error = lambda title, message, details=None: errors.append((title, message))
    return window, errors


def _open(qtbot, path: Path = REFERENCE_IMAGE) -> tuple[MainWindow, list[tuple[str, str]]]:
    window, errors = _window(qtbot)
    assert window.open_image(path) is True
    return window, errors


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


# --- construction and actions -------------------------------------------


def test_starts(qtbot) -> None:
    """The window can be created, registered, shown and is really visible."""
    window, _ = _window(qtbot)

    window.show()

    assert window.isVisible()
    assert window.windowTitle() == "PyImageConverter"
    assert (window.width(), window.height()) == INITIAL_SIZE
    assert (window.minimumWidth(), window.minimumHeight()) == MINIMUM_SIZE


def test_the_window_contains_the_components(qtbot) -> None:
    window, _ = _window(qtbot)

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


def test_the_menus_and_the_toolbar_expose_the_actions(qtbot) -> None:
    window, _ = _window(qtbot)

    menus = {menu.title().replace("&", ""): menu for menu in window.menuBar().findChildren(QMenu)}
    toolbar = window.findChild(QToolBar, "mainToolBar")

    assert set(menus) >= {"File", "Edit", "View", "Help"}
    file_actions = [
        a.text().replace("&", "") for a in menus["File"].actions() if not a.isSeparator()
    ]
    assert file_actions == ["Open…", "Exit"]
    assert [a.text().replace("&", "") for a in menus["Edit"].actions()] == ["Convert"]
    assert [a.text().replace("&", "") for a in menus["Help"].actions()] == [
        "About PyImageConverter"
    ]
    assert window.open_action.shortcut() == QKeySequence.StandardKey.Open
    assert toolbar is not None
    assert window.open_action in toolbar.actions()
    assert window.convert_action in toolbar.actions()
    assert all(action.isEnabled() is False for action in window.zoom_actions.values())
    assert window.fit_action.isEnabled() is False


def test_the_previews_are_labelled(qtbot) -> None:
    window, _ = _window(qtbot)

    captions = [
        label.text()
        for label in window.centralWidget().findChildren(QLabel)
        if label.text() in (SOURCE_CAPTION, RESULT_CAPTION)
    ]

    assert captions == [SOURCE_CAPTION, RESULT_CAPTION]


# --- opening images -----------------------------------------------------


def test_opening_an_image_loads_the_source_into_the_engine(qtbot) -> None:
    window, errors = _window(qtbot)

    assert window.open_image(REFERENCE_IMAGE) is True

    assert errors == []
    assert window.source_path == REFERENCE_IMAGE
    assert window.engine.source is not None
    assert window.latest_generation == 1
    assert window.is_stale is True, "no conversion exists for the new source yet"
    assert window.convert_action.isEnabled() is True
    assert all(action.isEnabled() for action in window.zoom_actions.values())
    assert window.fit_action.isEnabled() is True
    assert REFERENCE_IMAGE.name in window.file_label.text()
    assert window.state_badge.text().startswith("stale")


def test_the_dialog_filter_uses_the_supported_extensions(qtbot, monkeypatch) -> None:
    window, _ = _window(qtbot)
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
    window, _ = _window(qtbot)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("", ""))
    before = (window.latest_generation, window.displayed, window.source_path)

    window.open_dialog()

    assert (window.latest_generation, window.displayed, window.source_path) == before
    assert window.convert_action.isEnabled() is False


def test_an_unsupported_extension_is_refused(qtbot, tmp_path: Path) -> None:
    window, errors = _window(qtbot)
    bogus = tmp_path / "notes.txt"
    bogus.write_text("not an image", encoding="utf-8")

    assert window.open_image(bogus) is False

    assert len(errors) == 1
    assert errors[0][0] == "Unsupported file"
    assert window.latest_generation == 0
    assert window.source_path is None


def test_a_broken_image_is_reported_and_changes_nothing(qtbot, tmp_path: Path) -> None:
    window, errors = _window(qtbot)
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"definitely not a png")

    assert window.open_image(broken) is False

    assert len(errors) == 1
    assert errors[0][0] == "Could not open image"
    assert window.source_path is None
    assert window.latest_generation == 0


def test_drag_and_drop_opens_a_supported_file(qtbot) -> None:
    window, _ = _window(qtbot)
    event, _mime = _drop_event(REFERENCE_IMAGE)

    window.dropEvent(event)

    assert window.source_path == REFERENCE_IMAGE
    assert window.latest_generation == 1


def test_drag_and_drop_ignores_other_files(qtbot, tmp_path: Path) -> None:
    window, _ = _window(qtbot)
    event, _mime = _drop_event(tmp_path / "notes.txt")

    window.dropEvent(event)

    assert window.source_path is None
    assert window.latest_generation == 0


def test_drag_enter_accepts_only_supported_files(qtbot, tmp_path: Path) -> None:
    window, _ = _window(qtbot)
    accepted, accepted_mime = _drag_enter_event(REFERENCE_IMAGE)
    rejected, rejected_mime = _drag_enter_event(tmp_path / "notes.txt")

    window.dragEnterEvent(accepted)
    window.dragEnterEvent(rejected)

    assert accepted.isAccepted() is True
    assert rejected.isAccepted() is False
    assert accepted_mime is not None and rejected_mime is not None


# --- conversion ---------------------------------------------------------


def test_conversion_displays_both_previews_and_the_metrics(qtbot) -> None:
    window, errors = _open(qtbot)

    assert window.convert() is True

    displayed = window.displayed
    assert errors == []
    assert displayed is not None
    assert displayed.generation == window.latest_generation == 1
    assert window.is_stale is False
    assert window.state_badge.text() == "ready (gen 1)"
    assert len(window.preview.views) == 2
    for view in window.preview.views:
        pixmap = view.pixmap_item.pixmap()
        assert (pixmap.width(), pixmap.height()) == (160, 192)
    assert window.metrics_panel.metrics_form.rowCount() == 7
    assert set(window.metrics_panel.timing_values) >= {"prepare", "palette", "render"}
    assert displayed.elapsed > 0
    assert "ready" in window.statusBar().currentMessage()


def test_the_displayed_conversion_records_its_configuration(qtbot) -> None:
    window, _ = _open(qtbot)
    window.convert()

    displayed = window.displayed

    assert isinstance(displayed, DisplayedConversion)
    assert displayed.config == window.requested_config
    assert displayed.timings == displayed.result.timings
    assert displayed.result.metrics is not None


def test_a_config_change_marks_the_result_stale_without_converting(qtbot) -> None:
    window, _ = _open(qtbot)
    window.convert()
    calls: list[int] = []
    original = window.engine.convert
    window.engine.convert = lambda *args, **kwargs: calls.append(1) or original(*args, **kwargs)

    window.parameter_panel.colors_spin.setValue(2)

    assert calls == [], "a parameter edit must not convert on its own"
    assert window.latest_generation == 2
    assert window.is_stale is True
    assert window.displayed is not None and window.displayed.generation == 1
    assert window.state_badge.text() == "stale - showing gen 1"


def test_converting_again_restores_the_ready_state(qtbot) -> None:
    window, _ = _open(qtbot)
    window.convert()
    window.parameter_panel.colors_spin.setValue(2)

    assert window.convert() is True

    assert window.displayed is not None
    assert window.displayed.generation == window.latest_generation == 2
    assert window.displayed.config.colors == 2
    assert window.is_stale is False
    assert window.state_badge.text() == "ready (gen 2)"


def test_staleness_comes_from_generations_not_config_equality(qtbot) -> None:
    window, _ = _open(qtbot)
    window.convert()
    original_colors = window.requested_config.colors

    window.parameter_panel.colors_spin.setValue(2)
    window.parameter_panel.colors_spin.setValue(original_colors)

    assert window.displayed is not None
    assert window.requested_config == window.displayed.config, "the configurations match again"
    assert window.is_stale is True, "but the generation moved on, so the display is stale"


def test_opening_another_image_keeps_the_previous_preview_but_marks_it_stale(
    qtbot, tmp_path: Path
) -> None:
    window, _ = _open(qtbot)
    window.convert()
    displayed = window.displayed
    second = tmp_path / "copy.png"
    shutil.copyfile(REFERENCE_IMAGE, second)

    assert window.open_image(second) is True

    assert window.displayed is displayed, "the last good preview is kept"
    assert window.latest_generation == 2
    assert window.is_stale is True
    assert len(window.preview.views) == 2
    assert second.name in window.file_label.text()


def test_the_updating_state_is_visible_while_converting(qtbot) -> None:
    window, _ = _open(qtbot)
    seen: dict[str, object] = {}
    original = window.engine.convert

    def spy(*args, **kwargs):
        seen["badge"] = window.state_badge.text()
        seen["status"] = window.statusBar().currentMessage()
        seen["convert_action"] = window.convert_action.isEnabled()
        return original(*args, **kwargs)

    window.engine.convert = spy

    assert window.convert() is True

    assert seen["badge"] == "updating… (gen 1)"
    assert "converting" in str(seen["status"])
    assert seen["convert_action"] is False


def test_convert_without_a_source_does_nothing(qtbot) -> None:
    window, _ = _window(qtbot)

    assert window.convert() is False

    assert window.displayed is None
    assert window.latest_generation == 0


def test_convert_is_not_reentrant(qtbot) -> None:
    window, _ = _open(qtbot)
    nested: list[bool] = []
    original = window.engine.convert

    def reentrant(*args, **kwargs):
        nested.append(window.convert())
        return original(*args, **kwargs)

    window.engine.convert = reentrant

    assert window.convert() is True

    assert nested == [False], "the nested attempt is refused instead of converting twice"


# --- failures -----------------------------------------------------------


def test_a_failed_conversion_keeps_the_previous_result(qtbot) -> None:
    window, errors = _open(qtbot)
    window.convert()
    displayed = window.displayed
    metrics_before = window.metrics_panel.metric_values
    window.engine.convert = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    window.parameter_panel.colors_spin.setValue(2)  # gen 2

    assert window.convert() is False

    assert window.displayed is displayed, "the last good result must survive a failure"
    assert displayed is not None and displayed.generation == 1
    assert window.metrics_panel.metric_values == metrics_before
    assert errors[-1][0] == "Conversion failed"
    assert "boom" in errors[-1][1]
    assert window.last_error == "boom"
    assert window.state_badge.text().startswith("error")
    assert "still showing gen 1" in window.statusBar().currentMessage()
    assert window.convert_action.isEnabled() is True, "the user can retry"


def test_the_previous_state_returns_after_a_successful_retry(qtbot) -> None:
    window, _ = _open(qtbot)
    window.convert()
    window.engine.convert = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    window.convert()

    del window.engine.convert  # back to the real method
    assert window.convert() is True

    assert window.last_error is None
    assert window.is_stale is False
    assert window.state_badge.text().startswith("ready")


def test_the_wait_cursor_is_always_restored(qtbot) -> None:
    window, _ = _open(qtbot)

    assert window.convert() is True
    assert QApplication.overrideCursor() is None

    window.engine.convert = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    assert window.convert() is False
    assert QApplication.overrideCursor() is None, "the cursor is restored even on failure"


# --- small actions ------------------------------------------------------


def test_the_about_dialog_reports_the_version(qtbot, monkeypatch) -> None:
    window, _ = _window(qtbot)
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
    window, _ = _window(qtbot)
    window.show()
    assert window.isVisible() is True

    window.exit_action.trigger()

    assert window.isVisible() is False


def test_the_zoom_actions_drive_the_preview(qtbot) -> None:
    window, _ = _open(qtbot)

    window.zoom_actions[4.0].trigger()
    assert window.preview.zoom == pytest.approx(4.0)

    window.fit_action.trigger()
    assert window.preview.is_fit is True


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
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip()) == []
