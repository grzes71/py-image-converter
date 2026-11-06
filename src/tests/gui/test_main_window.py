"""Smoke tests for the T2.1 window and its entry points.

They check the real behaviour of the running application object: a visible ``QMainWindow`` with the
expected title and placeholder page, and entry points that reach :func:`converter_gui.app.main`
without entering a persistent event loop. Skipped when PySide6 is missing (see ``conftest.py``).
"""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6", reason="PySide6 is not installed (install the 'gui' extra)")

from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

import converter_gui  # noqa: E402
import converter_gui.app  # noqa: E402
from converter_gui.main_window import (  # noqa: E402
    INITIAL_SIZE,
    MINIMUM_SIZE,
    PLACEHOLDER_OBJECT_NAME,
    MainWindow,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"


def test_starts(qtbot) -> None:
    """The window can be created, registered, shown and is really visible."""
    window = MainWindow()
    qtbot.addWidget(window)

    window.show()

    assert window.isVisible()
    assert window.windowTitle() == "PyImageConverter"
    assert window.isWindow()
    assert (window.width(), window.height()) == INITIAL_SIZE
    assert (window.minimumWidth(), window.minimumHeight()) == MINIMUM_SIZE


def test_window_shows_a_placeholder_page(qtbot) -> None:
    """The central widget is a neutral "under development" page, not an empty shell."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    page = window.centralWidget()
    assert page is not None
    label = page.findChild(QLabel, PLACEHOLDER_OBJECT_NAME)
    assert label is not None
    assert "under development" in label.text()
    assert label.isVisible()


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
