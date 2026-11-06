"""Root pytest configuration.

``pytest-qt`` ships in the ``dev`` extra, but PySide6 ships only in ``gui``. Without a Qt binding the
plugin aborts the entire session ("pytest-qt requires either PySide6, PyQt5 or PyQt6 installed"), so
``addopts`` in ``pyproject.toml`` keeps it out of the autoloaded set and it is loaded here only when
PySide6 can actually be imported. The GUI tests then skip themselves through ``pytest.importorskip``,
and the core suite stays runnable without the ``gui`` extra.
"""

from __future__ import annotations

import importlib.util


def _pyside6_available() -> bool:
    """True when PySide6 can be located; a broken installation counts as "not available"."""
    try:
        return importlib.util.find_spec("PySide6") is not None
    except (ImportError, ValueError):
        return False


if _pyside6_available():
    pytest_plugins = ["pytestqt.plugin"]
