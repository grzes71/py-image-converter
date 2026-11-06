"""Shared setup for the GUI tests.

``QT_QPA_PLATFORM=offscreen`` lets Qt build widgets without a display server, and it is set here so
it applies before any test imports Qt.

The PySide6 skip itself lives in the test modules (``pytest.importorskip`` at import time): the same
call inside ``conftest.py`` would abort collection with an *error* instead of skipping the directory,
which is why it is not done here.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _stop_the_workers():
    """Shut every ``MainWindow`` a test built, so no ``QThread`` outlives its test.

    The import is late and forgiving on purpose: this module has to stay importable - and usable by the
    Qt-free protocol tests - when PySide6 is missing.
    """
    yield
    try:
        from gui_fakes import CREATED
    except ImportError:  # no Qt: no window can exist
        return

    while CREATED:
        window = CREATED.pop()
        if window._shutdown_result is not True:
            window.shutdown(50)
