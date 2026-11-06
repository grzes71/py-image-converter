"""Shared setup for the GUI tests.

``QT_QPA_PLATFORM=offscreen`` lets Qt build widgets without a display server, and it is set here so
it applies before any test imports Qt.

The PySide6 skip itself lives in the test modules (``pytest.importorskip`` at import time): the same
call inside ``conftest.py`` would abort collection with an *error* instead of skipping the directory,
which is why it is not done here.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
