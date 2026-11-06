"""Application start-up: QApplication, metadata, the main window and the event loop.

Every entry point (``python -m converter_gui``, ``convert_gui.py`` and the ``atari-convert-gui``
console script) delegates here, so the Qt initialisation exists exactly once.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtWidgets import QApplication

from converter_gui.main_window import MainWindow

APPLICATION_NAME = "PyImageConverter"
ORGANIZATION_NAME = "PyImageConverter"
ORGANIZATION_DOMAIN = "pyimageconverter.local"


def create_application(argv: Sequence[str] | None = None) -> QApplication:
    """Return the application object, creating it once and applying the metadata.

    An existing instance (created by a test harness, for example) is reused instead of creating a
    second ``QApplication``, which Qt does not allow.
    """
    application = QApplication.instance()
    if not isinstance(application, QApplication):
        application = QApplication(list(argv) if argv is not None else sys.argv)
    application.setApplicationName(APPLICATION_NAME)
    application.setApplicationDisplayName(APPLICATION_NAME)
    application.setOrganizationName(ORGANIZATION_NAME)
    application.setOrganizationDomain(ORGANIZATION_DOMAIN)
    return application


def main(argv: Sequence[str] | None = None) -> int:
    """Show the main window and return the exit code of the Qt event loop."""
    application = create_application(argv)
    window = MainWindow()
    window.show()
    return application.exec()
