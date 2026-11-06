"""Main window of the GUI skeleton.

T2.1 delivers the shell only: a real, visible window with a placeholder page. The preview, the
parameter panel and the export dialog arrive in later tasks (T2.3-T4.4), and nothing here talks to
the conversion engine yet.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

WINDOW_TITLE = "PyImageConverter"
PLACEHOLDER_OBJECT_NAME = "placeholderLabel"
PLACEHOLDER_TEXT = (
    "PyImageConverter is under development.\n"
    "The image preview and the conversion controls will appear here."
)
INITIAL_SIZE = (960, 640)
MINIMUM_SIZE = (640, 420)


class MainWindow(QMainWindow):
    """The application window: title, a reasonable size and a placeholder central widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(*INITIAL_SIZE)
        self.setMinimumSize(*MINIMUM_SIZE)
        self.setCentralWidget(self._build_placeholder())

    def _build_placeholder(self) -> QWidget:
        """Build the neutral placeholder page that later tasks replace with the real layout."""
        label = QLabel(PLACEHOLDER_TEXT)
        label.setObjectName(PLACEHOLDER_OBJECT_NAME)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(label)
        return page
