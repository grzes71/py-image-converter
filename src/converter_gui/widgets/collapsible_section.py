"""A section with a clickable header that shows or hides its content.

The diagnostics of the main window (metrics, stage timings) live in sections like this one: collapsed
they occupy the height of their header only, expanded they show exactly the widget tree they were given.
The header carries the title, an arrow that tells the state (right = collapsed, down = expanded) and
reacts to a click anywhere on the bar as well as to the keyboard (Space, the standard button behaviour).

The widget owns no state beyond "expanded": showing or hiding the body is the whole mechanism, so the
content keeps its own layout and its own behaviour and nothing has to be rebuilt when the section is
toggled - a collapsed section is still updated, it simply is not visible.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QSizePolicy, QToolButton, QVBoxLayout, QWidget

HEADER_OBJECT_NAME = "sectionHeader"


class CollapsibleSection(QWidget):
    """A titled section whose body can be collapsed to the height of its header."""

    toggled = Signal(bool)
    """Emitted with the new state after the user (or :meth:`set_expanded`) expanded or collapsed it."""

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        *,
        expanded: bool = False,
    ) -> None:
        super().__init__(parent)
        self._title = title

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.header = QToolButton(self)
        self.header.setObjectName(HEADER_OBJECT_NAME)
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setAutoRaise(True)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        font = QFont(self.header.font())
        font.setBold(True)
        self.header.setFont(font)
        self.header.setChecked(expanded)

        self.body = QWidget(self)
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(12, 4, 0, 4)
        self.body_layout.setSpacing(4)

        layout.addWidget(self.header)
        layout.addWidget(self.body)

        self._apply_state(expanded)
        self.header.toggled.connect(self._on_toggled)

    # --- public API -----------------------------------------------------

    @property
    def title(self) -> str:
        """The text shown in the header."""
        return self._title

    @property
    def is_expanded(self) -> bool:
        """True when the body is visible."""
        return self.header.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        """Expand or collapse the section (a no-op when it already is in that state)."""
        self.header.setChecked(bool(expanded))

    def toggle(self) -> None:
        """Flip the state, exactly like a click on the header."""
        self.set_expanded(not self.is_expanded)

    # --- internals ------------------------------------------------------

    def _on_toggled(self, expanded: bool) -> None:
        self._apply_state(expanded)
        self.toggled.emit(expanded)

    def _apply_state(self, expanded: bool) -> None:
        """Show or hide the body and point the arrow the right way."""
        self.body.setVisible(expanded)
        self.header.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.header.setToolTip(f"{'Hide' if expanded else 'Show'} {self._title.lower()}")
        self.header.setAccessibleName(self._title)
