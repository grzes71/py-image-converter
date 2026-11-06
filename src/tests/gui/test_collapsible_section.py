"""Tests for the collapsible section widget.

The section is the only new behaviour of the metrics ergonomics change, so it is tested on its own: the
default state, the arrow, a real header click, the signal it emits and the fact that a collapsed body
stops contributing to the layout.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6", reason="PySide6 is not installed (install the 'gui' extra)")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QSignalSpy  # noqa: E402
from PySide6.QtWidgets import QLabel  # noqa: E402

from converter_gui.widgets.collapsible_section import (  # noqa: E402
    HEADER_OBJECT_NAME,
    CollapsibleSection,
)

TITLE = "Details"


def _section(qtbot, *, expanded: bool = False) -> CollapsibleSection:
    section = CollapsibleSection(TITLE, expanded=expanded)
    qtbot.addWidget(section)
    return section


def _with_content(qtbot) -> tuple[CollapsibleSection, QLabel]:
    section = _section(qtbot)
    label = QLabel("a value")
    section.body_layout.addWidget(label)
    return section, label


# --- default state ------------------------------------------------------


def test_it_starts_collapsed_by_default(qtbot) -> None:
    section = _section(qtbot)

    assert section.is_expanded is False
    assert section.header.isChecked() is False
    assert section.body.isHidden() is True
    assert section.header.arrowType() == Qt.ArrowType.RightArrow
    assert section.header.text() == TITLE
    assert section.header.objectName() == HEADER_OBJECT_NAME
    assert section.title == TITLE


def test_it_can_start_expanded(qtbot) -> None:
    section = _section(qtbot, expanded=True)

    assert section.is_expanded is True
    assert section.body.isHidden() is False
    assert section.header.arrowType() == Qt.ArrowType.DownArrow


def test_a_collapsed_section_needs_only_the_height_of_its_header(qtbot) -> None:
    section, _label = _with_content(qtbot)

    assert section.sizeHint().height() == section.header.sizeHint().height()


# --- toggling -----------------------------------------------------------


def test_set_expanded_shows_the_body_and_points_the_arrow_down(qtbot) -> None:
    section, label = _with_content(qtbot)
    collapsed_height = section.sizeHint().height()

    section.set_expanded(True)

    assert section.is_expanded is True
    assert section.body.isHidden() is False
    assert label.isVisibleTo(section) is True
    assert section.header.arrowType() == Qt.ArrowType.DownArrow
    assert section.sizeHint().height() > collapsed_height


def test_collapsing_again_hides_the_body_and_frees_the_space(qtbot) -> None:
    section, label = _with_content(qtbot)
    expanded_height = (section.set_expanded(True), section.sizeHint().height())[1]

    section.set_expanded(False)

    assert section.is_expanded is False
    assert section.body.isHidden() is True
    assert label.isVisibleTo(section) is False
    assert section.sizeHint().height() == section.header.sizeHint().height()
    assert section.sizeHint().height() < expanded_height


def test_clicking_the_header_toggles_the_section(qtbot) -> None:
    section, label = _with_content(qtbot)
    section.show()
    qtbot.waitExposed(section)

    qtbot.mouseClick(section.header, Qt.MouseButton.LeftButton)
    assert section.is_expanded is True
    assert label.isVisibleTo(section) is True

    qtbot.mouseClick(section.header, Qt.MouseButton.LeftButton)
    assert section.is_expanded is False
    assert label.isVisibleTo(section) is False


def test_toggle_flips_the_state_like_a_click(qtbot) -> None:
    section = _section(qtbot)

    section.toggle()
    assert section.is_expanded is True

    section.toggle()
    assert section.is_expanded is False


def test_set_expanded_reports_each_real_change_once(qtbot) -> None:
    section = _section(qtbot)
    spy = QSignalSpy(section.toggled)

    section.set_expanded(True)
    section.set_expanded(True)  # already expanded: nothing to report
    section.set_expanded(False)

    assert spy.count() == 2
    assert spy.at(0) == [True]
    assert spy.at(1) == [False]
