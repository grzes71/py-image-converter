"""Tests for the synchronised image preview widget (T2.3).

Everything runs offscreen: the widget is shown so Qt performs a real layout (the views need a viewport
for Fit and for scrolling), and gestures are synthesised rather than performed by hand.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6", reason="PySide6 is not installed (install the 'gui' extra)")

from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QGraphicsView  # noqa: E402

from converter_gui.widgets.image_preview import ZOOM_LEVELS, ImagePreview  # noqa: E402

CTRL = Qt.KeyboardModifier.ControlModifier


def _qimage(width: int, height: int, colour: tuple[int, int, int] = (10, 20, 30)) -> QImage:
    image = QImage(width, height, QImage.Format.Format_RGB888)
    image.fill(QColor(*colour))
    return image


def _preview(qtbot, width: int = 400, height: int = 300) -> ImagePreview:
    """A shown, laid-out preview of a known size (the views need a real viewport)."""
    preview = ImagePreview()
    qtbot.addWidget(preview)
    preview.resize(width, height)
    preview.show()
    preview.layout().activate()
    QApplication.processEvents()
    return preview


def _wheel(
    view: QGraphicsView,
    delta: int,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> None:
    """Send a synthesised wheel event to the view's viewport (that is where Qt delivers it)."""
    centre = view.viewport().rect().center()
    event = QWheelEvent(
        QPointF(centre),
        QPointF(view.viewport().mapToGlobal(centre)),
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(view.viewport(), event)
    QApplication.processEvents()


# --- construction and content ------------------------------------------


def test_starts_with_one_empty_view(qtbot) -> None:
    preview = _preview(qtbot)

    assert len(preview.views) == 1
    view = preview.views[0]
    assert isinstance(view, QGraphicsView)
    assert view.scene() is not None
    assert view.pixmap_item.pixmap().isNull()
    assert preview.zoom == 1.0
    assert preview.is_fit is False


def test_setting_an_image_shows_it(qtbot) -> None:
    preview = _preview(qtbot)

    preview.set_image(_qimage(40, 30))

    view = preview.views[0]
    assert view.pixmap_item.pixmap().size() == QSize(40, 30)
    assert view.pixmap_item.isVisible()
    assert view.scene().sceneRect() == QRectF(0, 0, 40, 30)
    assert view.transform().m11() == pytest.approx(1.0)


def test_setting_a_pair_shows_two_views(qtbot) -> None:
    preview = _preview(qtbot)

    preview.set_image_pair(_qimage(40, 30), _qimage(20, 10, (200, 100, 50)))

    assert len(preview.views) == 2
    primary, secondary = preview.views
    assert primary.pixmap_item.pixmap().size() == QSize(40, 30)
    assert secondary.pixmap_item.pixmap().size() == QSize(20, 10)
    assert not secondary.isHidden()


def test_a_single_image_after_a_pair_returns_to_one_view(qtbot) -> None:
    preview = _preview(qtbot)
    preview.set_image_pair(_qimage(40, 30), _qimage(20, 10))

    preview.set_image(_qimage(12, 8))

    assert len(preview.views) == 1
    assert preview.views[0].pixmap_item.pixmap().size() == QSize(12, 8)


def test_clear_removes_the_images_and_the_second_view(qtbot) -> None:
    preview = _preview(qtbot)
    preview.set_image_pair(_qimage(40, 30), _qimage(20, 10))
    preview.set_zoom(4.0)

    preview.clear()

    assert len(preview.views) == 1
    view = preview.views[0]
    assert view.pixmap_item.pixmap().isNull()
    assert view.scene().sceneRect().isEmpty()
    assert preview.zoom == 1.0
    assert view.transform().m11() == pytest.approx(1.0)


# --- zoom ---------------------------------------------------------------


def test_zoom_levels_are_exposed() -> None:
    assert ZOOM_LEVELS == (1.0, 2.0, 4.0, 8.0)


@pytest.mark.parametrize("level", ZOOM_LEVELS)
def test_zoom_levels_set_the_view_transform(qtbot, level: float) -> None:
    preview = _preview(qtbot)
    preview.set_image(_qimage(40, 30))

    preview.set_zoom(level)

    view = preview.views[0]
    assert preview.zoom == pytest.approx(level)
    assert view.transform().m11() == pytest.approx(level)
    assert view.transform().m22() == pytest.approx(level)
    assert view.pixmap_item.pixmap().size() == QSize(40, 30), "the pixmap must not be resampled"


def test_zoom_emits_one_signal_per_change(qtbot) -> None:
    preview = _preview(qtbot)
    preview.set_image(_qimage(40, 30))
    received: list[float] = []
    preview.zoomChanged.connect(received.append)

    preview.set_zoom(4.0)

    assert received == [4.0]


def test_setting_a_new_image_resets_the_zoom(qtbot) -> None:
    preview = _preview(qtbot)
    preview.set_image(_qimage(40, 30))
    preview.set_zoom(8.0)

    preview.set_image(_qimage(40, 30))

    assert preview.zoom == 1.0


def test_ctrl_wheel_walks_through_the_zoom_levels(qtbot) -> None:
    preview = _preview(qtbot)
    preview.set_image(_qimage(40, 30))
    view = preview.views[0]

    _wheel(view, 120, CTRL)
    assert preview.zoom == pytest.approx(2.0)

    _wheel(view, 120, CTRL)
    assert preview.zoom == pytest.approx(4.0)

    _wheel(view, 120, CTRL)
    assert preview.zoom == pytest.approx(8.0)

    _wheel(view, 120, CTRL)
    assert preview.zoom == pytest.approx(8.0), "the last level is the upper bound"

    _wheel(view, -120, CTRL)
    assert preview.zoom == pytest.approx(4.0)

    _wheel(view, -120, CTRL)
    _wheel(view, -120, CTRL)
    _wheel(view, -120, CTRL)
    assert preview.zoom == pytest.approx(1.0), "1x is the lower bound"


def test_a_plain_wheel_does_not_zoom(qtbot) -> None:
    preview = _preview(qtbot)
    preview.set_image(_qimage(40, 30))

    _wheel(preview.views[0], 120)

    assert preview.zoom == 1.0


def test_invalid_zoom_values_are_rejected(qtbot) -> None:
    preview = _preview(qtbot)
    preview.set_image(_qimage(40, 30))

    for bad in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            preview.set_zoom(bad)
    assert preview.zoom == 1.0


# --- Fit ----------------------------------------------------------------


def test_fit_shows_the_whole_image(qtbot) -> None:
    preview = _preview(qtbot, 200, 150)
    preview.set_image(_qimage(100, 100))

    preview.fit()

    viewport = preview.views[0].viewport().size()
    factor = preview.zoom
    assert factor > 0
    assert 100 * factor <= viewport.width() + 1
    assert 100 * factor <= viewport.height() + 1
    assert preview.views[0].transform().m11() == pytest.approx(factor)
    assert preview.is_fit


def test_fit_is_pixel_perfect_when_it_can_be(qtbot) -> None:
    preview = _preview(qtbot, 400, 300)
    preview.set_image(_qimage(10, 10))

    preview.fit()

    viewport = preview.views[0].viewport().size()
    factor = preview.zoom
    assert factor == int(factor), "an integer zoom factor is expected"
    assert factor >= 1.0
    assert 10 * int(factor) <= viewport.width()
    assert 10 * int(factor) <= viewport.height()
    bigger = 10 * (int(factor) + 1)
    assert bigger > viewport.width() or bigger > viewport.height(), "the factor is the largest"


def test_fit_shrinks_an_image_larger_than_the_view(qtbot) -> None:
    preview = _preview(qtbot, 150, 100)
    preview.set_image(_qimage(600, 400))

    preview.fit()

    viewport = preview.views[0].viewport().size()
    factor = preview.zoom
    assert 0 < factor < 1
    assert 600 * factor <= viewport.width() + 1
    assert 400 * factor <= viewport.height() + 1


def test_fit_is_kept_after_a_resize(qtbot) -> None:
    preview = _preview(qtbot, 200, 200)
    preview.set_image(_qimage(50, 50))
    preview.fit()
    first = preview.zoom

    preview.resize(400, 400)
    preview.layout().activate()
    QApplication.processEvents()

    assert preview.zoom > first
    assert preview.zoom == int(preview.zoom)
    assert preview.is_fit


def test_an_explicit_zoom_leaves_fit_mode(qtbot) -> None:
    preview = _preview(qtbot, 200, 200)
    preview.set_image(_qimage(50, 50))
    preview.fit()

    preview.set_zoom(2.0)

    assert preview.is_fit is False
    assert preview.zoom == pytest.approx(2.0)


# --- synchronisation ----------------------------------------------------


def test_zoom_is_synchronised_between_the_views(qtbot) -> None:
    preview = _preview(qtbot)
    preview.set_image_pair(_qimage(100, 100), _qimage(100, 100))

    preview.set_zoom(4.0)

    for view in preview.views:
        assert view.transform().m11() == pytest.approx(4.0)
        assert view.transform().m22() == pytest.approx(4.0)


def test_wheel_zoom_in_one_view_updates_the_other(qtbot) -> None:
    preview = _preview(qtbot)
    preview.set_image_pair(_qimage(100, 100), _qimage(100, 100))
    primary, secondary = preview.views

    _wheel(secondary, 120, CTRL)

    assert preview.zoom == pytest.approx(2.0)
    assert primary.transform().m11() == pytest.approx(2.0)
    assert secondary.transform().m11() == pytest.approx(2.0)


def test_panning_is_synchronised_in_both_directions(qtbot) -> None:
    preview = _preview(qtbot, 200, 200)
    preview.set_image_pair(_qimage(100, 100), _qimage(100, 100))
    preview.set_zoom(8.0)
    primary, secondary = preview.views

    primary.horizontalScrollBar().setValue(120)
    primary.verticalScrollBar().setValue(90)
    QApplication.processEvents()

    assert secondary.horizontalScrollBar().value() == primary.horizontalScrollBar().value()
    assert secondary.verticalScrollBar().value() == primary.verticalScrollBar().value()

    secondary.horizontalScrollBar().setValue(20)
    QApplication.processEvents()

    assert primary.horizontalScrollBar().value() == secondary.horizontalScrollBar().value()


def test_synchronisation_does_not_recurse(qtbot) -> None:
    preview = _preview(qtbot, 200, 200)
    preview.set_image_pair(_qimage(100, 100), _qimage(100, 100))
    preview.set_zoom(8.0)
    primary, secondary = preview.views
    emissions = {"primary": 0, "secondary": 0}
    primary.scrolled.connect(lambda: emissions.update(primary=emissions["primary"] + 1))
    secondary.scrolled.connect(lambda: emissions.update(secondary=emissions["secondary"] + 1))

    primary.horizontalScrollBar().setValue(100)
    QApplication.processEvents()

    assert emissions["primary"] == 1
    assert emissions["secondary"] == 1, "exactly one propagation, no ping-pong"
    assert secondary.horizontalScrollBar().value() == primary.horizontalScrollBar().value()


def test_a_pair_with_different_sizes_stays_consistent(qtbot) -> None:
    preview = _preview(qtbot, 300, 200)
    preview.set_image_pair(_qimage(120, 80), _qimage(60, 40, (200, 100, 50)))
    primary, secondary = preview.views

    preview.fit()

    viewport = primary.viewport().size()
    factor = preview.zoom
    assert 120 * factor <= viewport.width() + 1, "Fit uses the larger image"
    assert 80 * factor <= viewport.height() + 1
    assert primary.transform().m11() == pytest.approx(secondary.transform().m11())

    preview.set_zoom(8.0)
    secondary.horizontalScrollBar().setValue(50)
    QApplication.processEvents()

    assert primary.horizontalScrollBar().value() == secondary.horizontalScrollBar().value()


# --- nearest-neighbour rendering ----------------------------------------


def test_zoom_is_done_by_transform_without_smoothing(qtbot) -> None:
    preview = _preview(qtbot)
    preview.set_image(_qimage(20, 10))
    preview.set_zoom(8.0)
    view = preview.views[0]

    assert view.pixmap_item.transformationMode() == Qt.TransformationMode.FastTransformation
    assert not view.renderHints() & QPainter.RenderHint.SmoothPixmapTransform
    assert not view.renderHints() & QPainter.RenderHint.Antialiasing
    assert view.transform().m11() == pytest.approx(8.0)
    assert view.pixmap_item.pixmap().size() == QSize(20, 10), "the pixmap stays at 1x"

    # One image pixel covers an 8x8 block of the viewport, so nothing is interpolated.
    origin = view.mapFromScene(QPointF(0.0, 0.0))
    next_pixel_x = view.mapFromScene(QPointF(1.0, 0.0))
    next_pixel_y = view.mapFromScene(QPointF(0.0, 1.0))
    assert next_pixel_x.x() - origin.x() == 8
    assert next_pixel_y.y() - origin.y() == 8


# --- invalid input ------------------------------------------------------


def test_invalid_images_are_rejected(qtbot) -> None:
    preview = _preview(qtbot)

    with pytest.raises(TypeError):
        preview.set_image("not an image")
    with pytest.raises(ValueError):
        preview.set_image(QImage())
    with pytest.raises(TypeError):
        preview.set_image_pair(_qimage(10, 10), 42)
    with pytest.raises(ValueError):
        preview.set_image_pair(QImage(), _qimage(10, 10))
    assert len(preview.views) == 1
