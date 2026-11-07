"""Zoomable, nearest-neighbour image preview (T2.3).

The widget shows either one image or a pair of images side by side. Both views share a single zoom
factor and a single scroll position: the preview owns that state, the views only report user gestures,
so a change made in one view is applied to the other exactly once (a re-entrancy guard stops the
resulting scroll/transform events from bouncing back).

Replacing the images keeps that state: a new conversion produces new pixels for the image the user is
already looking at, so ``set_image``/``set_image_pair`` keep the zoom and the scroll position (and Fit
stays Fit, following the new size). The view state changes only when the caller asks for it -
``set_zoom``, ``fit``, or ``clear`` back to an empty preview at 1x.

Scaling is done with the view transform, never by resampling the pixmap: the ``QGraphicsPixmapItem``
uses ``Qt.FastTransformation`` and the smoothing render hint is switched off, so at 8x one image pixel
is drawn as a crisp 8x8 block (plan §6.1).

The pair view can also show the second image with a non-square pixel: ``set_image_pair`` takes a
``secondary_aspect`` factor that multiplies the horizontal scale of that view only. That is a display
setting, not a data change - the Atari result keeps its exact pixels while looking the way it looks on
a PAL monitor. Zoom, Fit and panning keep working as before on both views; when the two views are
scaled differently, scrolling is synchronised on the scene position, so the same part of the image
stays in view.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap, QTransform, QWheelEvent
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QWidget,
)

#: The zoom steps offered by the widget; Fit may produce any value in between.
ZOOM_LEVELS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)

_MIN_ZOOM = 0.01
_EPSILON = 1e-9


def _as_pixmap(image: QImage | QPixmap) -> QPixmap:
    """Return a pixmap for ``image`` without resampling it."""
    if isinstance(image, QPixmap):
        return image
    if isinstance(image, QImage):
        if image.isNull():
            raise ValueError("cannot display a null QImage")
        return QPixmap.fromImage(image)
    raise TypeError(f"expected a QImage or QPixmap, got {type(image).__name__}")


def _next_zoom(current: float, step: int) -> float:
    """Next level from :data:`ZOOM_LEVELS`; a fractional Fit zoom snaps to the adjacent level."""
    if step > 0:
        for level in ZOOM_LEVELS:
            if level > current + _EPSILON:
                return level
        return ZOOM_LEVELS[-1]
    for level in reversed(ZOOM_LEVELS):
        if level < current - _EPSILON:
            return level
    return ZOOM_LEVELS[0]


def _as_aspect(factor: float) -> float:
    """Validate a horizontal display factor: finite and positive, ``1.0`` showing square pixels."""
    try:
        value = float(factor)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"expected a number, got {factor!r}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"pixel aspect must be a positive number, got {factor!r}")
    return value


class _PreviewView(QGraphicsView):
    """One view of the preview: nearest-neighbour pixmap, no scrollbars, reports gestures."""

    zoom_step_requested = Signal(int)
    scrolled = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        scene = QGraphicsScene(self)
        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.TransformationMode.FastTransformation)
        scene.addItem(self._item)
        self.setScene(scene)

        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)

    @property
    def pixmap_item(self) -> QGraphicsPixmapItem:
        """The item that draws the image, so tests and later widgets can inspect it."""
        return self._item

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._item.setPixmap(pixmap)
        self.scene().setSceneRect(QRectF(pixmap.rect()))

    def clear_pixmap(self) -> None:
        self._item.setPixmap(QPixmap())
        self.scene().setSceneRect(QRectF())

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta:
                self.zoom_step_requested.emit(1 if delta > 0 else -1)
            event.accept()
            return
        super().wheelEvent(event)

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        if dx or dy:
            self.scrolled.emit()


class ImagePreview(QWidget):
    """Single image or a synchronised pair, with 1x/2x/4x/8x and Fit zoom.

    The widget owns the zoom factor and the scroll position. Views only report gestures, and every
    change is applied to all views from here under a re-entrancy guard, so zooming or panning one view
    moves the other exactly once and cannot loop.
    """

    zoomChanged = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._zoom = 1.0
        self._fit_mode = False
        self._synchronising = False
        self._secondary: _PreviewView | None = None
        self._secondary_aspect = 1.0

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        self._primary = self._create_view()
        self._views: list[_PreviewView] = [self._primary]

    # --- public API -----------------------------------------------------

    @property
    def views(self) -> tuple[_PreviewView, ...]:
        """The views currently displayed: one for a single image, two for a pair."""
        return tuple(self._views)

    @property
    def zoom(self) -> float:
        """The current zoom factor (Fit may return a fractional value below 1)."""
        return self._zoom

    @property
    def is_fit(self) -> bool:
        """True while the zoom is kept equal to the Fit factor."""
        return self._fit_mode

    def set_image(self, image: QImage | QPixmap) -> None:
        """Show a single image in one view, keeping the current zoom and scroll position."""
        pixmap = _as_pixmap(image)
        self._show_single()
        self._primary.set_pixmap(pixmap)
        self._restore_view()

    def set_image_pair(
        self,
        primary: QImage | QPixmap,
        secondary: QImage | QPixmap,
        *,
        secondary_aspect: float = 1.0,
    ) -> None:
        """Show two images side by side (the sizes may differ), keeping the current view state.

        A conversion produces new pixels for the image the user is already looking at, so replacing them
        must not move the view: the zoom factor and the scroll position are kept, and Fit stays Fit (it
        simply follows the size of the new image).

        ``secondary_aspect`` stretches the second view horizontally only, which is how the non-square
        pixel of an Atari mode is displayed. It changes nothing about the image data and does not touch
        the zoom level; the primary view always shows square pixels.
        """
        primary_pixmap = _as_pixmap(primary)
        secondary_pixmap = _as_pixmap(secondary)
        self._secondary_aspect = _as_aspect(secondary_aspect)
        self._show_pair()
        self._primary.set_pixmap(primary_pixmap)
        secondary_view = self._secondary
        assert secondary_view is not None
        secondary_view.set_pixmap(secondary_pixmap)
        self._restore_view()

    def clear(self) -> None:
        """Remove every image and go back to a single empty view at 1x."""
        for view in self._views:
            view.clear_pixmap()
        self._show_single()
        self._fit_mode = False
        self._secondary_aspect = 1.0
        self._apply_zoom(1.0)

    def set_zoom(self, factor: float) -> None:
        """Apply an explicit zoom factor to all views (1x, 2x, 4x, 8x, ...)."""
        factor = float(factor)
        if not math.isfinite(factor) or factor <= 0.0:
            raise ValueError(f"zoom factor must be a positive number, got {factor!r}")
        self._fit_mode = False
        self._apply_zoom(max(factor, _MIN_ZOOM))

    def fit(self) -> None:
        """Fit the largest loaded image into the viewport, integer when possible."""
        self._fit_mode = True
        self._apply_zoom(self._fit_zoom())

    # --- mode switching -------------------------------------------------

    def _create_view(self) -> _PreviewView:
        view = _PreviewView(self)
        view.zoom_step_requested.connect(lambda step, source=view: self._on_zoom_step(source, step))
        view.scrolled.connect(lambda source=view: self._on_scrolled(source))
        self._layout.addWidget(view)
        return view

    def _show_single(self) -> None:
        if self._secondary is not None:
            self._layout.removeWidget(self._secondary)
            self._secondary.hide()
        self._views = [self._primary]

    def _show_pair(self) -> None:
        if self._secondary is None:
            self._secondary = self._create_view()
        if self._secondary not in self._views:
            self._views.append(self._secondary)
        self._layout.addWidget(self._secondary)
        self._secondary.show()

    # --- zoom and pan ---------------------------------------------------

    def _aspect_of(self, view: _PreviewView) -> float:
        """Horizontal display factor of one view: the pixel aspect for the pair's second image."""
        if self._secondary is not None and view is self._secondary:
            return self._secondary_aspect
        return 1.0

    def _restore_view(self) -> None:
        """Put the view state back on new pixels: Fit follows the image, a chosen zoom simply stays."""
        self._apply_zoom(self._fit_zoom() if self._fit_mode else self._zoom)

    def _apply_zoom(self, factor: float, source: _PreviewView | None = None) -> None:
        factor = max(float(factor), _MIN_ZOOM)
        self._synchronising = True
        try:
            for view in self._views:
                aspect = self._aspect_of(view)
                view.setTransform(QTransform.fromScale(factor * aspect, factor))
            self._copy_scroll(source if source is not None else self._views[0])
        finally:
            self._synchronising = False
        self._zoom = factor
        self.zoomChanged.emit(factor)

    def _copy_scroll(self, source: _PreviewView) -> None:
        """Show the same scene point at the top-left corner of every other view.

        Copying the raw scrollbar values would drift apart as soon as two views use different
        scaling (the horizontal scale differs by the pixel aspect), so the offset is converted
        through the scene. With equal factors the two are identical. Qt clamps the values, so a
        smaller image simply stays anchored at its edge.
        """
        if len(self._views) < 2:
            return
        anchor = source.mapToScene(0, 0)
        for view in self._views:
            if view is source:
                continue
            offset = view.mapFromScene(QPointF(anchor))
            if offset.x():
                bar = view.horizontalScrollBar()
                bar.setValue(bar.value() + offset.x())
            if offset.y():
                bar = view.verticalScrollBar()
                bar.setValue(bar.value() + offset.y())

    def _on_scrolled(self, source: _PreviewView) -> None:
        if self._synchronising:
            return
        self._synchronising = True
        try:
            self._copy_scroll(source)
        finally:
            self._synchronising = False

    def _on_zoom_step(self, source: _PreviewView, step: int) -> None:
        self._fit_mode = False
        self._apply_zoom(_next_zoom(self._zoom, step), source=source)

    def _fit_zoom(self) -> float:
        """Largest factor showing the whole image: integer when >= 1, fractional when smaller."""
        size = self._image_size()
        viewport = self._primary.viewport().size()
        if size.isEmpty() or viewport.width() <= 0 or viewport.height() <= 0:
            return 1.0
        factor = min(viewport.width() / size.width(), viewport.height() / size.height())
        if factor >= 1.0:
            return float(max(math.floor(factor), 1))
        return factor

    def _image_size(self) -> QSize:
        """Bounding size of the visible images *as displayed*, so a pair still fits.

        The second view is scaled by its pixel aspect, so its displayed width is wider than its
        pixel width; Fit has to use that, otherwise an ANTIC E result would overflow sideways.
        """
        sizes = []
        for view in self._views:
            if view.isHidden():
                continue
            size = view.pixmap_item.pixmap().size()
            if size.isEmpty():
                continue
            width = int(round(size.width() * self._aspect_of(view)))
            sizes.append(QSize(max(width, 1), size.height()))
        if not sizes:
            return QSize()
        return QSize(max(size.width() for size in sizes), max(size.height() for size in sizes))

    # --- events ---------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fit_mode and not self._image_size().isEmpty():
            self._apply_zoom(self._fit_zoom())
