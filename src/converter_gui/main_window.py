"""Main window of the PySide6 GUI (T2.6).

The window wires the existing components together: :class:`ParameterPanel` for the configuration,
:class:`ImagePreview` for the synchronised source/result views and :class:`MetricsPanel` for the numbers.
A single :class:`~converter.engine.ConversionEngine` owns the conversion, and the window keeps the
two-piece state from plan §7.5 - ``latest_generation`` (the newest requested configuration or source) and
``displayed`` (the last *successful* conversion with the configuration that produced it). Staleness is
therefore a generation comparison, never a comparison of configurations.

Conversion is synchronous for now (Phase 3 moves it to a worker); a wait cursor and a re-entrance guard
keep the UI honest while it runs, and a failure never replaces the last good result.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import converter
from converter.engine import ConversionEngine
from converter.image_loader import SUPPORTED_EXTENSIONS
from converter.types import ConversionConfig, ConversionResult
from converter_gui.imaging import ndarray_to_qimage
from converter_gui.widgets.image_preview import ZOOM_LEVELS, ImagePreview
from converter_gui.widgets.metrics_panel import MetricsPanel
from converter_gui.widgets.parameter_panel import ParameterPanel

LOGGER = logging.getLogger(__name__)

WINDOW_TITLE = "PyImageConverter"
INITIAL_SIZE = (1080, 720)
MINIMUM_SIZE = (760, 520)

SOURCE_CAPTION = "Source (resized)"
RESULT_CAPTION = "Atari result"
NO_SOURCE_TEXT = "No image loaded - use File > Open or drop an image here"

BADGE_OBJECT_NAME = "stateBadge"
BADGE_STYLES = {
    "ready": "background:#1f6f3f; color:#ffffff; padding:2px 8px; border-radius:3px;",
    "stale": "background:#8a5a00; color:#ffffff; padding:2px 8px; border-radius:3px;",
    "updating": "background:#1f4e79; color:#ffffff; padding:2px 8px; border-radius:3px;",
    "error": "background:#8a1f1f; color:#ffffff; padding:2px 8px; border-radius:3px;",
    "empty": "background:#4a4a4a; color:#ffffff; padding:2px 8px; border-radius:3px;",
}

OPEN_FILTER = "Images ({})".format(
    " ".join(f"*{suffix}" for suffix in sorted(SUPPORTED_EXTENSIONS))
)


@dataclass(frozen=True)
class DisplayedConversion:
    """What the views currently show: the result and the configuration that produced it (plan §7.5)."""

    generation: int
    config: ConversionConfig
    result: ConversionResult
    timings: dict[str, float]
    produced_at: float
    elapsed: float


class MainWindow(QMainWindow):
    """The application window: menus, toolbar, synchronised previews, parameters and metrics."""

    def __init__(
        self, parent: QWidget | None = None, engine: ConversionEngine | None = None
    ) -> None:
        super().__init__(parent)
        self.engine = engine if engine is not None else ConversionEngine()
        self._latest_generation = 0
        self._displayed: DisplayedConversion | None = None
        self._source_path: Path | None = None
        self._last_error: str | None = None
        self._converting = False

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(*INITIAL_SIZE)
        self.setMinimumSize(*MINIMUM_SIZE)
        self.setAcceptDrops(True)

        self._build_layout()
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_status_bar()

        self.parameter_panel.config_changed.connect(self._on_config_changed)
        self._requested_config = self.parameter_panel.config()
        self._update_state()

    # --- state exposed to tests and later tasks -------------------------

    @property
    def latest_generation(self) -> int:
        """Generation of the newest source/configuration change."""
        return self._latest_generation

    @property
    def displayed(self) -> DisplayedConversion | None:
        """The last successful conversion (``None`` before the first one)."""
        return self._displayed

    @property
    def requested_config(self) -> ConversionConfig:
        """The configuration the panel currently represents."""
        return self._requested_config

    @property
    def source_path(self) -> Path | None:
        """The image currently loaded into the engine."""
        return self._source_path

    @property
    def last_error(self) -> str | None:
        """The message of the most recent failed conversion, if any."""
        return self._last_error

    @property
    def is_stale(self) -> bool:
        """True when the displayed result does not belong to the newest generation."""
        return self._displayed is None or self._displayed.generation != self._latest_generation

    # --- actions, menus and toolbar -------------------------------------

    def _build_actions(self) -> None:
        self.open_action = QAction("&Open…", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.setStatusTip("Open an image (PNG, BMP, GIF or JPG)")
        self.open_action.triggered.connect(self.open_dialog)

        self.exit_action = QAction("E&xit", self)
        self.exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.exit_action.setStatusTip("Close the application")
        self.exit_action.triggered.connect(self.close)

        self.convert_action = QAction("&Convert", self)
        self.convert_action.setShortcut(QKeySequence("Ctrl+Return"))
        self.convert_action.setStatusTip("Convert the image with the current parameters")
        self.convert_action.setEnabled(False)
        self.convert_action.triggered.connect(self.convert)

        self.about_action = QAction("&About PyImageConverter", self)
        self.about_action.setStatusTip("Show the version and a short description")
        self.about_action.triggered.connect(self.show_about)

        self.zoom_actions: dict[float, QAction] = {}
        for level in ZOOM_LEVELS:
            action = QAction(f"{level:g}×", self)
            action.setStatusTip(f"Zoom both previews to {level:g}×")
            action.setEnabled(False)
            action.triggered.connect(
                lambda _checked=False, value=level: self.preview.set_zoom(value)
            )
            self.zoom_actions[level] = action
        self.fit_action = QAction("Fit", self)
        self.fit_action.setStatusTip("Fit both previews into the window")
        self.fit_action.setEnabled(False)
        self.fit_action.triggered.connect(self.preview.fit)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(self.convert_action)

        view_menu = self.menuBar().addMenu("&View")
        for action in self.zoom_actions.values():
            view_menu.addAction(action)
        view_menu.addAction(self.fit_action)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self.about_action)

    def _build_toolbar(self) -> None:
        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("mainToolBar")
        toolbar.setMovable(False)
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.convert_action)
        toolbar.addSeparator()
        for action in self.zoom_actions.values():
            toolbar.addAction(action)
        toolbar.addAction(self.fit_action)

    # --- layout ---------------------------------------------------------

    def _build_layout(self) -> None:
        self.preview = ImagePreview(self)

        header = QWidget(self)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        self.file_label = QLabel(NO_SOURCE_TEXT, header)
        self.file_label.setObjectName("fileLabel")
        self.state_badge = QLabel(header)
        self.state_badge.setObjectName(BADGE_OBJECT_NAME)
        header_layout.addWidget(self.file_label, 1)
        header_layout.addWidget(self.state_badge, 0, Qt.AlignmentFlag.AlignRight)

        captions = QWidget(self)
        caption_layout = QHBoxLayout(captions)
        caption_layout.setContentsMargins(0, 0, 0, 0)
        for text in (SOURCE_CAPTION, RESULT_CAPTION):
            caption = QLabel(text, captions)
            caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
            caption_layout.addWidget(caption)

        preview_column = QWidget(self)
        preview_layout = QVBoxLayout(preview_column)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(header)
        preview_layout.addWidget(captions)
        preview_layout.addWidget(self.preview, 1)

        self.parameter_panel = ParameterPanel(self)
        self.metrics_panel = MetricsPanel(self)
        parameters_scroll = QScrollArea(self)
        parameters_scroll.setWidgetResizable(True)
        parameters_scroll.setFrameShape(QFrame.Shape.NoFrame)
        parameters_scroll.setWidget(self.parameter_panel)

        side_splitter = QSplitter(Qt.Orientation.Vertical, self)
        side_splitter.addWidget(parameters_scroll)
        side_splitter.addWidget(self.metrics_panel)
        side_splitter.setStretchFactor(0, 3)
        side_splitter.setStretchFactor(1, 2)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(preview_column)
        splitter.addWidget(side_splitter)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([700, 380])
        self.setCentralWidget(splitter)

    def _build_status_bar(self) -> None:
        self.statusBar().showMessage(NO_SOURCE_TEXT)

    # --- opening images -------------------------------------------------

    def open_dialog(self) -> None:
        """Ask for a path and load it; cancelling the dialog changes nothing."""
        start = str(self._source_path.parent) if self._source_path is not None else ""
        path, _ = QFileDialog.getOpenFileName(self, "Open image", start, OPEN_FILTER)
        if not path:
            return
        self.open_image(path)

    def open_image(self, path: str | Path) -> bool:
        """Load ``path`` into the engine; returns ``True`` when the source changed.

        The previous preview is deliberately kept (it simply becomes stale), so a failed or cancelled
        open never leaves an empty window.
        """
        candidate = Path(path)
        if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
            self._report_error(
                "Unsupported file",
                f"{candidate.name} is not a supported image.",
                details=f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )
            return False
        try:
            self.engine.set_source(candidate)
        except Exception as exc:
            LOGGER.exception("Could not load %s", candidate)
            self._report_error(
                "Could not open image",
                f"{candidate.name} could not be loaded: {exc}",
                details=repr(exc),
            )
            return False

        self._source_path = candidate
        self._latest_generation += 1
        self._last_error = None
        self.convert_action.setEnabled(True)
        for action in self.zoom_actions.values():
            action.setEnabled(True)
        self.fit_action.setEnabled(True)
        self._update_state()
        return True

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._first_supported_path(event) is not None:
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        path = self._first_supported_path(event)
        if path is None:
            event.ignore()
            return
        event.acceptProposedAction()
        self.open_image(path)

    @staticmethod
    def _first_supported_path(event: QDragEnterEvent | QDropEvent) -> Path | None:
        mime = event.mimeData()
        if mime is None or not mime.hasUrls():
            return None
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            candidate = Path(url.toLocalFile())
            if candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                return candidate
        return None

    # --- configuration and conversion -----------------------------------

    def _on_config_changed(self, config: ConversionConfig) -> None:
        """A parameter edit makes the shown result stale; it never converts on its own (yet)."""
        self._requested_config = dataclasses.replace(config)
        self._latest_generation += 1
        self._update_state()

    def convert(self) -> bool:
        """Convert synchronously with the current parameters.

        Returns ``True`` when a new result was displayed. On failure the previous result, preview and
        metrics stay exactly as they were and the error is reported.
        """
        if self._source_path is None or self._converting:
            return False

        self._converting = True
        self.convert_action.setEnabled(False)
        self._update_state()  # "updating…" before the blocking call
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        started = time.perf_counter()
        result: ConversionResult | None = None
        failure: Exception | None = None
        try:
            result = self.engine.convert(self._requested_config, compute_metrics=True)
        except Exception as exc:
            LOGGER.exception("Conversion failed")
            failure = exc
        finally:
            QApplication.restoreOverrideCursor()
            self._converting = False
            self.convert_action.setEnabled(self._source_path is not None)

        if failure is not None:
            self._last_error = str(failure) or type(failure).__name__
            self._update_state()
            self._report_error(
                "Conversion failed",
                f"The image could not be converted: {self._last_error}",
                details=repr(failure),
            )
            return False

        assert result is not None, "no failure and no result: the engine returned nothing"
        self._show_result(result, time.perf_counter() - started)
        return True

    def _show_result(self, result: ConversionResult, elapsed: float) -> None:
        """Display a fresh result: both previews, the metrics and the palette swatch."""
        source_image = ndarray_to_qimage(result.source_resized)
        converted_image = ndarray_to_qimage(result.converted_rgb)
        self.preview.set_image_pair(source_image, converted_image)
        self.metrics_panel.set_result(result)
        self.parameter_panel.set_palette_colors(
            [tuple(int(channel) for channel in colour) for colour in result.palette_rgb]
        )
        self._displayed = DisplayedConversion(
            generation=self._latest_generation,
            config=dataclasses.replace(self._requested_config),
            result=result,
            timings=dict(result.timings),
            produced_at=time.perf_counter(),
            elapsed=elapsed,
        )
        self._last_error = None
        self._update_state()

    # --- state rendering ------------------------------------------------

    def _update_state(self) -> None:
        state, badge, status = self._state_snapshot()
        self.state_badge.setText(badge)
        self.state_badge.setStyleSheet(BADGE_STYLES[state])
        self.state_badge.setToolTip(status)
        self.file_label.setText(self._file_text())
        self.statusBar().showMessage(status)

    def _state_snapshot(self) -> tuple[str, str, str]:
        """Return ``(style_key, badge_text, status_text)`` for the current state (§7.5)."""
        if self._source_path is None:
            return "empty", "no image", NO_SOURCE_TEXT
        name = self._source_path.name
        if self._converting:
            text = f"{name} - converting… (gen {self._latest_generation})"
            return "updating", f"updating… (gen {self._latest_generation})", text
        displayed = self._displayed
        if self._last_error is not None:
            if displayed is None:
                text = f"{name} - error: {self._last_error} (no previous result)"
            else:
                text = (
                    f"{name} - error: {self._last_error} - still showing gen {displayed.generation}"
                    " (outdated)"
                )
            return "error", f"error - showing gen {displayed.generation}", text
        if displayed is None or displayed.generation != self._latest_generation:
            shown = "nothing yet" if displayed is None else f"gen {displayed.generation}"
            text = (
                f"{name} - stale, showing {shown}; press Convert for gen {self._latest_generation}"
            )
            return "stale", f"stale - showing {shown}", text
        width, height = displayed.result.indexed.shape[1], displayed.result.indexed.shape[0]
        text = (
            f"{name} - {width}×{height} - ready (gen {displayed.generation})"
            f" - {displayed.elapsed * 1000:.0f} ms"
        )
        return "ready", f"ready (gen {displayed.generation})", text

    def _file_text(self) -> str:
        if self._source_path is None:
            return NO_SOURCE_TEXT
        source = self.engine.source
        if source is None:
            return self._source_path.name
        height, width = source.shape[0], source.shape[1]
        return f"{self._source_path.name} - source {width}×{height}"

    # --- dialogs --------------------------------------------------------

    def _report_error(self, title: str, message: str, *, details: str | None = None) -> None:
        """Show a modal error box (tests replace this method to stay headless)."""
        box = QMessageBox(
            QMessageBox.Icon.Critical,
            title,
            message,
            QMessageBox.StandardButton.Ok,
            self,
        )
        if details:
            box.setDetailedText(details)
        box.exec()

    def show_about(self) -> None:
        """About box with the real package version."""
        QMessageBox.about(
            self,
            f"About {WINDOW_TITLE}",
            f"<h3>{WINDOW_TITLE} {converter.__version__}</h3>"
            "<p>Converts images to Atari 8-bit ANTIC graphics.</p>"
            "<p>Open a PNG, BMP, GIF or JPG image, adjust the parameters and press Convert.</p>",
        )
