"""Main window of the PySide6 GUI.

T2.6 built the window and T3.3 moved the conversion off the GUI thread: the window no longer owns an
engine and never converts synchronously. It creates a :class:`QThreadBackend` and a
:class:`ConversionController`, forwards user requests to the controller and renders the controller's
state - the controller alone owns ``requested_config``, ``displayed``, the generation counter,
``is_stale`` and ``is_updating`` (plan §7.1 and §7.5).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QDragEnterEvent,
    QDropEvent,
    QKeySequence,
    QShowEvent,
)
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import converter
from converter.aspect import preview_pixel_aspect
from converter.hardware_colors import HardwareFrame, HardwareFrameRenderer
from converter.image_loader import SUPPORTED_EXTENSIONS
from converter.types import ConversionConfig, ExportFormat
from converter_gui.controllers.conversion_controller import (
    DEBOUNCE_INTERVAL_MS,
    DEFAULT_SHUTDOWN_TIMEOUT_MS,
    ConversionController,
    DisplayedConversion,
)
from converter_gui.exporting import (
    EXPORT_DESCRIPTION,
    EXPORT_FORMATS,
    ExportPayload,
    availability,
    chosen_format,
    dialog_filters,
    export_displayed,
    path_with_suffix,
    suggested_path,
)
from converter_gui.imaging import ndarray_to_qimage
from converter_gui.widgets.image_preview import ZOOM_LEVELS, ImagePreview
from converter_gui.widgets.metrics_panel import MetricsPanel
from converter_gui.widgets.parameter_panel import ParameterPanel
from converter_gui.workers.base import ConversionBackend
from converter_gui.workers.conversion_worker import QThreadBackend

WINDOW_TITLE = "PyImageConverter"
INITIAL_SIZE = (1080, 720)
MINIMUM_SIZE = (760, 520)

SOURCE_CAPTION = "Source (resized)"
RESULT_CAPTION = "Atari result"
NO_SOURCE_TEXT = "No image loaded - use File > Open or drop an image here"
SHUTDOWN_TIMEOUT_MS = DEFAULT_SHUTDOWN_TIMEOUT_MS

METRICS_PANEL_HEIGHT = 150
"""Initial height of the diagnostics column: both sections collapsed plus the Copy button."""

BADGE_OBJECT_NAME = "stateBadge"
BADGE_STYLES = {
    "ready": "background:#1f6f3f; color:#ffffff; padding:2px 8px; border-radius:3px;",
    "stale": "background:#8a5a00; color:#ffffff; padding:2px 8px; border-radius:3px;",
    "updating": "background:#1f4e79; color:#ffffff; padding:2px 8px; border-radius:3px;",
    "loading": "background:#1f4e79; color:#ffffff; padding:2px 8px; border-radius:3px;",
    "error": "background:#8a1f1f; color:#ffffff; padding:2px 8px; border-radius:3px;",
    "empty": "background:#4a4a4a; color:#ffffff; padding:2px 8px; border-radius:3px;",
}

OPEN_FILTER = "Images ({})".format(
    " ".join(f"*{suffix}" for suffix in sorted(SUPPORTED_EXTENSIONS))
)

EXPORT_FILTER_SEPARATOR = ";;"


class MainWindow(QMainWindow):
    """The application window: menus, toolbar, previews, parameters, metrics and the async controller."""

    #: Emitted with an :class:`ExportPayload` when a current result may be exported.
    export_requested = Signal(object)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        backend: ConversionBackend | None = None,
        controller: ConversionController | None = None,
        debounce_ms: int = DEBOUNCE_INTERVAL_MS,
    ) -> None:
        super().__init__(parent)
        if controller is None:
            controller = ConversionController(
                backend if backend is not None else QThreadBackend(),
                debounce_ms=debounce_ms,
                auto_update=True,
            )
        self._controller = controller
        self._source_path: Path | None = None
        self._pending_source_path: Path | None = None
        self._export_directory: Path | None = None
        self._loading_source = False
        self._closing = False
        self._shutdown_result: bool | None = None
        self._banner_text = ""
        self._frame_renderer = HardwareFrameRenderer()
        self._frame: HardwareFrame | None = None

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(*INITIAL_SIZE)
        self.setMinimumSize(*MINIMUM_SIZE)
        self.setAcceptDrops(True)

        self._build_layout()
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_status_bar()
        self._connect_controller()
        self._refresh_state()

    # --- state exposed to tests and later tasks -------------------------

    @property
    def controller(self) -> ConversionController:
        """The controller that owns the conversion state."""
        return self._controller

    @property
    def backend(self) -> ConversionBackend:
        """The backend the controller drives (kept alive by the controller)."""
        return self._controller.backend

    @property
    def latest_generation(self) -> int:
        """Generation of the newest source or image-relevant configuration change."""
        return self._controller.latest_generation

    @property
    def displayed(self) -> DisplayedConversion | None:
        """The last successful conversion shown in the window."""
        return self._controller.displayed

    @property
    def requested_config(self) -> ConversionConfig:
        """The configuration the parameter panel currently represents."""
        return self._controller.requested_config

    @property
    def source_path(self) -> Path | None:
        """Path of the source that was last loaded successfully."""
        return self._source_path

    @property
    def last_error(self) -> str | None:
        """Message of the most recent failed load or conversion."""
        return self._controller.last_error

    @property
    def is_stale(self) -> bool:
        """True when the displayed result does not belong to the newest generation."""
        return self._controller.is_stale

    @property
    def is_updating(self) -> bool:
        """True while the controller is converting."""
        return self._controller.is_updating

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
        self.convert_action.setStatusTip("Convert now with the current parameters")
        self.convert_action.setEnabled(False)
        self.convert_action.triggered.connect(self.convert_now)

        self.export_action = QAction("&Export…", self)
        self.export_action.setShortcut(QKeySequence("Ctrl+E"))
        self.export_action.setStatusTip("Save the displayed result as PNG, BIN, ASM, DTA or XEX")
        self.export_action.setEnabled(False)
        self.export_action.triggered.connect(self.export_dialog)

        self.auto_update_action = QAction("&Auto-update", self)
        self.auto_update_action.setCheckable(True)
        self.auto_update_action.setChecked(self._controller.auto_update)
        self.auto_update_action.setStatusTip("Convert automatically after a parameter change")
        self.auto_update_action.toggled.connect(self._controller.set_auto_update)

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
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction(self.convert_action)
        edit_menu.addAction(self.auto_update_action)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.auto_update_action)
        view_menu.addSeparator()
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
        toolbar.addAction(self.export_action)
        toolbar.addAction(self.auto_update_action)
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
        preview_layout.addWidget(self._build_error_banner())
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
        # The parameters take every spare pixel; the diagnostics get exactly the height their collapsed
        # sections need and grow only when a section is expanded (`_fit_side_splitter`).
        side_splitter.setStretchFactor(0, 1)
        side_splitter.setStretchFactor(1, 0)
        side_splitter.setSizes([560, METRICS_PANEL_HEIGHT])
        self.side_splitter = side_splitter
        self.metrics_panel.metrics_section.toggled.connect(self._schedule_side_splitter_fit)
        self.metrics_panel.timings_section.toggled.connect(self._schedule_side_splitter_fit)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(preview_column)
        splitter.addWidget(side_splitter)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([700, 380])
        self.setCentralWidget(splitter)

    def _schedule_side_splitter_fit(self, *_state: object) -> None:
        """Re-fit the diagnostics column once Qt has recomputed the layout of the sections."""
        QTimer.singleShot(0, self._fit_side_splitter)

    def _fit_side_splitter(self, *_state: object) -> None:
        """Give the diagnostics the height its sections need and hand the rest to the parameters."""
        splitter = self.side_splitter
        panel = self.metrics_panel
        wanted = max(panel.sizeHint().height(), panel.minimumSizeHint().height())
        available = splitter.height()
        if available <= 0:
            return  # not laid out yet: the sizes set in _build_layout still apply
        first = max(available - wanted - splitter.handleWidth(), 0)
        splitter.setSizes([first, wanted])

    def showEvent(self, event: QShowEvent) -> None:
        """Fit the diagnostics column once the window has a real geometry."""
        super().showEvent(event)
        self._schedule_side_splitter_fit()

    def _build_error_banner(self) -> QWidget:
        """A non-modal, dismissible line above the preview (plan §7.6)."""
        self.error_banner = QFrame(self)
        self.error_banner.setObjectName("errorBanner")
        self.error_banner.setStyleSheet("background:#5a1f1f; color:#ffffff; border-radius:3px;")
        row = QHBoxLayout(self.error_banner)
        row.setContentsMargins(8, 4, 8, 4)
        self.error_label = QLabel(self.error_banner)
        self.error_label.setWordWrap(True)
        self.error_details_button = QPushButton("Details", self.error_banner)
        self.error_details_button.setToolTip("Show the full error message")
        self.error_details_button.clicked.connect(self.show_error_details)
        self.error_dismiss_button = QPushButton("Dismiss", self.error_banner)
        self.error_dismiss_button.clicked.connect(self.hide_error_banner)
        row.addWidget(self.error_label, 1)
        row.addWidget(self.error_details_button)
        row.addWidget(self.error_dismiss_button)
        self.error_banner.hide()
        return self.error_banner

    def _build_status_bar(self) -> None:
        self.statusBar().showMessage(NO_SOURCE_TEXT)

    def _connect_controller(self) -> None:
        """Wire the controller's signals; the parameter panel feeds the controller directly."""
        controller = self._controller
        controller.result_ready.connect(self._on_result_ready)
        controller.source_loaded.connect(self._on_source_loaded)
        controller.source_failed.connect(self._on_source_failed)
        controller.conversion_failed.connect(self._on_conversion_failed)
        controller.conversion_cancelled.connect(self._on_conversion_cancelled)
        controller.export_requested.connect(self._on_export_requested)
        controller.state_changed.connect(self._refresh_state)
        self.parameter_panel.config_changed.connect(controller.set_config)
        self.parameter_panel.config_changed.connect(self._on_config_changed)

    # --- user requests --------------------------------------------------

    def open_dialog(self) -> None:
        """Ask for a path and hand it to the controller; cancelling changes nothing."""
        start = str(self._source_path.parent) if self._source_path is not None else ""
        path, _ = QFileDialog.getOpenFileName(self, "Open image", start, OPEN_FILTER)
        if not path:
            return
        self.open_image(path)

    def open_image(self, path: str | Path) -> bool:
        """Ask the controller to load ``path``; returns ``False`` only for a refused request."""
        if self._closing:
            return False
        candidate = Path(path)
        if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
            self._report_error(
                "Unsupported file",
                f"{candidate.name} is not a supported image.",
                details=f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )
            return False
        self._pending_source_path = candidate
        self._loading_source = True
        self._controller.set_source(candidate)
        self._refresh_state()
        return True

    def convert_now(self) -> None:
        """Forward Convert to the controller; the GUI thread never converts anything."""
        if self._closing:
            return
        self._controller.convert_now()

    def request_export(self) -> bool:
        """Ask for an export; the controller decides whether the result is current enough."""
        if self._closing:
            return False
        return self._controller.request_export()

    def export_dialog(self) -> bool:
        """Save the displayed result: the controller gates it, the save dialog asks where.

        Returns ``True`` when files were written, ``False`` when the export was refused, cancelled or
        failed (the status bar and, for failures, an error box say why).
        """
        if self._closing:
            return False
        if not self._controller.request_export():
            self.statusBar().showMessage(self._export_blocked_reason())
            return False
        displayed = self._controller.displayed
        if displayed is None:  # the controller only says yes with a result on screen
            self.statusBar().showMessage(self._export_blocked_reason())
            return False
        return self._write_export(
            ExportPayload(displayed=displayed, export_config=self._controller.requested_config)
        )

    def _export_blocked_reason(self) -> str:
        """A one-line explanation for a refused export, taken from the controller's state."""
        controller = self._controller
        if controller.is_updating:
            return "The conversion is still running - export again when it finishes."
        if controller.displayed is None:
            return "Nothing to export yet - open an image and convert it first."
        if controller.is_stale:
            return "The displayed result no longer matches the settings - convert before exporting."
        return "Export is not available."

    def _write_export(self, payload: ExportPayload) -> bool:
        """Ask for a path and write the payload; every failure becomes a readable message."""
        result = payload.displayed.result
        verdicts = availability(result)
        formats = [fmt for fmt in EXPORT_FORMATS if verdicts[fmt] is None]
        if not formats:
            self._report_error(
                "Cannot export",
                "This result cannot be written in any supported format.",
                details="\n".join(
                    f"{EXPORT_DESCRIPTION[fmt]}: {verdicts[fmt]}"
                    for fmt in EXPORT_FORMATS
                    if verdicts[fmt]
                ),
            )
            return False

        target = self._ask_export_target(formats)
        if target is None:
            self.statusBar().showMessage("Export cancelled")
            return False
        path, fmt = target

        try:
            written = export_displayed(payload, fmt, path)
        except ValueError as exc:  # incompatible format chosen by name: explain, write nothing
            self._report_error("Cannot export", str(exc))
            self.statusBar().showMessage("Export failed - the format does not fit this result.")
            return False
        except OSError as exc:
            self._report_error(
                "Export failed",
                f"Could not write {path_with_suffix(path, fmt).name}.",
                details=str(exc),
            )
            self.statusBar().showMessage(f"Export failed - {exc}")
            return False

        self._export_directory = written[0].parent if written else Path(path).parent
        names = ", ".join(item.name for item in written)
        self.statusBar().showMessage(f"Exported {names} to {self._export_directory}")
        return True

    def _ask_export_target(self, formats: list[ExportFormat]) -> tuple[Path, ExportFormat] | None:
        """The save dialog: folder, name and format, pre-filled with the remembered folder."""
        filters = dialog_filters(formats)
        directory = self._export_directory
        if directory is None and self._source_path is not None:
            directory = self._source_path.parent
        stem = self._source_path.stem if self._source_path is not None else "screen"
        start = suggested_path(directory, stem, formats[0])
        selected = EXPORT_FILTER_SEPARATOR.join(text for text, _fmt in filters)
        path, chosen_filter = QFileDialog.getSaveFileName(
            self, "Export result", str(start), selected
        )
        if not path:
            return None
        fmt = chosen_format(path, chosen_filter, formats)
        if fmt is None:
            return None
        return Path(path), fmt

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

    # --- controller signals ---------------------------------------------

    def _on_result_ready(self, displayed: DisplayedConversion) -> None:
        if self._closing:
            return
        result = displayed.result
        frame = self._frame_renderer.frame(result, self._controller.requested_config)
        self._frame = frame
        height, width = result.indexed.shape
        self.preview.set_image_pair(
            ndarray_to_qimage(result.source_resized),
            ndarray_to_qimage(frame.rgb if frame is not None else result.converted_rgb),
            secondary_aspect=preview_pixel_aspect(
                self._controller.requested_config, size=(width, height)
            ),
        )
        self.metrics_panel.set_result(result)
        self.parameter_panel.set_palette_colors(
            [tuple(int(channel) for channel in colour) for colour in result.palette_rgb],
            size=(width, height),
        )
        self.hide_error_banner()
        self._refresh_state()

    def _on_config_changed(self, config: ConversionConfig) -> None:
        """Re-render ``Atari result`` when the export settings changed the hardware colours.

        Register edits and the *Use converter palette* switch are export-only: they must show up
        immediately, without waiting for (or triggering) another conversion. A change that leaves the
        bitmap, the mode and the registers alone hits the renderer's cache and redraws nothing.
        """
        if self._closing:
            return
        self._refresh_hardware_preview()

    def _refresh_hardware_preview(self) -> None:
        """Put the current hardware frame of the displayed result on screen (if it changed)."""
        displayed = self._controller.displayed
        if displayed is None:
            return
        frame = self._frame_renderer.frame(displayed.result, self._controller.requested_config)
        if frame is None or frame is self._frame:
            return  # no ANTIC mode for this size, or the very same pixels are already shown
        self._frame = frame
        height, width = displayed.result.indexed.shape
        self.preview.set_image_pair(
            ndarray_to_qimage(displayed.result.source_resized),
            ndarray_to_qimage(frame.rgb),
            secondary_aspect=preview_pixel_aspect(
                self._controller.requested_config, size=(width, height)
            ),
        )

    def _on_source_loaded(self, generation: int, identity: object) -> None:
        if self._closing:
            return
        self._loading_source = False
        if self._pending_source_path is not None:
            self._source_path = self._pending_source_path
            self._pending_source_path = None
        # A different image is a different thing to look at: fit it once. Later conversions of the same
        # source keep whatever view state the user chose (`ImagePreview.set_image_pair` preserves it).
        self.preview.fit()
        self._refresh_state()

    def _on_source_failed(self, generation: int, message: str) -> None:
        if self._closing:
            return
        self._loading_source = False
        self._pending_source_path = None
        self._show_error_banner("The image could not be loaded", message)
        self._refresh_state()

    def _on_conversion_failed(self, generation: int, message: str) -> None:
        if self._closing:
            return
        self._show_error_banner("The conversion failed", message)
        self._refresh_state()

    def _on_conversion_cancelled(self, generation: int) -> None:
        if self._closing:
            return
        self._refresh_state()  # a cancellation is not an error and shows no banner

    def _on_export_requested(self, displayed: DisplayedConversion) -> None:
        if self._closing:
            return
        payload = ExportPayload(
            displayed=displayed,
            export_config=self._controller.requested_config,
        )
        self.export_requested.emit(payload)

    # --- rendering ------------------------------------------------------

    def _refresh_state(self) -> None:
        """Render the controller's state and the window's own loading flag."""
        style, badge = self._badge()
        self.state_badge.setText(badge)
        self.state_badge.setStyleSheet(BADGE_STYLES[style])
        self.file_label.setText(self._file_text())
        self.statusBar().showMessage(self._status_text())
        self._update_actions()
        self._update_metrics_outdated()

    def _badge(self) -> tuple[str, str]:
        controller = self._controller
        if self._loading_source:
            name = self._pending_source_path.name if self._pending_source_path else "source"
            return "loading", f"loading {name}…"
        if self._closing:
            return "updating", controller.status_text()
        if self._source_path is None and controller.displayed is None:
            return "empty", "no image"
        if controller.last_error is not None and not controller.is_updating:
            return "error", controller.status_text()
        if controller.is_updating:
            return "updating", controller.status_text()
        if controller.displayed is None:
            return "stale", controller.status_text()
        return ("stale" if controller.is_stale else "ready"), controller.status_text()

    def _status_text(self) -> str:
        controller = self._controller
        if self._source_path is None and not self._loading_source:
            return NO_SOURCE_TEXT
        parts = [self._file_text(), controller.status_text()]
        displayed = controller.displayed
        if displayed is not None:
            total = sum(
                float(value)
                for value in displayed.timings.values()
                if isinstance(value, (int, float))
            )
            parts.append(f"stages {total * 1000:.1f} ms")
        return " - ".join(parts)

    def _file_text(self) -> str:
        if self._source_path is not None:
            return self._source_path.name
        if self._pending_source_path is not None:
            return f"{self._pending_source_path.name} (loading)"
        return NO_SOURCE_TEXT

    def _update_actions(self) -> None:
        controller = self._controller
        has_source = controller.source_identity is not None
        self.convert_action.setEnabled(
            has_source and not controller.is_updating and not self._closing
        )
        has_result = controller.displayed is not None
        self.export_action.setEnabled(
            has_result
            and not controller.is_updating
            and not controller.is_stale
            and not self._closing
        )
        self.export_action.setToolTip(
            "Export is only possible while the displayed result matches the settings"
            if has_result and controller.is_stale
            else "Save the displayed result (PNG, BIN, ASM, DTA or XEX)"
        )
        for action in self.zoom_actions.values():
            action.setEnabled(has_result)
        self.fit_action.setEnabled(has_result)
        if self.auto_update_action.isChecked() != controller.auto_update:
            self.auto_update_action.setChecked(controller.auto_update)

    def _update_metrics_outdated(self) -> None:
        displayed = self._controller.displayed
        if displayed is None or not self._controller.is_stale:
            self.metrics_panel.set_outdated(None)
        else:
            self.metrics_panel.set_outdated(displayed.generation)

    # --- error banner ---------------------------------------------------

    def _show_error_banner(self, title: str, message: str) -> None:
        """Non-modal error line: the preview and the metrics stay usable (plan §7.6)."""
        self._banner_text = f"{title}: {message}"
        self.error_label.setText(self._banner_text)
        self.error_banner.show()

    def hide_error_banner(self) -> None:
        """Hide the banner and forget its message (also called on every new result)."""
        self.error_banner.hide()
        self.error_label.setText("")
        self._banner_text = ""

    def show_error_details(self) -> None:
        """Details on demand - the banner itself only shows one line."""
        if not self._banner_text:
            return
        box = QMessageBox(
            QMessageBox.Icon.Warning,
            "Error details",
            self._banner_text,
            QMessageBox.StandardButton.Close,
            self,
        )
        box.exec()

    def _report_error(self, title: str, message: str, *, details: str | None = None) -> None:
        """Modal box for exceptional situations (unsupported file, shutdown timeout)."""
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

    # --- shutdown -------------------------------------------------------

    def shutdown(self, timeout_ms: int = SHUTDOWN_TIMEOUT_MS) -> bool:
        """Stop the controller and its backend; returns whether they really stopped."""
        if self._shutdown_result is True:
            return True
        self._closing = True
        self._loading_source = False
        stopped = self._controller.shutdown(timeout_ms)
        self._shutdown_result = stopped
        self._refresh_state()
        return stopped

    def closeEvent(self, event: QCloseEvent) -> None:
        """Never destroy a worker that is still running: on a timeout the window stays open."""
        if self.shutdown():
            event.accept()
            return
        event.ignore()
        self._warn_shutdown_timeout()

    def _warn_shutdown_timeout(self) -> None:
        self._report_error(
            "Still closing",
            "The conversion worker is still running and did not stop in time. "
            "Close the window again to retry.",
        )

    def show_about(self) -> None:
        """About box with the real package version."""
        QMessageBox.about(
            self,
            f"About {WINDOW_TITLE}",
            f"<h3>{WINDOW_TITLE} {converter.__version__}</h3>"
            "<p>Converts images to Atari 8-bit ANTIC graphics.</p>"
            "<p>Open a PNG, BMP, GIF or JPG image, adjust the parameters and press Convert.</p>",
        )
