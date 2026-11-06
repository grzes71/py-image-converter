"""Shared doubles for the GUI integration tests (T3.4).

``FakeBackend`` answers only when a test asks it to, so the ordering that matters - debounce, coalescing,
stale results, cancellation, shutdown - is reproduced exactly instead of being timed. The builders keep
the window-level suites free of boilerplate and in one place, so ``test_main_window.py`` and
``test_stale_guard.py`` share one fake instead of growing two.

Importing this module needs PySide6; every test module guards with ``pytest.importorskip`` first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from PySide6.QtCore import QObject, Signal

from converter.types import (
    ConversionConfig,
    ConversionResult,
    MetricResult,
    SourceIdentity,
)
from converter_gui.main_window import MainWindow

DEBOUNCE_MS = 5
"""Short on purpose: the window accepts a debounce override, the production default stays 300 ms."""

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
REFERENCE_IMAGE = FIXTURES / "reference.png"

CREATED: list[MainWindow] = []
"""Every window built through this module; the autouse fixture in ``conftest.py`` stops them all."""


def make_metrics() -> MetricResult:
    """Fixed metrics so assertions can check exact rows."""
    return MetricResult(
        mse=12.5,
        psnr=27.25,
        ssim=0.9375,
        mean_rgb_error=5.0,
        mean_luminance_error=4.25,
        colors_used=4,
        histogram_coverage=0.625,
    )


def make_palette(count: int = 4) -> np.ndarray:
    """A tiny palette whose first colour is easy to recognise in a widget."""
    colours = [(10, 20, 30), (40, 50, 60), (70, 80, 90), (1, 2, 3)]
    return np.array(colours[:count], dtype=np.uint8).reshape(count, 3)


def make_result(colors: int = 4, height: int = 8, width: int = 10) -> ConversionResult:
    """A tiny but complete result: two RGB images, a palette, metrics and stage timings."""
    return ConversionResult(
        indexed=np.zeros((height, width), dtype=np.uint8),
        palette_indices=list(range(colors)),
        palette_rgb=make_palette(colors),
        source_resized=np.zeros((height, width, 3), dtype=np.uint8),
        metrics=make_metrics(),
        timings={
            "prepare": 0.010,
            "palette": 0.002,
            "edges": 0.001,
            "render": 0.100,
            "metrics": 0.002,
        },
    )


class FakeBackend(QObject):
    """A backend that records requests and answers only when the test asks it to."""

    source_loaded = Signal(int, object)
    finished = Signal(int, object)
    failed = Signal(int, str)
    cancelled = Signal(int)

    def __init__(
        self, *, auto_load: bool = True, auto_convert: bool = False, shutdown_result: bool = True
    ) -> None:
        super().__init__()
        self.calls: list[tuple] = []
        self.auto_load = auto_load
        self.auto_convert = auto_convert
        self.shutdown_result = shutdown_result
        self.fail_loads = False
        self.running: int | None = None

    # --- the ConversionBackend protocol --------------------------------

    def load_source(self, gen: int, path: str, verify_content: bool = False) -> None:
        self.calls.append(("load", gen, str(path), verify_content))
        if not self.auto_load:
            return
        if self.fail_loads:
            self.failed.emit(gen, "unreadable image")
            return
        self.source_loaded.emit(gen, SourceIdentity(path=str(path), size=3, mtime_ns=4))

    def convert(self, gen: int, config: ConversionConfig) -> None:
        self.calls.append(("convert", gen, config))
        self.running = gen
        if self.auto_convert:
            self.finish()

    def cancel(self, gen: int) -> None:
        self.calls.append(("cancel", gen))

    def shutdown(self, timeout_ms: int) -> bool:
        self.calls.append(("shutdown", timeout_ms))
        if self.shutdown_result:
            self.running = None
        return self.shutdown_result

    # --- test controls -------------------------------------------------

    def kinds(self) -> list[str]:
        """The recorded request kinds, in order."""
        return [call[0] for call in self.calls]

    def conversions(self) -> list[int]:
        """Generations of the recorded conversions."""
        return [call[1] for call in self.calls if call[0] == "convert"]

    def configs(self) -> list[ConversionConfig]:
        """The configurations handed to the backend, in order."""
        return [call[2] for call in self.calls if call[0] == "convert"]

    def finish(self, result: ConversionResult | None = None) -> None:
        gen, self.running = self.running, None
        assert gen is not None, "no conversion is running"
        self.finished.emit(gen, result if result is not None else make_result())

    def fail(self, message: str = "boom") -> None:
        gen, self.running = self.running, None
        assert gen is not None, "no conversion is running"
        self.failed.emit(gen, message)

    def emit_cancelled(self) -> None:
        gen, self.running = self.running, None
        assert gen is not None, "no conversion is running"
        self.cancelled.emit(gen)

    def finish_late(self, gen: int, result: ConversionResult) -> None:
        """Answer for a generation the controller has already superseded."""
        self.finished.emit(gen, result)

    def fail_late(self, gen: int, message: str) -> None:
        self.failed.emit(gen, message)


def build_window(
    qtbot, backend: FakeBackend | None = None, **kwargs
) -> tuple[MainWindow, FakeBackend, list[tuple[str, str]]]:
    """A window plus its fake backend and the modal boxes it would have opened."""
    fake = FakeBackend() if backend is None else backend
    kwargs.setdefault("debounce_ms", DEBOUNCE_MS)
    window = MainWindow(backend=fake, **kwargs)
    qtbot.addWidget(window)
    CREATED.append(window)
    boxes: list[tuple[str, str]] = []
    window._report_error = lambda title, message, **rest: boxes.append((title, message))
    window._warn_shutdown_timeout = lambda: boxes.append(("Still closing", ""))
    return window, fake, boxes


def open_window(
    qtbot, path: Path = REFERENCE_IMAGE
) -> tuple[MainWindow, FakeBackend, list[tuple[str, str]]]:
    """A window with ``path`` handed to the controller (and, by default, one conversion running)."""
    window, backend, boxes = build_window(qtbot)
    assert window.open_image(path) is True
    return window, backend, boxes


def converted_window(
    qtbot, path: Path = REFERENCE_IMAGE
) -> tuple[MainWindow, FakeBackend, list[tuple[str, str]]]:
    """A window with one finished conversion of ``path`` on screen."""
    window, backend, boxes = open_window(qtbot, path)
    backend.finish()
    return window, backend, boxes
