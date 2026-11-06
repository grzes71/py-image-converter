"""The v1 conversion backend: one QThread, one worker object, one engine (plan §7.1, T3.1).

The engine is created once and then used exclusively inside the worker thread: ``set_source`` and
``convert`` are always executed there, and results travel back as Qt signals through queued connections,
so the GUI thread only ever receives plain values.

Cancellation is cooperative and generation-scoped. Every ``convert`` request owns a
:class:`threading.Event`; the worker passes ``event.is_set`` to the engine as its ``cancel`` callable, so
the engine stops at the next stage boundary. ``cancel(gen)`` only touches the event of that generation,
which means cancelling generation 10 can never cancel generation 11, and the GUI thread can set the event
while the worker is busy (a queued slot would never arrive until the conversion returned).
"""

from __future__ import annotations

import logging
import threading

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot

from converter.engine import ConversionCancelled, ConversionEngine
from converter.types import ConversionConfig

LOGGER = logging.getLogger(__name__)


class _EngineWorker(QObject):
    """Executes engine calls in the worker thread and reports their outcome."""

    source_loaded = Signal(int, object)
    finished = Signal(int, object)
    failed = Signal(int, str)
    cancelled = Signal(int)

    def __init__(self, engine: ConversionEngine) -> None:
        super().__init__()
        self._engine = engine

    @Slot(int, str, bool)
    def load_source(self, gen: int, path: str, verify_content: bool) -> None:
        """Install ``path`` as the engine source (runs in the worker thread)."""
        try:
            self._engine.set_source(path, verify_content=verify_content)
        except Exception as exc:
            LOGGER.exception("Worker could not load %s (generation %d)", path, gen)
            self.failed.emit(gen, str(exc) or type(exc).__name__)
            return
        self.source_loaded.emit(gen, self._engine.source_identity)

    @Slot(int, object, object)
    def convert(self, gen: int, config: ConversionConfig, cancel_event: threading.Event) -> None:
        """Run one conversion (runs in the worker thread)."""
        try:
            result = self._engine.convert(
                config,
                compute_metrics=True,
                cancel=cancel_event.is_set,
            )
        except ConversionCancelled:
            LOGGER.debug("Conversion of generation %d was cancelled", gen)
            self.cancelled.emit(gen)
        except Exception as exc:
            LOGGER.exception("Worker could not convert generation %d", gen)
            self.failed.emit(gen, str(exc) or type(exc).__name__)
        else:
            self.finished.emit(gen, result)


class QThreadBackend(QObject):
    """The v1 backend: one dedicated ``QThread`` with one worker object and one engine."""

    source_loaded = Signal(int, object)
    finished = Signal(int, object)
    failed = Signal(int, str)
    cancelled = Signal(int)

    _load_requested = Signal(int, str, bool)
    _convert_requested = Signal(int, object, object)

    def __init__(
        self, engine: ConversionEngine | None = None, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._cancel_events: dict[int, threading.Event] = {}
        self._shutdown_called = False

        self._thread = QThread()
        self._thread.setObjectName("ConversionWorker")
        self._worker = _EngineWorker(engine if engine is not None else ConversionEngine())
        self._worker.moveToThread(self._thread)

        self._worker.source_loaded.connect(self.source_loaded)
        self._worker.finished.connect(self.finished)
        self._worker.failed.connect(self.failed)
        self._worker.cancelled.connect(self.cancelled)
        self._worker.finished.connect(self._forget_result)
        self._worker.failed.connect(self._forget_failure)
        self._worker.cancelled.connect(self._forget_cancelled)

        self._load_requested.connect(self._worker.load_source, Qt.ConnectionType.QueuedConnection)
        self._convert_requested.connect(self._worker.convert, Qt.ConnectionType.QueuedConnection)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.start()

    # --- public API -----------------------------------------------------

    @property
    def is_running(self) -> bool:
        """True while the worker thread is alive; :meth:`shutdown` stops it."""
        return self._thread.isRunning()

    def load_source(self, gen: int, path: str, verify_content: bool = False) -> None:
        """Queue the source load; the answer arrives as ``source_loaded`` or ``failed``."""
        self._ensure_alive()
        self._load_requested.emit(gen, str(path), verify_content)

    def convert(self, gen: int, config: ConversionConfig) -> None:
        """Queue a conversion that ``cancel(gen)`` can stop between stages."""
        self._ensure_alive()
        event = threading.Event()
        with self._lock:
            self._cancel_events[gen] = event
        self._convert_requested.emit(gen, config, event)

    def cancel(self, gen: int) -> None:
        """Cancel generation ``gen`` from any thread; earlier or later generations are untouched."""
        with self._lock:
            event = self._cancel_events.get(gen)
        if event is None:
            LOGGER.debug("cancel(%d): no operation of that generation is known", gen)
            return
        event.set()

    def shutdown(self, timeout_ms: int) -> bool:
        """Cancel the current work, quit the worker loop and wait up to ``timeout_ms``.

        Returns ``True`` when the thread really stopped. ``False`` means a stage was still running when
        the timeout expired: the backend stays alive (the thread and the worker are *not* destroyed), the
        caller may wait again, and no Qt object is deleted while the thread still uses it. Calling
        ``shutdown`` twice is safe.
        """
        with self._lock:
            self._shutdown_called = True
            events = list(self._cancel_events.values())
        for event in events:  # cancel whatever is running or still queued
            event.set()
        if not self._thread.isRunning():
            return True
        self._thread.quit()
        stopped = self._thread.wait(timeout_ms)
        if not stopped:
            LOGGER.warning("Conversion thread did not stop within %d ms", timeout_ms)
        return stopped

    # --- internals ------------------------------------------------------

    def _ensure_alive(self) -> None:
        if self._shutdown_called or not self._thread.isRunning():
            raise RuntimeError("the conversion backend has been shut down")

    @Slot(int, object)
    def _forget_result(self, gen: int, payload: object) -> None:
        self._forget(gen)

    @Slot(int, str)
    def _forget_failure(self, gen: int, message: str) -> None:
        self._forget(gen)

    @Slot(int)
    def _forget_cancelled(self, gen: int) -> None:
        self._forget(gen)

    def _forget(self, gen: int) -> None:
        """Drop the cancellation event of a finished generation."""
        with self._lock:
            self._cancel_events.pop(gen, None)
