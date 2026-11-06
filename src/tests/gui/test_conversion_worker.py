"""Tests for the QThread backend (T3.1).

Everything is synchronised with ``threading.Event`` handshakes and pytest-qt's signal helpers - no test
asserts a timing or waits for a fixed sleep. The double records the thread it ran on, the configuration
it received and what its ``cancel`` callable reported, so the assertions cover behaviour rather than the
mere emission of a signal.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

pytest.importorskip("PySide6", reason="PySide6 is not installed (install the 'gui' extra)")

from PySide6.QtTest import QSignalSpy  # noqa: E402

from converter.engine import ConversionCancelled  # noqa: E402
from converter.types import (  # noqa: E402
    ConversionConfig,
    ConversionResult,
    DitherMethod,
    PaletteMethod,
    SourceIdentity,
)
from converter_gui.workers.base import ConversionBackend  # noqa: E402
from converter_gui.workers.conversion_worker import QThreadBackend  # noqa: E402

REFERENCE_IMAGE = Path(__file__).resolve().parents[1] / "fixtures" / "reference.png"
GUI_THREAD = threading.get_ident()


class FakeEngine:
    """Engine double: records the thread, can block, and can refuse to notice cancellation."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.thread_ids: dict[str, int] = {}
        self.source_identity: SourceIdentity | None = None
        self.path: str | None = None
        self.verify_content: bool | None = None
        self.load_error: BaseException | None = None
        self.convert_error: BaseException | None = None
        self.block = False
        self.ignore_cancel = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.cancel_seen: dict[int, bool] = {}
        self.configs: list[ConversionConfig] = []

    def set_source(self, path: str, *, verify_content: bool = False) -> None:
        self.calls.append("set_source")
        self.thread_ids["set_source"] = threading.get_ident()
        self.path = path
        self.verify_content = verify_content
        if self.load_error is not None:
            raise self.load_error
        self.source_identity = SourceIdentity(path=str(path), size=11, mtime_ns=22)

    def convert(
        self,
        config: ConversionConfig,
        *,
        compute_metrics: bool | None = None,
        cancel: object = None,
    ) -> object:
        self.calls.append("convert")
        self.thread_ids["convert"] = threading.get_ident()
        self.configs.append(config)
        key = config.colors
        self.entered.set()
        if self.block:
            # The 5 ms wait only paces the poll loop that lets the double notice a cancellation; no
            # assertion depends on how long anything takes.
            while not self.release.is_set():
                cancelled = bool(cancel()) if callable(cancel) else False
                self.cancel_seen[key] = cancelled
                if cancelled and not self.ignore_cancel:
                    raise ConversionCancelled()
                self.release.wait(0.005)
        self.cancel_seen[key] = bool(cancel()) if callable(cancel) else False
        if self.convert_error is not None:
            raise self.convert_error
        return f"result-{key}"


@pytest.fixture()
def make_backend(qtbot):
    """Create backends and always shut them down, so no thread outlives its test."""
    created: list[QThreadBackend] = []

    def factory(engine: object | None = None) -> QThreadBackend:
        backend = QThreadBackend(engine=engine)  # type: ignore[arg-type]
        created.append(backend)
        return backend

    yield factory

    for backend in created:
        if backend.is_running:
            backend.shutdown(5000)


def _config(colors: int) -> ConversionConfig:
    return ConversionConfig(
        width=160,
        height=192,
        colors=colors,
        dither=DitherMethod.NONE,
        palette=PaletteMethod.MEDIAN_CUT,
    )


def _qt_log_text(qtlog: object) -> str:
    """Every Qt message captured by pytest-qt, joined into one searchable string."""
    records = getattr(qtlog, "records", [])
    return " | ".join(str(getattr(record, "message", record)) for record in records)


def _enter_and_wait(qtbot, engine: FakeEngine) -> None:
    """Wait until the worker is inside ``convert``."""
    qtbot.waitUntil(lambda: engine.entered.is_set(), timeout=10000)


# --- protocol conformance and threading ---------------------------------


def test_the_backend_satisfies_the_protocol(qtbot, make_backend) -> None:
    backend = make_backend(FakeEngine())

    for name in ("load_source", "convert", "cancel", "shutdown"):
        assert callable(getattr(backend, name))
    for name in ("source_loaded", "finished", "failed", "cancelled"):
        assert hasattr(backend, name)
    assert isinstance(backend, ConversionBackend) is True


def test_the_worker_runs_off_the_gui_thread(qtbot, make_backend) -> None:
    engine = FakeEngine()
    backend = make_backend(engine)

    with qtbot.waitSignal(backend.source_loaded, timeout=15000):
        backend.load_source(1, "source.png")

    assert engine.thread_ids["set_source"] != GUI_THREAD
    assert backend.is_running is True


def test_loading_and_converting_share_the_worker_thread(qtbot, make_backend) -> None:
    engine = FakeEngine()
    backend = make_backend(engine)

    with qtbot.waitSignal(backend.source_loaded, timeout=15000):
        backend.load_source(1, "source.png")
    with qtbot.waitSignal(backend.finished, timeout=15000):
        backend.convert(2, _config(4))

    assert engine.thread_ids["set_source"] == engine.thread_ids["convert"]
    assert engine.thread_ids["convert"] != GUI_THREAD
    assert engine.calls == ["set_source", "convert"]


# --- payloads -----------------------------------------------------------


def test_source_loaded_reports_the_generation_and_the_identity(qtbot, make_backend) -> None:
    engine = FakeEngine()
    backend = make_backend(engine)

    with qtbot.waitSignal(backend.source_loaded, timeout=15000) as blocker:
        backend.load_source(4, "source.png", verify_content=True)

    generation, info = blocker.args
    assert generation == 4
    assert isinstance(info, SourceIdentity)
    assert info is engine.source_identity
    assert engine.verify_content is True
    assert engine.path == "source.png"


def test_finished_reports_the_generation_and_the_result(qtbot, make_backend) -> None:
    engine = FakeEngine()
    backend = make_backend(engine)

    with qtbot.waitSignal(backend.source_loaded, timeout=15000):
        backend.load_source(1, "source.png")
    with qtbot.waitSignal(backend.finished, timeout=15000) as blocker:
        backend.convert(9, _config(3))

    generation, result = blocker.args
    assert generation == 9
    assert result == "result-3"
    assert engine.configs[-1].colors == 3


def test_a_load_failure_emits_failed_and_nothing_else(qtbot, make_backend) -> None:
    engine = FakeEngine()
    engine.load_error = OSError("cannot read the file")
    backend = make_backend(engine)
    loaded = QSignalSpy(backend.source_loaded)

    with qtbot.waitSignal(backend.failed, timeout=15000) as blocker:
        backend.load_source(2, "broken.png")

    assert blocker.args == [2, "cannot read the file"]
    assert loaded.count() == 0


def test_a_conversion_failure_emits_failed_and_nothing_else(qtbot, make_backend) -> None:
    engine = FakeEngine()
    engine.convert_error = RuntimeError("palette exploded")
    backend = make_backend(engine)
    finished = QSignalSpy(backend.finished)

    with qtbot.waitSignal(backend.failed, timeout=15000) as blocker:
        backend.convert(5, _config(4))

    assert blocker.args == [5, "palette exploded"]
    assert finished.count() == 0


# --- cancellation -------------------------------------------------------


def test_conversion_cancelled_emits_cancelled(qtbot, make_backend) -> None:
    engine = FakeEngine()
    engine.block = True
    backend = make_backend(engine)
    finished = QSignalSpy(backend.finished)
    cancelled = QSignalSpy(backend.cancelled)

    backend.convert(5, _config(2))
    _enter_and_wait(qtbot, engine)
    backend.cancel(5)
    qtbot.waitUntil(lambda: cancelled.count() == 1, timeout=15000)

    assert cancelled.at(0) == [5]
    assert finished.count() == 0
    assert engine.cancel_seen[2] is True


def test_cancel_is_observed_from_the_gui_thread_while_converting(qtbot, make_backend) -> None:
    """A queued slot would never arrive while the worker is busy; the event must be set directly."""
    engine = FakeEngine()
    engine.block = True
    backend = make_backend(engine)

    backend.convert(6, _config(2))
    _enter_and_wait(qtbot, engine)
    assert engine.cancel_seen.get(2) is False, "nothing cancelled it yet"

    backend.cancel(6)

    qtbot.waitUntil(lambda: engine.cancel_seen.get(2) is True, timeout=15000)


def test_cancelling_one_generation_does_not_cancel_the_next(qtbot, make_backend) -> None:
    engine = FakeEngine()
    engine.block = True
    backend = make_backend(engine)

    backend.convert(10, _config(2))
    _enter_and_wait(qtbot, engine)
    with qtbot.waitSignal(backend.cancelled, timeout=15000):
        backend.cancel(10)

    engine.release.set()  # let the following conversion run to completion
    with qtbot.waitSignal(backend.finished, timeout=15000) as blocker:
        backend.convert(11, _config(3))

    assert blocker.args[0] == 11
    assert engine.cancel_seen[2] is True, "generation 10 was cancelled"
    assert engine.cancel_seen[3] is False, "generation 11 was not"


def test_the_worker_accepts_a_new_conversion_after_a_cancellation(qtbot, make_backend) -> None:
    engine = FakeEngine()
    engine.block = True
    backend = make_backend(engine)

    backend.convert(1, _config(2))
    _enter_and_wait(qtbot, engine)
    with qtbot.waitSignal(backend.cancelled, timeout=15000):
        backend.cancel(1)

    engine.release.set()
    with qtbot.waitSignal(backend.finished, timeout=15000) as blocker:
        backend.convert(2, _config(3))

    assert blocker.args[0] == 2


def test_cancel_for_an_unknown_generation_is_ignored(qtbot, make_backend) -> None:
    engine = FakeEngine()
    backend = make_backend(engine)

    backend.cancel(999)  # nothing of that generation is running

    with qtbot.waitSignal(backend.finished, timeout=15000) as blocker:
        backend.convert(1, _config(4))
    assert blocker.args[0] == 1


def test_the_worker_accepts_a_new_conversion_after_a_failure(qtbot, make_backend) -> None:
    engine = FakeEngine()
    engine.convert_error = RuntimeError("boom")
    backend = make_backend(engine)

    with qtbot.waitSignal(backend.failed, timeout=15000):
        backend.convert(1, _config(2))

    engine.convert_error = None
    with qtbot.waitSignal(backend.finished, timeout=15000) as blocker:
        backend.convert(2, _config(3))

    assert blocker.args[0] == 2
    assert engine.calls == ["convert", "convert"]


# --- shutdown -----------------------------------------------------------


def test_shutdown_when_idle_stops_the_thread(make_backend) -> None:
    backend = make_backend(FakeEngine())
    assert backend.is_running is True

    assert backend.shutdown(5000) is True

    assert backend.is_running is False
    assert backend.shutdown(1000) is True, "a second shutdown is harmless"


def test_shutdown_during_a_conversion_cancels_it(qtbot, make_backend, qtlog) -> None:
    engine = FakeEngine()
    engine.block = True
    backend = make_backend(engine)
    cancelled = QSignalSpy(backend.cancelled)

    backend.convert(1, _config(2))
    _enter_and_wait(qtbot, engine)

    stopped = backend.shutdown(15000)

    assert stopped is True
    assert backend.is_running is False
    assert engine.cancel_seen[2] is True, "shutdown cancelled the running generation"
    qtbot.waitUntil(lambda: cancelled.count() == 1, timeout=15000)
    assert cancelled.at(0) == [1]
    assert "still running" not in _qt_log_text(qtlog)


def test_a_shutdown_timeout_keeps_the_thread_alive(qtbot, make_backend, qtlog) -> None:
    """A stage that ignores cancellation cannot be interrupted: report it, never fake the stop."""
    engine = FakeEngine()
    engine.block = True
    engine.ignore_cancel = True
    backend = make_backend(engine)

    backend.convert(1, _config(2))
    _enter_and_wait(qtbot, engine)

    assert backend.shutdown(100) is False
    assert backend.is_running is True, "the thread is still inside the stage"
    assert "still running" not in _qt_log_text(qtlog)

    engine.release.set()
    assert backend.shutdown(15000) is True
    assert backend.is_running is False


def test_requests_after_shutdown_are_refused(make_backend) -> None:
    backend = make_backend(FakeEngine())
    assert backend.shutdown(5000) is True

    with pytest.raises(RuntimeError, match="shut down"):
        backend.convert(1, _config(4))
    with pytest.raises(RuntimeError, match="shut down"):
        backend.load_source(2, "source.png")


# --- integration with the real engine -----------------------------------


def test_a_real_conversion_runs_through_the_engine(qtbot, make_backend) -> None:
    backend = make_backend()

    with qtbot.waitSignal(backend.source_loaded, timeout=30000) as loaded:
        backend.load_source(1, str(REFERENCE_IMAGE))
    generation, identity = loaded.args
    assert generation == 1
    assert isinstance(identity, SourceIdentity)
    assert identity.path == str(REFERENCE_IMAGE.resolve())

    config = ConversionConfig(
        width=160,
        height=192,
        dither=DitherMethod.NONE,
        palette=PaletteMethod.MEDIAN_CUT,
    )
    with qtbot.waitSignal(backend.finished, timeout=60000) as finished:
        backend.convert(2, config)

    result_generation, result = finished.args
    assert result_generation == 2
    assert isinstance(result, ConversionResult)
    assert result.indexed.shape == (192, 160)
    assert result.metrics is not None
    assert result.indexed.flags.writeable is False, "engine buffers stay frozen across threads"
    assert backend.shutdown(15000) is True
    assert backend.is_running is False
