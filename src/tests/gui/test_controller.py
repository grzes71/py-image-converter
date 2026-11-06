"""Tests for the conversion controller (T3.2).

The fake backend lets each test decide when a load or a conversion finishes, so the ordering that matters
- debounce, coalescing, stale results, cancellation, shutdown - is reproduced exactly instead of being
timed. pytest-qt's waitSignal/waitUntil drive the event loop; nothing waits for a fixed sleep.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

pytest.importorskip("PySide6", reason="PySide6 is not installed (install the 'gui' extra)")

from PySide6.QtCore import QObject, Signal  # noqa: E402
from PySide6.QtTest import QSignalSpy  # noqa: E402

from converter.types import (  # noqa: E402
    ConversionConfig,
    ConversionResult,
    ExportFormat,
    SourceIdentity,
)
from converter_gui.controllers.conversion_controller import (  # noqa: E402
    ConversionController,
    DisplayedConversion,
    image_key,
)
from converter_gui.workers.base import ConversionBackend  # noqa: E402
from converter_gui.workers.conversion_worker import QThreadBackend  # noqa: E402

REFERENCE_IMAGE = Path(__file__).resolve().parents[1] / "fixtures" / "reference.png"


class FakeResult:
    """Stand-in for ``ConversionResult``: the controller only reads ``timings`` from it."""

    def __init__(self, generation: int) -> None:
        self.generation = generation
        self.timings = {"prepare": 0.001 * generation, "render": 0.002}


class FakeBackend(QObject):
    """A controllable backend: requests are recorded, replies are emitted by the test."""

    source_loaded = Signal(int, object)
    finished = Signal(int, object)
    failed = Signal(int, str)
    cancelled = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple] = []
        self.conversions: list[tuple[int, ConversionConfig]] = []
        self.cancellations: list[int] = []
        self.shutdown_calls: list[int] = []
        self.auto_load = True
        self.auto_finish = False
        self.shutdown_result = True
        self.running: int | None = None

    # ConversionBackend --------------------------------------------------

    def load_source(self, gen: int, path: str, verify_content: bool = False) -> None:
        self.calls.append(("load", gen, str(path), verify_content))
        if self.auto_load:
            self.source_loaded.emit(gen, SourceIdentity(path=str(path), size=1, mtime_ns=2))

    def convert(self, gen: int, config: ConversionConfig) -> None:
        self.calls.append(("convert", gen, config))
        self.conversions.append((gen, config))
        self.running = gen
        if self.auto_finish:
            self.finish(gen)

    def cancel(self, gen: int) -> None:
        self.calls.append(("cancel", gen))
        self.cancellations.append(gen)

    def shutdown(self, timeout_ms: int) -> bool:
        self.calls.append(("shutdown", timeout_ms))
        self.shutdown_calls.append(timeout_ms)
        return self.shutdown_result

    # helpers for the tests ----------------------------------------------

    @property
    def convert_generations(self) -> list[int]:
        return [generation for generation, _ in self.conversions]

    def finish(self, gen: int | None = None, result: object | None = None) -> None:
        generation = self.running if gen is None else gen
        assert generation is not None, "nothing is running"
        self.running = None
        self.finished.emit(generation, result if result is not None else FakeResult(generation))

    def fail(self, gen: int | None = None, message: str = "boom") -> None:
        generation = self.running if gen is None else gen
        assert generation is not None, "nothing is running"
        self.running = None
        self.failed.emit(generation, message)

    def fail_load(self, gen: int, message: str = "cannot load") -> None:
        self.failed.emit(gen, message)

    def confirm_cancelled(self, gen: int | None = None) -> None:
        generation = self.running if gen is None else gen
        assert generation is not None, "nothing is running"
        self.running = None
        self.cancelled.emit(generation)


@pytest.fixture()
def make_controller(qtbot):
    """Build a controller around a fresh fake backend and shut everything down afterwards."""
    created: list[ConversionController] = []

    def build(
        *, debounce_ms: int = 5, auto_update: bool = True
    ) -> tuple[ConversionController, FakeBackend]:
        backend = FakeBackend()
        controller = ConversionController(backend, debounce_ms=debounce_ms, auto_update=auto_update)
        created.append(controller)
        return controller, backend

    yield build

    for controller in created:
        controller.shutdown(1000)


def _with(controller: ConversionController, **changes: object) -> ConversionConfig:
    """A copy of the requested configuration with some fields changed."""
    return dataclasses.replace(controller.requested_config, **changes)


def _load(controller: ConversionController, backend: FakeBackend, path: str = "photo.png") -> None:
    """Load a source and make sure exactly one conversion is running for it."""
    controller.set_source(path)
    if not backend.conversions:
        controller.convert_now()  # auto-update may be off
    assert len(backend.conversions) == 1


def _ready(controller: ConversionController, backend: FakeBackend) -> DisplayedConversion:
    """Bring the controller into the ready state with generation 1 displayed."""
    _load(controller, backend)
    backend.finish()
    displayed = controller.displayed
    assert isinstance(displayed, DisplayedConversion)
    assert controller.is_stale is False
    return displayed


# --- the five mandatory scenarios ---------------------------------------


def test_the_fake_backend_satisfies_the_protocol() -> None:
    assert isinstance(FakeBackend(), ConversionBackend) is True


def test_debounce_turns_five_changes_into_one_conversion(qtbot, make_controller) -> None:
    controller, backend = make_controller(debounce_ms=50)
    _ready(controller, backend)
    assert backend.convert_generations == [1]

    for colors in (2, 3, 2, 3, 2):
        controller.set_config(_with(controller, colors=colors))

    qtbot.waitUntil(lambda: len(backend.conversions) == 2, timeout=3000)
    generation, config = backend.conversions[-1]
    assert backend.convert_generations == [1, generation]
    assert generation == controller.latest_generation
    assert generation == 6, "one bump per change"
    assert config.colors == 2, "the converted configuration is the newest one"


def test_a_late_result_of_an_old_generation_is_dropped(qtbot, make_controller) -> None:
    controller, backend = make_controller()
    _load(controller, backend)  # generation 1 is converting
    ready_spy = QSignalSpy(controller.result_ready)

    controller.set_config(
        _with(controller, colors=2)
    )  # generation 2 pending, generation 1 cancelled
    assert backend.cancellations == [1]

    backend.confirm_cancelled(1)  # the controller hears back and starts generation 2
    qtbot.waitUntil(lambda: backend.convert_generations == [1, 2], timeout=3000)

    backend.finished.emit(1, FakeResult(1))  # a stray result of the cancelled generation

    assert ready_spy.count() == 0
    assert controller.displayed is None, "an outdated result is never shown"

    backend.finish(2)

    assert ready_spy.count() == 1
    assert controller.displayed is not None
    assert controller.displayed.generation == 2


def test_a_result_for_a_superseded_generation_is_dropped(qtbot, make_controller) -> None:
    controller, backend = make_controller()
    displayed = _ready(controller, backend)

    controller.set_config(_with(controller, colors=2))  # generation 2 starts
    qtbot.waitUntil(lambda: backend.convert_generations == [1, 2], timeout=3000)
    controller.set_config(_with(controller, colors=3))  # generation 3 waits

    backend.finish(2)  # completes after it stopped being current

    assert controller.displayed is displayed, "generation 2 must not be shown"
    qtbot.waitUntil(lambda: backend.convert_generations == [1, 2, 3], timeout=3000)
    backend.finish(3)
    assert controller.displayed is not None
    assert controller.displayed.generation == 3
    assert controller.is_stale is False


def test_changes_during_a_conversion_coalesce_into_one_follow_up(qtbot, make_controller) -> None:
    controller, backend = make_controller()
    _load(controller, backend)  # generation 1 is converting
    assert backend.convert_generations == [1]

    for colors in (2, 3, 2):  # generations 2, 3 and 4
        controller.set_config(_with(controller, colors=colors))

    assert backend.convert_generations == [1], "nothing runs in parallel"
    assert backend.cancellations == [1], "the outdated generation is cancelled once"

    backend.finish(1)  # generation 1 answers (outdated) and releases the queue

    qtbot.waitUntil(lambda: len(backend.conversions) == 2, timeout=3000)
    assert backend.convert_generations == [1, 4], "generations 2 and 3 were never converted"
    assert backend.conversions[-1][1].colors == 2


def test_a_failed_conversion_keeps_the_last_good_result(qtbot, make_controller) -> None:
    controller, backend = make_controller()
    displayed = _ready(controller, backend)
    failures = QSignalSpy(controller.conversion_failed)

    controller.set_config(_with(controller, colors=2))
    qtbot.waitUntil(lambda: len(backend.conversions) == 2, timeout=3000)
    backend.fail()

    assert failures.count() == 1
    assert failures.at(0)[1] == "boom"
    assert controller.displayed is displayed, "the last good result survives an error"
    assert controller.last_error == "boom"
    assert controller.status_text() == "error - still showing gen 1"

    controller.convert_now()  # the controller keeps working after a failure
    qtbot.waitUntil(lambda: len(backend.conversions) == 3, timeout=3000)
    backend.finish()

    assert controller.last_error is None
    assert controller.displayed is not None
    assert controller.displayed.generation == controller.latest_generation
    assert controller.is_stale is False


def test_shutdown_cancels_the_running_job_and_stops_the_backend(qtbot, make_controller) -> None:
    controller, backend = make_controller()
    _load(controller, backend)
    assert backend.convert_generations == [1]

    assert controller.shutdown(500) is True

    assert backend.cancellations == [1], "the running generation was cancelled"
    assert backend.shutdown_calls == [500]
    assert controller.is_updating is False
    assert controller.status_text() == "shut down"

    calls_before = len(backend.calls)
    controller.set_config(_with(controller, colors=2))
    controller.convert_now()
    controller.set_source("other.png")
    assert controller.request_export() is False
    assert len(backend.calls) == calls_before, "nothing reaches the backend after shutdown"

    assert controller.shutdown(500) is True, "a repeated shutdown is a no-op"
    assert backend.shutdown_calls == [500]


# --- source loading -----------------------------------------------------


def test_loading_a_source_reports_it_and_converts_it(qtbot, make_controller) -> None:
    controller, backend = make_controller()
    loaded = QSignalSpy(controller.source_loaded)

    controller.set_source("photo.png", verify_content=True)

    assert backend.calls[0] == ("load", 1, "photo.png", True)
    assert loaded.count() == 1
    generation, identity = loaded.at(0)
    assert generation == 1
    assert isinstance(identity, SourceIdentity)
    assert controller.source_identity is identity
    assert backend.convert_generations == [1], "the conversion follows the load"


def test_a_failed_load_does_not_look_like_a_loaded_source(qtbot, make_controller) -> None:
    controller, backend = make_controller()
    backend.auto_load = False
    failures = QSignalSpy(controller.source_failed)

    controller.set_source("broken.png")
    backend.fail_load(1, "cannot read the file")

    assert failures.count() == 1
    assert failures.at(0) == [1, "cannot read the file"]
    assert controller.source_identity is None
    assert controller.last_error == "cannot read the file"
    assert backend.convert_generations == [], "no conversion on an unknown source"
    assert controller.displayed is None


def test_changing_the_source_during_a_conversion_waits_for_the_new_source(
    qtbot, make_controller
) -> None:
    controller, backend = make_controller()
    _load(controller, backend)  # generation 1 converts the first source
    backend.auto_load = False

    controller.set_source("second.png")  # generation 2: load in flight

    assert ("cancel", 1) in backend.calls
    assert backend.convert_generations == [1], "no conversion before the new source arrives"

    backend.source_loaded.emit(2, SourceIdentity(path="second.png", size=2, mtime_ns=3))
    backend.confirm_cancelled(1)  # the backend reports the cancellation of generation 1
    qtbot.waitUntil(lambda: len(backend.conversions) == 2, timeout=3000)

    assert backend.convert_generations == [1, 2]
    assert controller.source_identity is not None
    assert controller.source_identity.path == "second.png"


def test_a_late_load_of_a_superseded_generation_is_ignored(qtbot, make_controller) -> None:
    controller, backend = make_controller()
    backend.auto_load = False
    loaded = QSignalSpy(controller.source_loaded)

    controller.set_source("first.png")  # generation 1
    controller.set_source("second.png")  # generation 2 supersedes it
    backend.source_loaded.emit(1, SourceIdentity(path="first.png"))
    backend.source_loaded.emit(2, SourceIdentity(path="second.png"))

    assert loaded.count() == 1, "only the newest load counts"
    assert loaded.at(0)[0] == 2
    assert controller.source_identity is not None
    assert controller.source_identity.path == "second.png"


# --- auto-update and manual convert -------------------------------------


def test_auto_update_off_only_makes_the_result_stale(qtbot, make_controller) -> None:
    controller, backend = make_controller(debounce_ms=5, auto_update=False)
    _ready(controller, backend)

    controller.set_config(_with(controller, colors=2))

    assert controller.latest_generation == 2
    assert controller.is_stale is True
    assert backend.convert_generations == [1], "auto-update is off: nothing converts"

    controller.convert_now()

    qtbot.waitUntil(lambda: backend.convert_generations == [1, 2], timeout=3000)


def test_manual_convert_ignores_the_debounce_delay(qtbot, make_controller) -> None:
    controller, backend = make_controller(debounce_ms=60000, auto_update=False)
    _ready(controller, backend)

    controller.set_config(_with(controller, colors=3))
    controller.convert_now()

    assert backend.convert_generations == [1, 2], "dispatched immediately"


def test_enabling_auto_update_schedules_the_pending_change(qtbot, make_controller) -> None:
    controller, backend = make_controller(debounce_ms=5, auto_update=False)
    _ready(controller, backend)
    controller.set_config(_with(controller, colors=2))
    assert backend.convert_generations == [1]

    controller.set_auto_update(True)

    qtbot.waitUntil(lambda: backend.convert_generations == [1, 2], timeout=3000)


def test_manual_convert_is_a_no_op_when_the_result_is_current(qtbot, make_controller) -> None:
    controller, backend = make_controller()
    _ready(controller, backend)

    controller.convert_now()

    assert backend.convert_generations == [1]


def test_an_export_only_change_does_not_convert_or_invalidate(qtbot, make_controller) -> None:
    controller, backend = make_controller()
    displayed = _ready(controller, backend)

    controller.set_config(
        _with(
            controller,
            export_formats=[ExportFormat.PNG],
            preview=True,
            label_name="SCREEN_X",
            atari_colors=[1, 2, 3, 4, 5],
            metrics=True,
        )
    )

    assert controller.latest_generation == displayed.generation
    assert controller.is_stale is False, "export-only fields cannot change the image"
    assert controller.requested_config.export_formats == [ExportFormat.PNG]
    assert backend.convert_generations == [1]


def test_a_repeated_identical_configuration_is_ignored(qtbot, make_controller) -> None:
    controller, backend = make_controller()
    _ready(controller, backend)

    controller.set_config(controller.requested_config)

    assert controller.latest_generation == 1
    assert backend.convert_generations == [1]


def test_image_key_matches_the_engine_inputs() -> None:
    base = ConversionConfig()

    assert image_key(base) != image_key(dataclasses.replace(base, background_index=3))
    assert image_key(base) == image_key(
        dataclasses.replace(base, export_formats=[ExportFormat.PNG])
    )
    assert image_key(base) == image_key(dataclasses.replace(base, preview=True))
    assert image_key(base) == image_key(dataclasses.replace(base, label_name="X"))
    assert image_key(base) == image_key(dataclasses.replace(base, atari_colors=[1, 2, 3, 4, 5]))
    assert image_key(base) == image_key(dataclasses.replace(base, metrics=True))


# --- displayed state, copies and late signals ---------------------------


def test_the_displayed_conversion_keeps_its_own_copies(qtbot, make_controller) -> None:
    controller, backend = make_controller()
    displayed = _ready(controller, backend)

    mutated = controller.requested_config
    mutated.colors = 3
    mutated.export_formats.append(ExportFormat.XEX)

    assert displayed.config.colors != 3, "the stored configuration is a copy"
    assert controller.requested_config.colors == 4
    assert displayed.config.export_formats == [ExportFormat.BIN]


def test_is_updating_and_is_stale_track_the_lifecycle(qtbot, make_controller) -> None:
    controller, backend = make_controller()
    assert controller.is_stale is True, "nothing is displayed yet"

    _load(controller, backend)

    assert controller.is_updating is True
    assert controller.status_text() == "updating… (gen 1)"

    backend.finish()

    assert controller.is_updating is False
    assert controller.is_stale is False
    assert controller.status_text() == "ready (gen 1)"

    controller.set_config(_with(controller, colors=2))

    assert controller.is_stale is True
    assert controller.status_text() == "stale - showing gen 1"


def test_signals_of_unknown_generations_are_ignored(qtbot, make_controller) -> None:
    controller, backend = make_controller()
    displayed = _ready(controller, backend)
    ready = QSignalSpy(controller.result_ready)
    failures = QSignalSpy(controller.conversion_failed)
    cancelled = QSignalSpy(controller.conversion_cancelled)

    backend.finished.emit(999, FakeResult(999))
    backend.failed.emit(998, "late")
    backend.cancelled.emit(997)

    assert ready.count() == 0
    assert failures.count() == 0
    assert cancelled.count() == 0
    assert controller.displayed is displayed
    assert controller.last_error is None


def test_only_one_conversion_runs_at_a_time(qtbot, make_controller) -> None:
    controller, backend = make_controller()
    _load(controller, backend)

    for colors in (2, 3, 4):
        controller.set_config(_with(controller, colors=colors))
        controller.convert_now()

    assert backend.convert_generations == [1], "requests never start a second parallel job"


# --- export intent ------------------------------------------------------


def test_export_runs_immediately_when_the_result_is_current(qtbot, make_controller) -> None:
    controller, backend = make_controller()
    displayed = _ready(controller, backend)
    exports = QSignalSpy(controller.export_requested)

    assert controller.request_export() is True

    assert exports.count() == 1
    assert exports.at(0)[0] is displayed
    assert controller.has_pending_export is False


def test_export_is_deferred_until_a_current_result_arrives(qtbot, make_controller) -> None:
    controller, backend = make_controller()
    _ready(controller, backend)
    exports = QSignalSpy(controller.export_requested)
    controller.set_config(_with(controller, colors=2))  # generation 2 becomes stale

    assert controller.request_export() is False
    assert controller.request_export() is False, "further requests do not queue up"
    assert controller.has_pending_export is True
    assert exports.count() == 0

    qtbot.waitUntil(lambda: len(backend.conversions) == 2, timeout=3000)
    backend.finish()

    assert exports.count() == 1
    payload = exports.at(0)[0]
    assert isinstance(payload, DisplayedConversion)
    assert payload.generation == 2
    assert controller.has_pending_export is False


def test_a_failed_conversion_does_not_export_a_stale_result(qtbot, make_controller) -> None:
    controller, backend = make_controller()
    _ready(controller, backend)
    exports = QSignalSpy(controller.export_requested)
    controller.set_config(_with(controller, colors=2))
    controller.request_export()

    qtbot.waitUntil(lambda: len(backend.conversions) == 2, timeout=3000)
    backend.fail()

    assert exports.count() == 0, "a failure never exports the outdated result"


# --- shutdown details ---------------------------------------------------


def test_a_shutdown_timeout_is_reported_and_the_backend_is_kept(qtbot, make_controller) -> None:
    controller, backend = make_controller()
    _load(controller, backend)
    backend.shutdown_result = False

    assert controller.shutdown(100) is False

    assert backend.shutdown_calls == [100]
    assert controller.backend is backend, "the backend must stay alive while its thread runs"
    assert controller.status_text() == "shutting down…"

    backend.shutdown_result = True
    assert controller.shutdown(2000) is True
    assert backend.shutdown_calls == [100, 2000]
    assert controller.status_text() == "shut down"


def test_a_result_arriving_during_shutdown_is_not_displayed(qtbot, make_controller) -> None:
    controller, backend = make_controller()
    _load(controller, backend)
    ready = QSignalSpy(controller.result_ready)

    controller.shutdown(100)
    backend.finish(1)

    assert ready.count() == 0
    assert controller.displayed is None


# --- integration with the real backend and engine -----------------------


def test_a_real_backend_produces_a_displayed_conversion(qtbot) -> None:
    """The whole chain: controller -> QThreadBackend -> ConversionEngine -> signals -> displayed."""
    backend = QThreadBackend()
    controller = ConversionController(backend, debounce_ms=10)
    try:
        with qtbot.waitSignal(controller.result_ready, timeout=60000) as loaded:
            controller.set_source(str(REFERENCE_IMAGE))

        displayed = loaded.args[0]
        assert isinstance(displayed, DisplayedConversion)
        assert isinstance(displayed.result, ConversionResult)
        assert displayed.result.indexed.shape == (192, 160)
        assert displayed.timings == displayed.result.timings
        assert displayed.result.indexed.flags.writeable is False
        assert controller.is_stale is False
        assert controller.status_text() == f"ready (gen {displayed.generation})"

        first_generation = displayed.generation
        with qtbot.waitSignal(controller.result_ready, timeout=60000) as changed:
            controller.set_config(_with(controller, colors=2))

        updated = changed.args[0]
        assert isinstance(updated, DisplayedConversion)
        assert updated.generation > first_generation
        assert updated.config.colors == 2
        assert controller.is_stale is False
    finally:
        assert controller.shutdown(15000) is True
