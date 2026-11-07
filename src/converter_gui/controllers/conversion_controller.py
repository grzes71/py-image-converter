"""Conversion controller (T3.2, plan §7.1, §7.5 and §7.6).

The controller is the single owner of the conversion state: the generation counter, the debounce timer,
the queue of at most one pending request, the displayed result and the pending export intent. It talks to
the backend only through :class:`~converter_gui.workers.base.ConversionBackend`, so the QThread
implementation (or a future process backend) stays an implementation detail, and it never calls the engine
itself.

Two invariants drive the whole design:

* a result of an older generation can never be shown as current - results are accepted only when their
  generation is the newest one, and ``is_stale`` is a generation comparison, never a comparison of
  configurations;
* at most one conversion runs and at most one waits; every newer request replaces the waiting one, so
  five quick parameter edits produce a single job with the newest configuration.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

from converter.types import ConversionConfig, ConversionResult, SourceIdentity
from converter_gui.workers.base import ConversionBackend

LOGGER = logging.getLogger(__name__)

DEBOUNCE_INTERVAL_MS = 300
DEFAULT_SHUTDOWN_TIMEOUT_MS = 5000

#: Fields that only affect the written output. The engine's cache keys (L1-L3) do not contain them, so
#: changing one must not schedule a conversion (plan §7.5, rule 4: the display is not stale then).
EXPORT_ONLY_FIELDS = ("export_formats", "preview", "label_name", "atari_colors")

#: The image-relevant fields, i.e. the engine's L1-L3 keys expressed as configuration fields. ``metrics``
#: is excluded as well: it only toggles whether metrics are computed and the backend always asks for them.
IMAGE_KEY_FIELDS = (
    "mode",
    "width",
    "height",
    "resize",
    "aspect",
    "pixel_aspect",
    "sharpen",
    "sharpen_strength",
    "palette",
    "colors",
    "background_index",
    "dither",
    "quantizer",
)


def image_key(config: ConversionConfig) -> tuple:
    """The part of a configuration that can change the converted image."""
    return tuple(getattr(config, name) for name in IMAGE_KEY_FIELDS)


@dataclass(frozen=True)
class DisplayedConversion:
    """The result the views show, with the configuration that really produced it (plan §7.5)."""

    generation: int
    config: ConversionConfig
    result: ConversionResult
    timings: dict[str, float]
    produced_at: float


class ConversionController(QObject):
    """Drives one :class:`ConversionBackend` on behalf of the GUI.

    Signals for the window: ``result_ready(DisplayedConversion)``, ``source_loaded(int, SourceIdentity)``,
    ``source_failed(int, str)``, ``conversion_failed(int, str)``, ``conversion_cancelled(int)``,
    ``export_requested(DisplayedConversion)`` and ``state_changed()`` (emitted whenever the derived state
    - updating/stale/ready/error - may have changed, for a status bar or a badge).
    """

    result_ready = Signal(object)
    source_loaded = Signal(int, object)
    source_failed = Signal(int, str)
    conversion_failed = Signal(int, str)
    conversion_cancelled = Signal(int)
    export_requested = Signal(object)
    state_changed = Signal()

    def __init__(
        self,
        backend: ConversionBackend,
        *,
        debounce_ms: int = DEBOUNCE_INTERVAL_MS,
        auto_update: bool = True,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._backend = backend
        self._requested_config = ConversionConfig()
        self._last_image_key = image_key(self._requested_config)
        self._latest_generation = 0
        self._displayed: DisplayedConversion | None = None
        self._running_generation: int | None = None
        self._running_config: ConversionConfig | None = None
        self._cancel_sent_for: int | None = None
        self._pending: tuple[int, ConversionConfig] | None = None
        self._load_generation: int | None = None
        self._source_identity: SourceIdentity | None = None
        self._last_error: str | None = None
        self._auto_update = bool(auto_update)
        self._pending_export = False
        self._shutting_down = False
        self._shutdown_result: bool | None = None

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(max(0, int(debounce_ms)))
        self._debounce.timeout.connect(self._on_debounce_timeout)

        backend.source_loaded.connect(self._on_source_loaded)
        backend.finished.connect(self._on_finished)
        backend.failed.connect(self._on_failed)
        backend.cancelled.connect(self._on_cancelled)

    # --- state for the window -------------------------------------------

    @property
    def backend(self) -> ConversionBackend:
        """The injected backend (kept alive by the controller, also after a shutdown timeout)."""
        return self._backend

    @property
    def requested_config(self) -> ConversionConfig:
        """The newest configuration asked for by the user (a copy, so callers cannot mutate it)."""
        return self._copy_config(self._requested_config)

    @property
    def latest_generation(self) -> int:
        """Generation of the newest source or image-relevant configuration change."""
        return self._latest_generation

    @property
    def displayed(self) -> DisplayedConversion | None:
        """The last successful conversion, with the configuration that produced it."""
        return self._displayed

    @property
    def running_generation(self) -> int | None:
        """Generation currently being converted (``None`` while idle)."""
        return self._running_generation

    @property
    def last_error(self) -> str | None:
        """Message of the most recent failed conversion or source load."""
        return self._last_error

    @property
    def source_identity(self) -> SourceIdentity | None:
        """Identity of the source the backend reported as loaded."""
        return self._source_identity

    @property
    def auto_update(self) -> bool:
        """Whether parameter changes schedule a conversion automatically."""
        return self._auto_update

    @property
    def is_updating(self) -> bool:
        """True while a conversion is running."""
        return self._running_generation is not None

    @property
    def is_stale(self) -> bool:
        """True when the displayed result does not belong to the newest generation."""
        return self._displayed is None or self._displayed.generation != self._latest_generation

    @property
    def has_pending_export(self) -> bool:
        """True when an export intent is waiting for a fresh result."""
        return self._pending_export

    def status_text(self) -> str:
        """One-line state for a status bar or a badge (plan §7.5 and §7.6)."""
        if self._shutdown_result is True:
            return "shut down"
        if self._shutting_down:
            return "shutting down…"
        if self.is_updating:
            return f"updating… (gen {self._running_generation})"
        if self._last_error is not None:
            if self._displayed is None:
                return "error - no result yet"
            return f"error - still showing gen {self._displayed.generation}"
        if self._displayed is None:
            return "no result yet"
        if self.is_stale:
            return f"stale - showing gen {self._displayed.generation}"
        return f"ready (gen {self._displayed.generation})"

    # --- requests from the window ---------------------------------------

    def set_config(self, config: ConversionConfig) -> None:
        """Accept a parameter change: bump the generation and debounce a conversion.

        A change that only touches export-only fields is remembered for the export path but never
        schedules work: it cannot change the image and the engine's cache keys do not see it either.
        """
        if self._shutting_down:
            return
        new_key = image_key(config)
        self._requested_config = self._copy_config(config)
        if new_key == self._last_image_key:
            self._emit_state()
            return
        self._last_image_key = new_key
        self._latest_generation += 1
        self._cancel_outdated_running()
        if self._auto_update:
            self._schedule()
        self._emit_state()

    def set_source(self, path: str | Path, *, verify_content: bool = False) -> None:
        """Ask the backend to load a new source; conversions wait until it has arrived."""
        if self._shutting_down:
            return
        self._latest_generation += 1
        generation = self._latest_generation
        self._pending = None
        self._debounce.stop()
        self._load_generation = generation
        self._cancel_outdated_running()
        self._backend.load_source(generation, str(path), verify_content)
        self._emit_state()

    def convert_now(self) -> None:
        """Convert the current configuration immediately, without waiting for the debounce."""
        if self._shutting_down or self._source_identity is None:
            return
        self._debounce.stop()
        if self._running_generation == self._latest_generation:
            return  # that generation is already being converted
        if self._displayed is not None and self._displayed.generation == self._latest_generation:
            return  # the shown result already belongs to the current state
        self._pending = (self._latest_generation, self._copy_config(self._requested_config))
        self._start_pending()

    def set_auto_update(self, enabled: bool) -> None:
        """Turn the automatic conversion after a parameter change on or off."""
        enabled = bool(enabled)
        if enabled == self._auto_update:
            return
        self._auto_update = enabled
        if not enabled:
            self._debounce.stop()
            self._pending = None
        elif not self._shutting_down:
            self._schedule()
        self._emit_state()

    def request_export(self) -> bool:
        """Ask for an export of the displayed result.

        Returns ``True`` when the caller may export immediately; otherwise a single intent is remembered
        and ``export_requested`` fires as soon as a current result arrives. Further requests do not queue
        up.
        """
        if self._shutting_down:
            return False
        if self._displayed is not None and not self.is_stale and not self.is_updating:
            self.export_requested.emit(self._displayed)
            return True
        self._pending_export = True
        return False

    def shutdown(self, timeout_ms: int = DEFAULT_SHUTDOWN_TIMEOUT_MS) -> bool:
        """Stop the pipeline and the backend; returns whether the backend really stopped.

        After a timeout the controller keeps its backend reference (nothing is destroyed under a running
        thread), stops accepting requests and reports ``False``, so the caller may try again. A repeated
        call after a successful stop is a no-op.
        """
        if self._shutdown_result is True:
            return True
        self._shutting_down = True
        self._debounce.stop()
        self._pending = None
        self._pending_export = False
        self._latest_generation += 1  # nothing in flight can become current anymore
        if self._running_generation is not None:
            self._backend.cancel(self._running_generation)
        self._running_generation = None
        self._running_config = None
        self._cancel_sent_for = None
        self._emit_state()
        stopped = bool(self._backend.shutdown(timeout_ms))
        self._shutdown_result = stopped
        if not stopped:
            LOGGER.warning("the conversion backend did not stop within %d ms", timeout_ms)
        self._emit_state()
        return stopped

    # --- scheduling -----------------------------------------------------

    def _schedule(self) -> None:
        """Remember the newest request and (re)start the single debounce timer."""
        if self._shutting_down or self._source_identity is None:
            return
        self._pending = (self._latest_generation, self._copy_config(self._requested_config))
        self._debounce.start()

    def _on_debounce_timeout(self) -> None:
        if self._shutting_down:
            return
        self._start_pending()

    def _start_pending(self) -> None:
        """Dispatch the waiting request when nothing else runs and the source is ready."""
        if self._shutting_down or self._pending is None:
            return
        if self._running_generation is not None or self._load_generation is not None:
            return  # one job at a time: the answer of the current one comes first
        generation, config = self._pending
        self._pending = None
        if generation != self._latest_generation or self._source_identity is None:
            return  # superseded while waiting
        self._debounce.stop()
        self._running_generation = generation
        self._running_config = config
        self._cancel_sent_for = None
        self._backend.convert(generation, config)
        self._emit_state()

    def _cancel_outdated_running(self) -> None:
        """Ask the backend once to stop a conversion that can no longer be current."""
        generation = self._running_generation
        if generation is None or generation == self._latest_generation:
            return
        if self._cancel_sent_for == generation:
            return  # already asked; the backend is busy until it answers
        self._cancel_sent_for = generation
        self._backend.cancel(generation)

    def _emit_state(self) -> None:
        self.state_changed.emit()

    @staticmethod
    def _copy_config(config: ConversionConfig) -> ConversionConfig:
        """A copy with its own list fields, so later edits cannot reach a stored configuration."""
        return dataclasses.replace(
            config,
            export_formats=list(config.export_formats),
            atari_colors=list(config.atari_colors) if config.atari_colors is not None else None,
        )

    # --- backend signals ------------------------------------------------

    def _on_source_loaded(self, generation: int, identity: object) -> None:
        if self._shutting_down:
            return
        if self._load_generation is not None and generation < self._load_generation:
            LOGGER.debug("ignoring the source of the superseded generation %d", generation)
            return
        self._load_generation = None
        self._source_identity = identity if isinstance(identity, SourceIdentity) else None
        self.source_loaded.emit(generation, identity)
        if self._auto_update:
            self._pending = (self._latest_generation, self._copy_config(self._requested_config))
        self._start_pending()
        self._emit_state()

    def _on_finished(self, generation: int, result: ConversionResult) -> None:
        if self._shutting_down:
            return
        if self._running_generation is None or generation != self._running_generation:
            LOGGER.debug("ignoring the result of the unknown generation %d", generation)
            return
        config = (
            self._running_config if self._running_config is not None else self._requested_config
        )
        self._running_generation = None
        self._running_config = None
        self._cancel_sent_for = None
        if generation != self._latest_generation:
            LOGGER.debug("dropping the result of the outdated generation %d", generation)
            self._start_pending()
            self._emit_state()
            return
        self._displayed = DisplayedConversion(
            generation=generation,
            config=self._copy_config(config),
            result=result,
            timings=dict(result.timings),
            produced_at=time.perf_counter(),
        )
        self._last_error = None
        self.result_ready.emit(self._displayed)
        if self._pending_export:
            self._pending_export = False
            self.export_requested.emit(self._displayed)
        self._start_pending()
        self._emit_state()

    def _on_failed(self, generation: int, message: str) -> None:
        if self._shutting_down:
            return
        if self._load_generation is not None and generation == self._load_generation:
            self._load_generation = None
            self._last_error = message
            self.source_failed.emit(generation, message)
            self._emit_state()
            return
        if self._running_generation is not None and generation == self._running_generation:
            self._running_generation = None
            self._running_config = None
            self._cancel_sent_for = None
            self._last_error = message
            self.conversion_failed.emit(generation, message)
            self._start_pending()
            self._emit_state()
            return
        LOGGER.debug("ignoring the failure of the unknown generation %d", generation)

    def _on_cancelled(self, generation: int) -> None:
        if self._shutting_down:
            return
        if self._running_generation is None or generation != self._running_generation:
            LOGGER.debug("ignoring the cancellation of the unknown generation %d", generation)
            return
        self._running_generation = None
        self._running_config = None
        self._cancel_sent_for = None
        self.conversion_cancelled.emit(generation)
        self._start_pending()
        self._emit_state()
