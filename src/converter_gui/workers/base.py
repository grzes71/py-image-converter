"""Backend seam for conversions (plan §7.3).

The future controller (T3.2) talks only to :class:`ConversionBackend`, so the QThread implementation can
be replaced by a process-based one later without touching the controller or the views. This module must
stay Qt-free: the signals are described structurally (``connect``/``emit``) instead of importing
``QObject``, ``QThread``, ``Signal`` or ``Slot``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from converter.types import ConversionConfig


@runtime_checkable
class SignalLike(Protocol):
    """What the backend's signals must look like, expressed without Qt."""

    def connect(self, slot: object) -> object:
        """Attach a receiver; returns a connection handle."""
        ...

    def emit(self, *args: object) -> None:
        """Deliver the signal to every receiver."""
        ...


@runtime_checkable
class ConversionBackend(Protocol):
    """One conversion engine driven from outside the GUI thread.

    Signal payloads (they are Qt signals in the implementation, plain values on the wire):

    * ``source_loaded(int, object)`` - generation and the :class:`~converter.types.SourceIdentity` of the
      source that was installed;
    * ``finished(int, object)`` - generation and the :class:`~converter.types.ConversionResult`;
    * ``failed(int, str)`` - generation and a readable error message;
    * ``cancelled(int)`` - generation of the operation that was cancelled.

    Every request carries the caller's generation and every reply echoes it, so a receiver can discard
    results it no longer cares about. The backend performs exactly the requests it is given: no
    debouncing, no latest-wins and no coalescing - that belongs to the controller.
    """

    source_loaded: SignalLike
    finished: SignalLike
    failed: SignalLike
    cancelled: SignalLike

    def load_source(self, gen: int, path: str, verify_content: bool = False) -> None:
        """Install ``path`` as the engine source; answers ``source_loaded`` or ``failed``."""
        ...

    def convert(self, gen: int, config: ConversionConfig) -> None:
        """Convert the installed source; answers ``finished``, ``failed`` or ``cancelled``."""
        ...

    def cancel(self, gen: int) -> None:
        """Cancel the operation of generation ``gen`` (a no-op when nothing runs for it)."""
        ...

    def shutdown(self, timeout_ms: int) -> bool:
        """Stop the worker: cancel the current work, quit its loop and wait up to ``timeout_ms``.

        Returns ``True`` when the worker really stopped. ``False`` means the timeout expired while a
        stage was still running; the caller must then keep the backend alive and may call
        ``shutdown`` again. (The plan sketches this method as returning ``None``; a timeout has to be
        reportable without pretending the thread is gone, hence the boolean.)
        """
        ...
