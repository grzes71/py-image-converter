"""Tests for the Qt-free backend protocol (T3.1, plan §7.3)."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import get_type_hints

import pytest

from converter.types import ConversionConfig
from converter_gui.workers.base import ConversionBackend, SignalLike

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"

_BLOCK_PYSIDE6 = """
import sys

class _BlockPySide6:
    def find_spec(self, name, path=None, target=None):
        if name == "PySide6" or name.startswith("PySide6."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None

sys.meta_path.insert(0, _BlockPySide6())
"""


class _Signal:
    """Minimal stand-in for a Qt signal: enough to satisfy :class:`SignalLike`."""

    def __init__(self) -> None:
        self.received: list[tuple] = []

    def connect(self, slot: object) -> object:
        return slot

    def emit(self, *args: object) -> None:
        self.received.append(args)


class _DuckBackend:
    """A backend written with plain Python - no Qt anywhere."""

    def __init__(self) -> None:
        self.source_loaded = _Signal()
        self.finished = _Signal()
        self.failed = _Signal()
        self.cancelled = _Signal()
        self.requests: list[tuple] = []

    def load_source(self, gen: int, path: str, verify_content: bool = False) -> None:
        self.requests.append(("load", gen, path, verify_content))

    def convert(self, gen: int, config: ConversionConfig) -> None:
        self.requests.append(("convert", gen, config))

    def cancel(self, gen: int) -> None:
        self.requests.append(("cancel", gen))

    def shutdown(self, timeout_ms: int) -> bool:
        self.requests.append(("shutdown", timeout_ms))
        return True


class _BrokenBackend:
    """Missing ``cancel`` on purpose."""

    source_loaded = _Signal()
    finished = _Signal()
    failed = _Signal()
    cancelled = _Signal()

    def load_source(self, gen: int, path: str, verify_content: bool = False) -> None: ...

    def convert(self, gen: int, config: ConversionConfig) -> None: ...

    def shutdown(self, timeout_ms: int) -> bool:
        return True


def test_the_protocol_module_does_not_import_qt() -> None:
    probe = (
        _BLOCK_PYSIDE6
        + """
import json
import sys

from converter_gui.workers.base import ConversionBackend

print(json.dumps({
    "protocol": ConversionBackend.__name__,
    "pyside6": sorted(name for name in sys.modules if name.startswith("PySide6")),
}))
"""
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip()) == {"protocol": "ConversionBackend", "pyside6": []}


def test_the_protocol_declares_the_documented_signals() -> None:
    annotations = ConversionBackend.__annotations__

    assert set(annotations) == {"source_loaded", "finished", "failed", "cancelled"}
    # ``from __future__ import annotations`` keeps the annotations as strings.
    assert {str(annotation) for annotation in annotations.values()} == {"SignalLike"}


def test_the_signal_shim_only_requires_connect_and_emit() -> None:
    signal = _Signal()

    assert isinstance(signal, SignalLike)
    signal.emit(1, "payload")
    assert signal.received == [(1, "payload")]


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("load_source", ["self", "gen", "path", "verify_content"]),
        ("convert", ["self", "gen", "config"]),
        ("cancel", ["self", "gen"]),
        ("shutdown", ["self", "timeout_ms"]),
    ],
)
def test_the_method_signatures_match_the_contract(method: str, expected: list[str]) -> None:
    function = getattr(ConversionBackend, method)

    assert list(inspect.signature(function).parameters) == expected
    if "gen" in expected:
        assert get_type_hints(function)["gen"] is int


def test_verify_content_is_optional() -> None:
    signature = inspect.signature(ConversionBackend.load_source)

    assert signature.parameters["verify_content"].default is False


def test_shutdown_reports_whether_the_thread_stopped() -> None:
    """A timeout must be reportable, so the contract returns a boolean rather than ``None``."""
    assert get_type_hints(ConversionBackend.shutdown)["return"] is bool


def test_a_plain_python_backend_satisfies_the_protocol() -> None:
    backend = _DuckBackend()

    assert isinstance(backend, ConversionBackend) is True
    backend.convert(7, ConversionConfig())
    assert backend.requests == [("convert", 7, backend.requests[-1][2])]


def test_a_backend_without_cancel_does_not_satisfy_the_protocol() -> None:
    assert isinstance(_BrokenBackend(), ConversionBackend) is False
