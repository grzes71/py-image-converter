"""Tests for the public API surface of the ``converter`` package (T1.5).

Client code imports the engine, the result/config types and ``convert_image`` from ``converter`` rather
than from ``converter.engine`` / ``converter.types`` / ``converter.pipeline``.

The PySide6 isolation tests run in a **separate interpreter**. Inspecting ``sys.modules`` inside the
pytest process would prove nothing: another test (now or once the GUI tests exist) may already have
imported PySide6, while the requirement is about a fresh ``import converter``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import converter
import converter.engine
import converter.pipeline
import converter.types
from converter import ConversionConfig, ConversionEngine, ConversionResult, convert_image
from converter.types import DitherMethod, PaletteMethod

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
REFERENCE_IMAGE = Path(__file__).parent / "fixtures" / "reference.png"

PUBLIC_NAMES = ["ConversionConfig", "ConversionEngine", "ConversionResult", "convert_image"]


def test_public_names_are_importable_from_the_package() -> None:
    """Each re-exported name is the very object defined in its source module."""
    assert ConversionEngine is converter.engine.ConversionEngine
    assert ConversionConfig is converter.types.ConversionConfig
    assert ConversionResult is converter.types.ConversionResult
    assert convert_image is converter.pipeline.convert_image


def test_all_advertises_exactly_the_public_api() -> None:
    assert converter.__all__ == PUBLIC_NAMES, "sorted, so `from converter import *` is predictable"
    assert sorted(converter.__all__) == converter.__all__
    assert all(hasattr(converter, name) for name in converter.__all__)


def test_version_is_still_exposed() -> None:
    """The package metadata fallback survived the rewrite of ``__init__``."""
    assert isinstance(converter.__version__, str)
    assert converter.__version__


def test_convert_image_reexport_runs_end_to_end() -> None:
    """The re-export is usable, not just present in the namespace."""
    config = ConversionConfig(
        width=32,
        height=24,
        colors=4,
        palette=PaletteMethod.MEDIAN_CUT,
        dither=DitherMethod.NONE,
    )

    result = convert_image(REFERENCE_IMAGE, config)

    assert isinstance(result, ConversionResult)
    assert result.indexed.shape == (24, 32)
    assert len(result.palette_indices) == 4


# --- PySide6 isolation (separate interpreters) --------------------------

_PYSIDE6_FILTER = """
    "pyside6": sorted(
        name for name in sys.modules if name == "PySide6" or name.startswith("PySide6.")
    ),
"""

_API_PROBE = f"""
import json
import sys

import converter

print(json.dumps({{
{_PYSIDE6_FILTER}
    "public_api": sorted(
        name
        for name in ("ConversionConfig", "ConversionEngine", "ConversionResult", "convert_image")
        if hasattr(converter, name)
    ),
}}))
"""

_CLI_PROBE = f"""
import json
import sys

import converter
import converter.main

print(json.dumps({{
{_PYSIDE6_FILTER}
    "converter_imported": "converter" in sys.modules,
    "cli_imported": "converter.main" in sys.modules,
}}))
"""

_CONTROL_PROBE = f"""
import json
import sys

import PySide6

print(json.dumps({{
{_PYSIDE6_FILTER}
}}))
"""


def _run_probe(source: str) -> dict:
    """Run ``source`` in a fresh interpreter and return the JSON object it printed."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{SRC}{os.pathsep}{existing}" if existing else str(SRC)
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, f"probe failed:\n{completed.stderr}"
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_a_fresh_interpreter_exposes_the_public_api() -> None:
    probe = _run_probe(_API_PROBE)

    assert probe["public_api"] == sorted(PUBLIC_NAMES)


def test_importing_converter_does_not_import_pyside6() -> None:
    probe = _run_probe(_API_PROBE)

    assert probe["pyside6"] == []


def test_importing_the_cli_does_not_import_pyside6() -> None:
    probe = _run_probe(_CLI_PROBE)

    assert probe["converter_imported"] is True
    assert probe["cli_imported"] is True
    assert probe["pyside6"] == []


def test_the_isolation_probe_detects_pyside6_when_it_is_imported() -> None:
    """A probe that cannot see PySide6 would make the assertions above vacuous."""
    pytest.importorskip("PySide6", reason="PySide6 is not installed in this environment")

    probe = _run_probe(_CONTROL_PROBE)

    assert probe["pyside6"], "the probe must report PySide6 once it has really been imported"
