"""Packaging and entry-point checks for the optional GUI (T2.1).

Nothing here imports Qt at module level, so the module also runs on a checkout without the ``gui``
extra. The point is to prove that the core and the CLI keep working when PySide6 is unavailable, and
that only ``[gui]`` pulls it in.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
REFERENCE_IMAGE = Path(__file__).parent / "fixtures" / "reference.png"

_PROJECT = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

# Pretend the "gui" extra was never installed: any PySide6 import fails.
_BLOCK_PYSIDE6 = """
import sys

class _BlockPySide6:
    def find_spec(self, name, path=None, target=None):
        if name == "PySide6" or name.startswith("PySide6."):
            raise ImportError("PySide6 is not installed")
        return None

sys.meta_path.insert(0, _BlockPySide6())
"""


def _extra(name: str) -> list[str]:
    return _PROJECT["optional-dependencies"][name]


def _run(code: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter whose only ``src`` entry is this checkout."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    return subprocess.run(
        [sys.executable, "-c", code, *arguments],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_the_gui_extra_declares_pyside6() -> None:
    assert any(requirement.startswith("PySide6") for requirement in _extra("gui"))


def test_the_dev_extra_declares_pytest_qt_but_not_pyside6() -> None:
    dev = _extra("dev")
    assert any(requirement.startswith("pytest-qt") for requirement in dev)
    assert not any("pyside" in requirement.lower() for requirement in dev), "GUI must stay optional"


def test_pyside6_is_not_a_hard_dependency() -> None:
    assert not any("pyside" in requirement.lower() for requirement in _PROJECT["dependencies"])


def test_the_entry_points_are_registered() -> None:
    assert _PROJECT["gui-scripts"]["atari-convert-gui"] == "converter_gui.app:main"
    assert _PROJECT["scripts"]["atari-convert"] == "converter.main:main"


def test_the_packaging_pattern_covers_the_gui_package() -> None:
    include = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pattern = include["tool"]["setuptools"]["packages"]["find"]["include"][0]

    assert pattern == "converter*"
    assert Path("converter_gui").match(pattern)
    assert (SRC / "converter_gui" / "__init__.py").is_file()


def test_the_gui_entry_point_resolves_to_a_callable() -> None:
    pytest.importorskip("PySide6", reason="PySide6 is not installed (install the 'gui' extra)")
    module_name, _, attribute = _PROJECT["gui-scripts"]["atari-convert-gui"].partition(":")

    module = importlib.import_module(module_name)

    assert callable(getattr(module, attribute))


def test_the_cli_runs_without_pyside6(tmp_path: Path) -> None:
    """A real conversion through the CLI in an interpreter where PySide6 cannot be imported."""
    output = tmp_path / "screen.bin"
    probe = (
        _BLOCK_PYSIDE6
        + """
import json
import sys

from converter.main import main

code = main([sys.argv[1], sys.argv[2], "--dither", "none", "--palette", "median_cut"])
print(json.dumps({
    "exit": code,
    "pyside6": sorted(name for name in sys.modules if name.startswith("PySide6")),
}))
"""
    )

    completed = _run(probe, str(REFERENCE_IMAGE), str(output))

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result == {"exit": 0, "pyside6": []}
    assert output.stat().st_size == 7680, "the CLI must produce the same 160x192 ANTIC E screen"


def test_importing_the_gui_package_needs_no_pyside6() -> None:
    """``converter_gui/__init__`` stays Qt-free; only the modules inside it import PySide6."""
    probe = (
        _BLOCK_PYSIDE6
        + """
import converter_gui

print(converter_gui.__name__)
"""
    )

    completed = _run(probe)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "converter_gui"
