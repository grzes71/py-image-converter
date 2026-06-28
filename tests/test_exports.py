"""Tests for additional export formats."""

from pathlib import Path

import numpy as np

from converter.exporters.asm_export import export_asm
from converter.exporters.atari_dta_export import export_dta
from converter.exporters.xex_export import export_xex


def test_asm_export(tmp_path: Path) -> None:
    indexed = np.zeros((192, 160), dtype=np.uint8)
    path = tmp_path / "screen.s"
    export_asm(indexed, path)
    text = path.read_text(encoding="utf-8")
    assert "SCREEN_DATA" in text
    assert ".byte" in text


def test_dta_export(tmp_path: Path) -> None:
    indexed = np.zeros((192, 160), dtype=np.uint8)
    path = tmp_path / "screen.dta"
    export_dta(indexed, path)
    assert path.exists()
    assert ".byte" in path.read_text(encoding="utf-8")


def test_xex_export(tmp_path: Path) -> None:
    indexed = np.zeros((192, 160), dtype=np.uint8)
    path = tmp_path / "screen.xex"
    export_xex(indexed, path)
    data = path.read_bytes()
    assert data[0] == 0xFF
    assert data[1] == 0xFF
    assert len(data) > 7680
