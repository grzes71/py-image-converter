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
    export_xex(indexed, path, palette_indices=[0x00, 0x22, 0x44, 0x66, 0x88], antic_mode="E")
    data = path.read_bytes()
    assert data[:2] == bytes([0xFF, 0xFF])


def _parse_xex_segments(data: bytes) -> list[tuple[int, int, bytes]]:
    assert data[:2] == bytes([0xFF, 0xFF])
    i = 2
    out: list[tuple[int, int, bytes]] = []
    while i < len(data):
        start = data[i] | (data[i + 1] << 8)
        end = data[i + 2] | (data[i + 3] << 8)
        i += 4
        size = end - start + 1
        payload = data[i : i + size]
        out.append((start, end, payload))
        i += size
    return out


def test_xex_segments_and_dlist_mode_d(tmp_path: Path) -> None:
    indexed = np.zeros((96, 160), dtype=np.uint8)
    path = tmp_path / "screen_d.xex"
    export_xex(indexed, path, antic_mode="D")

    data = path.read_bytes()
    segments = _parse_xex_segments(data)
    assert len(segments) == 4

    s0_start, s0_end, s0_payload = segments[0]
    assert s0_start == 0x4000
    assert s0_end == 0x4EFF
    assert len(s0_payload) == 3840

    s1_start, _, dlist = segments[1]
    assert s1_start == 0x3E00
    assert dlist[:3] == bytes([0x70, 0x70, 0x70])
    assert dlist[3:6] == bytes([0x4D, 0x00, 0x40])
    assert dlist[6 : 6 + 95] == bytes([0x0D] * 95)
    assert dlist[-3:] == bytes([0x41, 0x00, 0x3E])

    s2_start, _, init_code = segments[2]
    assert s2_start == 0x3800
    assert init_code[:5] == bytes([0xA9, 0x22, 0x8D, 0x29, 0x02])

    s3_start, s3_end, runad = segments[3]
    assert s3_start == 0x02E0
    assert s3_end == 0x02E1
    assert runad == bytes([0x00, 0x38])


def test_xex_segments_and_dlist_mode_e(tmp_path: Path) -> None:
    indexed = np.zeros((192, 160), dtype=np.uint8)
    path = tmp_path / "screen_e.xex"
    export_xex(indexed, path, antic_mode="E")

    data = path.read_bytes()
    segments = _parse_xex_segments(data)
    assert len(segments) == 4

    s0_start, s0_end, s0_payload = segments[0]
    assert s0_start == 0x4000
    assert s0_end == 0x5DFF
    assert len(s0_payload) == 7680

    _, _, dlist = segments[1]
    assert dlist[:3] == bytes([0x70, 0x70, 0x70])
    assert dlist[3:6] == bytes([0x4E, 0x00, 0x40])
    assert dlist[6 : 6 + 95] == bytes([0x0E] * 95)
    second_lms = 6 + 95
    assert dlist[second_lms : second_lms + 3] == bytes([0x4E, 0x00, 0x4F])
    assert dlist[second_lms + 3 : second_lms + 3 + 95] == bytes([0x0E] * 95)
    assert dlist[-3:] == bytes([0x41, 0x00, 0x3E])


def test_xex_segments_and_dlist_mode_f(tmp_path: Path) -> None:
    indexed = np.zeros((192, 320), dtype=np.uint8)
    indexed[:, ::2] = 1
    path = tmp_path / "screen_f.xex"
    export_xex(indexed, path, antic_mode="F")

    data = path.read_bytes()
    segments = _parse_xex_segments(data)
    assert len(segments) == 4

    s0_start, s0_end, s0_payload = segments[0]
    assert s0_start == 0x4000
    assert s0_end == 0x5DFF
    assert len(s0_payload) == 7680

    _, _, dlist = segments[1]
    assert dlist[:3] == bytes([0x70, 0x70, 0x70])
    assert dlist[3:6] == bytes([0x4F, 0x00, 0x40])
    assert dlist[6 : 6 + 95] == bytes([0x0F] * 95)
    second_lms = 6 + 95
    assert dlist[second_lms : second_lms + 3] == bytes([0x4F, 0x00, 0x4F])
    assert dlist[second_lms + 3 : second_lms + 3 + 95] == bytes([0x0F] * 95)
    assert dlist[-3:] == bytes([0x41, 0x00, 0x3E])
