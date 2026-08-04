"""Atari XEX exporter for ANTIC preview viewers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from converter.exporters.atari_bin_export import pack_antic_e
from converter.types import IndexedImage

_XEX_MAGIC = (0xFF, 0xFF)

_VRAM_ADDR = 0x4000
_DLIST_ADDR = 0x3E00
_INIT_ADDR = 0x3800
_RUNAD_ADDR = 0x02E0

_MODE_MAP: dict[str, str] = {
    "D": "D",
    "E": "E",
    "F": "F",
    "7": "D",
    "15": "E",
    "8": "F",
}

_MODE_BYTES: dict[str, int] = {
    "D": 0x0D,
    "E": 0x0E,
    "F": 0x0F,
}

_LMS_BYTES: dict[str, int] = {
    "D": 0x4D,
    "E": 0x4E,
    "F": 0x4F,
}


def _pack_2bpp(indexed: IndexedImage, width: int, height: int) -> bytes:
    h, w = indexed.shape
    if (h, w) != (height, width):
        raise ValueError(f"ANTIC mode requires {width}x{height}, got {w}x{h}")
    if int(indexed.min()) < 0 or int(indexed.max()) > 3:
        raise ValueError("Indexed values must be 0-3 for 2bpp mode")

    data = indexed.astype(np.uint8)
    packed = np.zeros((height, width // 4), dtype=np.uint8)
    for row in range(height):
        for col in range(0, width, 4):
            p0 = int(data[row, col]) & 0x03
            p1 = int(data[row, col + 1]) & 0x03
            p2 = int(data[row, col + 2]) & 0x03
            p3 = int(data[row, col + 3]) & 0x03
            packed[row, col // 4] = (p0 << 6) | (p1 << 4) | (p2 << 2) | p3

    return packed.tobytes()


def _pack_1bpp(indexed: IndexedImage, width: int, height: int) -> bytes:
    h, w = indexed.shape
    if (h, w) != (height, width):
        raise ValueError(f"ANTIC mode requires {width}x{height}, got {w}x{h}")
    if int(indexed.min()) < 0 or int(indexed.max()) > 1:
        raise ValueError("Indexed values must be 0-1 for ANTIC F (1bpp)")

    data = indexed.astype(np.uint8)
    packed = np.zeros((height, width // 8), dtype=np.uint8)
    for row in range(height):
        for col in range(0, width, 8):
            byte = 0
            for bit in range(8):
                byte |= (int(data[row, col + bit]) & 0x01) << (7 - bit)
            packed[row, col // 8] = byte

    return packed.tobytes()


def _infer_antic_mode(indexed: IndexedImage) -> str:
    h, w = indexed.shape
    if (h, w) == (96, 160):
        return "D"
    if (h, w) == (192, 160):
        return "E"
    if (h, w) == (192, 320):
        return "F"
    raise ValueError(
        "Cannot infer ANTIC mode from image shape. Use --antic-mode with one of: D, E, F, 7, 15, 8."
    )


def _normalize_antic_mode(antic_mode: str | None, indexed: IndexedImage) -> str:
    if antic_mode is None:
        return _infer_antic_mode(indexed)
    normalized = _MODE_MAP.get(str(antic_mode).upper())
    if normalized is None:
        raise ValueError("Invalid ANTIC mode. Use one of: D, E, F, 7, 15, 8")
    return normalized


def _build_dlist(mode: str) -> bytearray:
    mode_byte = _MODE_BYTES[mode]
    lms_byte = _LMS_BYTES[mode]

    dlist = bytearray([0x70, 0x70, 0x70])
    dlist.extend([lms_byte, _VRAM_ADDR & 0xFF, (_VRAM_ADDR >> 8) & 0xFF])

    if mode == "D":
        dlist.extend([mode_byte] * 95)
    else:
        dlist.extend([mode_byte] * 95)
        dlist.extend([lms_byte, 0x00, 0x4F])
        dlist.extend([mode_byte] * 95)

    dlist.extend([0x41, _DLIST_ADDR & 0xFF, (_DLIST_ADDR >> 8) & 0xFF])
    return dlist


def _build_init_routine(palette_indices: list[int] | None) -> bytearray:
    palette = list(palette_indices or [])[:5]
    while len(palette) < 5:
        palette.append(0)

    code = bytearray(
        [
            0xA9,
            0x22,
            0x8D,
            0x29,
            0x02,
            0xA9,
            _DLIST_ADDR & 0xFF,
            0x8D,
            0x30,
            0x02,
            0xA9,
            (_DLIST_ADDR >> 8) & 0xFF,
            0x8D,
            0x31,
            0x02,
        ]
    )

    color_base = 0x02C4
    for i, color in enumerate(palette):
        addr = color_base + i
        code.extend([0xA9, int(color) & 0xFF, 0x8D, addr & 0xFF, (addr >> 8) & 0xFF])

    code.extend([0x4C, _INIT_ADDR & 0xFF, (_INIT_ADDR >> 8) & 0xFF])
    return code


def _append_segment(chunks: bytearray, start_addr: int, data: bytes | bytearray) -> None:
    end_addr = start_addr + len(data) - 1
    chunks.extend(
        [
            start_addr & 0xFF,
            (start_addr >> 8) & 0xFF,
            end_addr & 0xFF,
            (end_addr >> 8) & 0xFF,
        ]
    )
    chunks.extend(data)


def export_xex(
    indexed: IndexedImage,
    path: str | Path,
    palette_indices: list[int] | None = None,
    antic_mode: str | None = None,
) -> None:
    """Export an executable Atari XEX viewer for ANTIC D/E/F image preview."""
    path = Path(path)
    mode = _normalize_antic_mode(antic_mode, indexed)

    if mode == "D":
        vram_data = _pack_2bpp(indexed, width=160, height=96)
    elif mode == "E":
        # Reuse existing exporter path for ANTIC E, where possible.
        vram_data = pack_antic_e(indexed)
    else:
        vram_data = _pack_1bpp(indexed, width=320, height=192)

    dlist = _build_dlist(mode)
    init_routine = _build_init_routine(palette_indices)

    chunks = bytearray(_XEX_MAGIC)
    _append_segment(chunks, _VRAM_ADDR, vram_data)
    _append_segment(chunks, _DLIST_ADDR, dlist)
    _append_segment(chunks, _INIT_ADDR, init_routine)
    _append_segment(chunks, _RUNAD_ADDR, bytes([_INIT_ADDR & 0xFF, (_INIT_ADDR >> 8) & 0xFF]))

    path.write_bytes(chunks)
