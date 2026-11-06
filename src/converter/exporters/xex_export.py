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


def _pack_2bpp(
    indexed: IndexedImage, width: int, height: int, pad_4kb: bool = True
) -> bytes:
    h, w = indexed.shape
    if (h, w) != (height, width):
        raise ValueError(f"ANTIC mode requires {width}x{height}, got {w}x{h}")
    if int(indexed.min()) < 0 or int(indexed.max()) > 3:
        raise ValueError("Indexed values must be 0-3 for 2bpp mode")

    bytes_per_line = width // 4
    max_lines_per_segment = 4096 // bytes_per_line

    data = indexed.astype(np.uint8)
    packed = bytearray()
    remaining_lines = height
    row = 0

    while remaining_lines > 0:
        lines_in_segment = min(remaining_lines, max_lines_per_segment)
        for r in range(row, row + lines_in_segment):
            for col in range(0, width, 4):
                p0 = int(data[r, col]) & 0x03
                p1 = int(data[r, col + 1]) & 0x03
                p2 = int(data[r, col + 2]) & 0x03
                p3 = int(data[r, col + 3]) & 0x03
                packed.append((p0 << 6) | (p1 << 4) | (p2 << 2) | p3)

        row += lines_in_segment
        remaining_lines -= lines_in_segment

        if remaining_lines > 0 and pad_4kb:
            segment_bytes = lines_in_segment * bytes_per_line
            padding_bytes = 4096 - segment_bytes
            if padding_bytes > 0:
                packed.extend([0] * padding_bytes)

    return bytes(packed)


def _pack_1bpp(
    indexed: IndexedImage, width: int, height: int, pad_4kb: bool = True
) -> bytes:
    h, w = indexed.shape
    if (h, w) != (height, width):
        raise ValueError(f"ANTIC mode requires {width}x{height}, got {w}x{h}")
    if int(indexed.min()) < 0 or int(indexed.max()) > 1:
        raise ValueError("Indexed values must be 0-1 for ANTIC F (1bpp)")

    bytes_per_line = width // 8
    max_lines_per_segment = 4096 // bytes_per_line

    data = indexed.astype(np.uint8)
    packed = bytearray()
    remaining_lines = height
    row = 0

    while remaining_lines > 0:
        lines_in_segment = min(remaining_lines, max_lines_per_segment)
        for r in range(row, row + lines_in_segment):
            for col in range(0, width, 8):
                byte = 0
                for bit in range(8):
                    byte |= (int(data[r, col + bit]) & 0x01) << (7 - bit)
                packed.append(byte)

        row += lines_in_segment
        remaining_lines -= lines_in_segment

        if remaining_lines > 0 and pad_4kb:
            segment_bytes = lines_in_segment * bytes_per_line
            padding_bytes = 4096 - segment_bytes
            if padding_bytes > 0:
                packed.extend([0] * padding_bytes)

    return bytes(packed)


def _infer_antic_mode(indexed: IndexedImage) -> str:
    h, w = indexed.shape
    if (h, w) == (96, 160):
        return "D"
    if w == 160:
        return "E"
    if w == 320:
        return "F"
    raise ValueError(
        "Cannot infer ANTIC mode from image shape. Use --antic-mode with one of: D, E, F, 7, 15, 8."
    )


def normalize_antic_mode(antic_mode: str | None, indexed: IndexedImage) -> str:
    """Return the effective ANTIC mode ("D", "E" or "F") for an image.

    ``antic_mode=None`` infers the mode from the image shape; an explicit value accepts D/E/F and the
    BASIC aliases 7/15/8 (case-insensitive). Raises ``ValueError`` when the mode is invalid or cannot
    be inferred. Shared by the XEX exporter and the export-compatibility helper.
    """
    if antic_mode is None:
        return _infer_antic_mode(indexed)
    normalized = _MODE_MAP.get(str(antic_mode).upper())
    if normalized is None:
        raise ValueError("Invalid ANTIC mode. Use one of: D, E, F, 7, 15, 8")
    return normalized


def _build_dlist(
    mode: str,
    height: int = 192,
    width: int | None = None,
    screen_base_address: int = _VRAM_ADDR,
    pad_4kb: bool = True,
) -> bytearray:
    mode_byte = _MODE_BYTES[mode]
    lms_byte = _LMS_BYTES[mode]

    if width is None:
        width = 320 if mode == "F" else 160

    bits_per_pixel = 1 if mode == "F" else 2
    bytes_per_line = (width * bits_per_pixel) // 8
    max_lines_per_segment = 4096 // bytes_per_line

    dlist = bytearray([0x70, 0x70, 0x70])

    remaining_lines = height
    current_byte_offset = 0

    while remaining_lines > 0:
        lines_in_segment = min(remaining_lines, max_lines_per_segment)
        segment_bytes = lines_in_segment * bytes_per_line

        # Validation / Assertion: Verify no single segment attempts to render scanlines past a 4KB memory boundary without an LMS reload.
        assert (
            segment_bytes <= 4096
        ), f"Segment size {segment_bytes} bytes exceeds 4KB (4096 bytes) boundary limit."
        assert (
            lines_in_segment <= max_lines_per_segment
        ), f"Segment scanlines {lines_in_segment} exceeds max scanlines {max_lines_per_segment} per 4KB block."

        target_addr = screen_base_address + current_byte_offset
        dlist.extend([lms_byte, target_addr & 0xFF, (target_addr >> 8) & 0xFF])

        if lines_in_segment > 1:
            dlist.extend([mode_byte] * (lines_in_segment - 1))

        if pad_4kb and remaining_lines > lines_in_segment:
            current_byte_offset += 4096
        else:
            current_byte_offset += segment_bytes

        remaining_lines -= lines_in_segment

    dlist.extend([0x41, _DLIST_ADDR & 0xFF, (_DLIST_ADDR >> 8) & 0xFF])
    return dlist


def _build_init_routine(
    palette_indices: list[int] | None,
    atari_colors: list[int] | None = None,
) -> bytearray:
    if atari_colors is not None:
        palette = list(atari_colors)[:5]
    else:
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
    atari_colors: list[int] | None = None,
) -> None:
    """Export an executable Atari XEX viewer for ANTIC D/E/F image preview."""
    path = Path(path)
    mode = normalize_antic_mode(antic_mode, indexed)
    h, w = indexed.shape

    if mode == "D":
        vram_data = _pack_2bpp(indexed, width=w, height=h, pad_4kb=True)
    elif mode == "E":
        if (h, w) == (192, 160) or (h, w) == (190, 160):
            vram_data = pack_antic_e(indexed, pad_4kb=True)
        else:
            vram_data = _pack_2bpp(indexed, width=w, height=h, pad_4kb=True)
    else:
        vram_data = _pack_1bpp(indexed, width=w, height=h, pad_4kb=True)

    dlist = _build_dlist(mode, height=h, width=w, pad_4kb=True)
    init_routine = _build_init_routine(palette_indices, atari_colors=atari_colors)

    chunks = bytearray(_XEX_MAGIC)
    _append_segment(chunks, _VRAM_ADDR, vram_data)
    _append_segment(chunks, _DLIST_ADDR, dlist)
    _append_segment(chunks, _INIT_ADDR, init_routine)
    _append_segment(chunks, _RUNAD_ADDR, bytes([_INIT_ADDR & 0xFF, (_INIT_ADDR >> 8) & 0xFF]))

    path.write_bytes(chunks)
