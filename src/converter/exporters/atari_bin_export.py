"""ANTIC mode E binary export.

ANTIC mode E (160x192, 2 bits per pixel):
- 160 pixels wide => 40 bytes per scanline (4 pixels per byte)
- 192 scanlines => 7680 bytes total
- Pixels packed MSB first: bits 7-6 = pixel 0, 5-4 = pixel 1, 3-2 = pixel 2, 1-0 = pixel 3
- Each 2-bit value is an index 0-3 into the 4-color playfield palette
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from converter.types import IndexedImage

ANTIC_E_WIDTH = 160
ANTIC_E_HEIGHT = 192
ANTIC_E_BYTES = (ANTIC_E_WIDTH // 4) * ANTIC_E_HEIGHT  # 7680


def pack_antic_e(indexed: IndexedImage, pad_4kb: bool = False) -> bytes:
    """Pack indexed image (values 0-3) into ANTIC mode E screen memory."""
    h, w = indexed.shape
    if w != ANTIC_E_WIDTH or h not in (190, 192):
        raise ValueError(f"ANTIC E requires {ANTIC_E_WIDTH}x190 or {ANTIC_E_WIDTH}x{ANTIC_E_HEIGHT}, got {w}x{h}")

    if indexed.max() > 3 or indexed.min() < 0:
        raise ValueError("Indexed values must be 0-3 for 2bpp mode")

    bytes_per_line = w // 4
    max_lines_per_segment = 4096 // bytes_per_line

    data = indexed.astype(np.uint8)
    packed = bytearray()
    remaining_lines = h
    row = 0

    while remaining_lines > 0:
        lines_in_segment = min(remaining_lines, max_lines_per_segment)
        for r in range(row, row + lines_in_segment):
            for col in range(0, w, 4):
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


def unpack_bin(data: bytes, width: int = ANTIC_E_WIDTH, height: int = ANTIC_E_HEIGHT) -> IndexedImage:
    """Unpack ANTIC mode E bytes to indexed image."""
    bytes_per_line = width // 4
    max_lines_per_segment = 4096 // bytes_per_line
    expected_unpadded = bytes_per_line * height

    if len(data) > expected_unpadded:
        unpadded = bytearray()
        remaining_lines = height
        idx = 0
        while remaining_lines > 0:
            lines_in_segment = min(remaining_lines, max_lines_per_segment)
            seg_bytes = lines_in_segment * bytes_per_line
            unpadded.extend(data[idx : idx + seg_bytes])
            idx += seg_bytes
            remaining_lines -= lines_in_segment
            if remaining_lines > 0:
                pad = 4096 - seg_bytes
                idx += pad
        data = bytes(unpadded)

    if len(data) != expected_unpadded:
        raise ValueError(f"Expected {expected_unpadded} bytes, got {len(data)}")

    arr = np.frombuffer(data, dtype=np.uint8).reshape(height, width // 4)
    indexed = np.zeros((height, width), dtype=np.uint8)
    for row in range(height):
        for col in range(0, width, 4):
            byte = arr[row, col // 4]
            indexed[row, col] = (byte >> 6) & 0x03
            indexed[row, col + 1] = (byte >> 4) & 0x03
            indexed[row, col + 2] = (byte >> 2) & 0x03
            indexed[row, col + 3] = byte & 0x03
    return indexed


def export_bin(indexed: IndexedImage, path: str | Path) -> None:
    """Write ANTIC mode E screen data to a binary file."""
    path = Path(path)
    path.write_bytes(pack_antic_e(indexed, pad_4kb=False))
