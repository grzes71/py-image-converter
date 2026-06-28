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


def pack_antic_e(indexed: IndexedImage) -> bytes:
    """Pack indexed image (values 0-3) into ANTIC mode E screen memory."""
    h, w = indexed.shape
    if w != ANTIC_E_WIDTH or h != ANTIC_E_HEIGHT:
        raise ValueError(f"ANTIC E requires {ANTIC_E_WIDTH}x{ANTIC_E_HEIGHT}, got {w}x{h}")

    if indexed.max() > 3 or indexed.min() < 0:
        raise ValueError("Indexed values must be 0-3 for 2bpp mode")

    data = indexed.astype(np.uint8)
    packed = np.zeros((h, w // 4), dtype=np.uint8)
    for row in range(h):
        for col in range(0, w, 4):
            p0 = int(data[row, col]) & 0x03
            p1 = int(data[row, col + 1]) & 0x03
            p2 = int(data[row, col + 2]) & 0x03
            p3 = int(data[row, col + 3]) & 0x03
            packed[row, col // 4] = (p0 << 6) | (p1 << 4) | (p2 << 2) | p3

    return packed.tobytes()


def unpack_bin(data: bytes, width: int = ANTIC_E_WIDTH, height: int = ANTIC_E_HEIGHT) -> IndexedImage:
    """Unpack ANTIC mode E bytes to indexed image."""
    expected = (width // 4) * height
    if len(data) != expected:
        raise ValueError(f"Expected {expected} bytes, got {len(data)}")

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
    path.write_bytes(pack_antic_e(indexed))
