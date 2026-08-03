"""Minimal XEX export wrapper."""

from __future__ import annotations

from pathlib import Path

from converter.exporters.atari_bin_export import pack_antic_e
from converter.types import IndexedImage

# Minimal executable header: load address + data + run address stub
_DEFAULT_LOAD = 0x0800


def export_xex(
    indexed: IndexedImage,
    path: str | Path,
    load_address: int = _DEFAULT_LOAD,
) -> None:
    """Export screen data wrapped in a minimal Atari XEX container.

    Format: $FFFF header words followed by load address, length, data.
    This is a minimal stub suitable for loading screen data via DOS loaders.
    """
    path = Path(path)
    screen_data = pack_antic_e(indexed)
    data_len = len(screen_data)

    chunks: list[int] = []
    # XEX header: start/end markers
    chunks.extend([0xFF, 0xFF])
    # Load address
    chunks.extend([load_address & 0xFF, (load_address >> 8) & 0xFF])
    # End address (exclusive)
    end_addr = load_address + data_len
    chunks.extend([end_addr & 0xFF, (end_addr >> 8) & 0xFF])
    chunks.extend(screen_data)
    chunks.extend([0xFF, 0xFF])  # End of file marker

    path.write_bytes(bytes(chunks))
