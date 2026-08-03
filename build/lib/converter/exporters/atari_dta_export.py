"""Atari DTA (display list / graphics) export."""

from __future__ import annotations

from pathlib import Path

from converter.exporters.atari_bin_export import pack_antic_e
from converter.types import IndexedImage


def export_dta(indexed: IndexedImage, path: str | Path, load_address: int = 0x6000) -> None:
    """Export screen data as a DTA file with load header."""
    path = Path(path)
    screen_data = pack_antic_e(indexed)
    header = f"; Atari DTA export\n; Load address: ${load_address:04X}\n"
    header += f"; Size: {len(screen_data)} bytes\n"
    lines = [header, f"    .org ${load_address:04X}"]
    for i in range(0, len(screen_data), 16):
        chunk = screen_data[i : i + 16]
        hex_bytes = ", ".join(f"${b:02X}" for b in chunk)
        lines.append(f"    .byte {hex_bytes}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
