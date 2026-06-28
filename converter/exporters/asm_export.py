"""Assembly source export for screen data."""

from __future__ import annotations

from pathlib import Path

from converter.exporters.atari_bin_export import pack_antic_e
from converter.types import IndexedImage


def export_asm(
    indexed: IndexedImage,
    path: str | Path,
    label: str = "SCREEN",
    load_address: int = 0x6000,
) -> None:
    """Export screen memory as ca65-compatible assembly."""
    path = Path(path)
    screen_data = pack_antic_e(indexed)
    lines = [
        f"; Generated screen data for ANTIC mode E",
        f"; {indexed.shape[1]}x{indexed.shape[0]}, 2bpp",
        f"{label} = ${load_address:04X}",
        "",
        f"    .org ${load_address:04X}",
        f"{label}_DATA:",
    ]
    for i in range(0, len(screen_data), 16):
        chunk = screen_data[i : i + 16]
        hex_bytes = ", ".join(f"${b:02X}" for b in chunk)
        lines.append(f"    .byte {hex_bytes}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
