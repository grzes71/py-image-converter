"""Assembly source export for screen data."""

from __future__ import annotations

from pathlib import Path

from converter.exporters.atari_bin_export import pack_antic_e
from converter.hardware_colors import GTIA_REGISTERS
from converter.types import IndexedImage


def export_asm(
    indexed: IndexedImage,
    path: str | Path,
    label: str = "SCREEN",
    load_address: int = 0x6000,
    atari_colors: list[int] | None = None,
) -> None:
    """Export screen memory as ca65-compatible assembly.

    ``atari_colors`` are the ``COLOR0..COLOR4`` shadow bytes (``COLPF0, COLPF1, COLPF2, COLPF3,
    COLBK``) and are only emitted when the caller provides them: the CLI writes this format without
    colour data by default, and that output must not change. The XEX export is the one that always
    needs the registers, because there the file itself has to produce the picture.
    """
    path = Path(path)
    screen_data = pack_antic_e(indexed)
    h, w = indexed.shape
    lines = [
        f"; Generated screen data for ANTIC mode E",
        f"; {w}x{h}, 2bpp",
        f"{label} = ${load_address:04X}",
        "",
        f"    .org ${load_address:04X}",
        f"{label}_DATA:",
    ]
    for i in range(0, len(screen_data), 16):
        chunk = screen_data[i : i + 16]
        hex_bytes = ", ".join(f"${b:02X}" for b in chunk)
        lines.append(f"    .byte {hex_bytes}")

    if atari_colors is not None:
        colors = list(atari_colors)[:5]
        while len(colors) < 5:
            colors.append(0)
        lines.append("")
        lines.append("; COLOR0-COLOR4 = " + ", ".join(GTIA_REGISTERS))
        lines.append(f"{label}_COLORS:")
        hex_colors = ", ".join(f"${c:02X}" for c in colors)
        lines.append(f"    .byte {hex_colors}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
