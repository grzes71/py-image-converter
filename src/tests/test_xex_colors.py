"""The colour bytes an XEX really contains (`docs/XEX_COLOR_REGISTERS.md`).

These tests decode the init routine of the generated file instead of trusting a function call: the
segment at ``$3800`` is a fixed sequence of ``LDA #value / STA address`` writes, so the shadow bytes
can be read back exactly the way the Atari would see them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from converter.exporters.asm_export import export_asm
from converter.exporters.xex_export import export_xex
from converter.hardware_colors import (
    derived_registers,
    derived_registers_from_indices,
    resolved_registers,
)
from converter.pipeline import export_result
from converter.types import (
    ConversionConfig,
    ConversionResult,
    ExportFormat,
)

from xex_bytes import colour_registers, colour_shadows


def _result(
    width: int = 160,
    height: int = 192,
    palette_rgb: list[tuple[int, int, int]] | None = None,
) -> ConversionResult:
    colours = palette_rgb if palette_rgb is not None else [(0, 0, 0)] * 4
    indexed = np.zeros((height, width), dtype=np.uint8)
    return ConversionResult(
        indexed=indexed,
        palette_indices=list(range(len(colours))),
        palette_rgb=np.array(colours, dtype=np.uint8).reshape(len(colours), 3),
        source_resized=np.zeros((height, width, 3), dtype=np.uint8),
    )


#: A palette whose measured Atari colours are known: dark grey, white, green, purple.
KNOWN_PALETTE = [(0x32, 0x31, 0x32), (0xFF, 0xFF, 0xFF), (0x2C, 0x8E, 0x50), (0x62, 0x0E, 0x71)]
KNOWN_MAPPING = [0x00, 0x0F, 0xA4, 0x40]


def test_the_shadows_are_decoded_from_the_real_structure(tmp_path: Path) -> None:
    path = tmp_path / "screen.xex"
    export_xex(
        np.zeros((192, 160), dtype=np.uint8), path, antic_mode="E", atari_colors=[1, 2, 3, 4, 5]
    )

    assert colour_shadows(path) == {
        0x02C4: 1,
        0x02C5: 2,
        0x02C6: 3,
        0x02C7: 4,
        0x02C8: 5,
    }


@pytest.mark.parametrize("mode", ["D", "E"])
def test_manual_colours_are_written_verbatim(mode: str, tmp_path: Path) -> None:
    indexed = np.zeros((192, 160) if mode == "E" else (96, 160), dtype=np.uint8)
    path = tmp_path / f"manual_{mode}.xex"

    export_xex(indexed, path, antic_mode=mode, atari_colors=[0x00, 0x04, 0x0A, 0x0E])

    assert colour_shadows(path) == {
        0x02C4: 0x00,
        0x02C5: 0x04,
        0x02C6: 0x0A,
        0x02C7: 0x0E,
        0x02C8: 0x00,
    }


@pytest.mark.parametrize("mode", ["D", "E"])
def test_the_automatic_mapping_reaches_the_file(mode: str, tmp_path: Path) -> None:
    """Acceptance criterion: the file holds exactly what the RGB -> Atari mapping produced."""
    height = 192 if mode == "E" else 96
    result = _result(width=160, height=height, palette_rgb=KNOWN_PALETTE)
    config = ConversionConfig(export_formats=[ExportFormat.XEX])

    export_result(result, tmp_path / f"auto_{mode}.xex", config)

    shadows = colour_shadows(tmp_path / f"auto_{mode}.xex")
    expected = resolved_registers(result, config, mode)
    assert colour_registers(tmp_path / f"auto_{mode}.xex") == expected
    assert shadows[0x02C8] == KNOWN_MAPPING[0], "the background is in COLBK"
    assert shadows[0x02C4] == KNOWN_MAPPING[1], "palette colour 1 is in COLPF0"
    assert shadows[0x02C5] == KNOWN_MAPPING[2], "palette colour 2 is in COLPF1"
    assert shadows[0x02C6] == KNOWN_MAPPING[3], "palette colour 3 is in COLPF2"


def test_mode_f_writes_the_mode_f_register_pair(tmp_path: Path) -> None:
    result = _result(width=320, height=192, palette_rgb=[(0x0E, 0x0E, 0x0E), (0xB8, 0x5E, 0x60)])
    config = ConversionConfig(export_formats=[ExportFormat.XEX])

    export_result(result, tmp_path / "auto_f.xex", config)

    shadows = colour_shadows(tmp_path / "auto_f.xex")
    hue = derived_registers(result.palette_rgb, "F")
    assert shadows[0x02C6] == hue[2], "COLPF2 carries the hue and the background luminance"
    assert shadows[0x02C5] == hue[1], "COLPF1 carries the foreground luminance"
    assert shadows[0x02C6] & 0xF0 == shadows[0x02C5] & 0xF0, "both share one hue"
    assert shadows[0x02C4] == 0x00 and shadows[0x02C7] == 0x00


def test_converter_indices_are_translated_before_they_reach_the_file(tmp_path: Path) -> None:
    """The old fallback wrote ``palette_indices`` raw; ``0xEC`` is not a hardware colour."""
    path = tmp_path / "indices.xex"
    export_xex(
        np.zeros((192, 160), dtype=np.uint8),
        path,
        palette_indices=[0x00, 0xEC, 0xB2, 0xAB],
        antic_mode="E",
    )

    shadows = colour_shadows(path)
    expected = derived_registers_from_indices([0x00, 0xEC, 0xB2, 0xAB], "E")
    assert colour_registers(path) == expected
    assert shadows[0x02C4] != 0xEC, "a converter index is never written as a colour byte"


def test_an_export_without_any_colour_information_writes_zeros(tmp_path: Path) -> None:
    path = tmp_path / "empty.xex"

    export_xex(np.zeros((192, 160), dtype=np.uint8), path, antic_mode="E")

    assert set(colour_shadows(path).values()) == {0}


# --- the other formats stay as they were --------------------------------


def test_asm_omits_the_colours_unless_it_is_given_some(tmp_path: Path) -> None:
    indexed = np.zeros((192, 160), dtype=np.uint8)

    export_asm(indexed, tmp_path / "plain.s")
    assert "_COLORS" not in (tmp_path / "plain.s").read_text(encoding="utf-8")

    export_asm(indexed, tmp_path / "manual.s", atari_colors=[0x11, 0x22, 0x33, 0x44])
    text = (tmp_path / "manual.s").read_text(encoding="utf-8")
    assert "; COLOR0-COLOR4 = COLPF0, COLPF1, COLPF2, COLPF3, COLBK" in text
    assert "    .byte $11, $22, $33, $44, $00" in text


def test_bin_png_and_dta_ignore_the_registers(tmp_path: Path) -> None:
    """The data formats must stay byte-identical, manual colours or not."""
    result = _result(palette_rgb=KNOWN_PALETTE)
    formats = [ExportFormat.BIN, ExportFormat.PNG, ExportFormat.ASM, ExportFormat.DTA]
    automatic = ConversionConfig(export_formats=formats)
    manual = ConversionConfig(export_formats=formats, atari_colors=[1, 2, 3, 4, 5])

    export_result(result, tmp_path / "auto" / "screen.bin", automatic)
    export_result(result, tmp_path / "manual" / "screen.bin", manual)

    for name in ("screen.bin", "screen.png", "screen.dta"):
        assert (tmp_path / "auto" / name).read_bytes() == (tmp_path / "manual" / name).read_bytes()

    # The assembler source is the only other format that can carry the registers, and only when the
    # values were configured by hand (the CLI's default output has no colour block at all).
    assert "_COLORS" not in (tmp_path / "auto" / "screen.s").read_text(encoding="utf-8")
    assert "_COLORS" in (tmp_path / "manual" / "screen.s").read_text(encoding="utf-8")


def test_the_png_still_shows_the_converted_palette(tmp_path: Path) -> None:
    """The register mapping is export-only: the PNG keeps the converter's own colours."""
    from PIL import Image

    result = _result(
        palette_rgb=[(0, 0, 0), (255, 255, 255), (0x2C, 0x8E, 0x50), (0x62, 0x0E, 0x71)]
    )
    config = ConversionConfig(export_formats=[ExportFormat.PNG])

    export_result(result, tmp_path / "screen.png", config)

    image = Image.open(tmp_path / "screen.png").convert("RGB")
    assert image.getpixel((0, 0)) == (0, 0, 0)
