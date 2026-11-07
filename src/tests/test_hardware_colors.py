"""Tests for the hardware colour registers (`converter.hardware_colors`).

The register *order* is the part that used to be wrong (``docs/XEX_COLOR_REGISTERS.md`` §8.1), so the
arrangement is pinned to explicit numbers per mode instead of "the list was returned".
"""

from __future__ import annotations

import numpy as np
import pytest

from converter.atari_match import nearest_hardware_color
from converter.hardware_colors import (
    GTIA_REGISTERS,
    REGISTER_COUNT,
    REGISTER_NAMES,
    arrange_registers,
    derived_registers,
    derived_registers_from_indices,
    mode_for_size,
    pad_registers,
    resolved_registers,
    resolve_antic_mode,
)
from converter.types import ConversionConfig, ConversionResult


def _result(
    width: int = 160,
    height: int = 192,
    palette_rgb: list[tuple[int, int, int]] | None = None,
    palette_indices: list[int] | None = None,
) -> ConversionResult:
    colours = palette_rgb if palette_rgb is not None else [(0, 0, 0)] * 4
    return ConversionResult(
        indexed=np.zeros((height, width), dtype=np.uint8),
        palette_indices=palette_indices
        if palette_indices is not None
        else list(range(len(colours))),
        palette_rgb=np.array(colours, dtype=np.uint8).reshape(len(colours), 3),
        source_resized=np.zeros((height, width, 3), dtype=np.uint8),
    )


# --- the register table -------------------------------------------------


def test_the_register_names_follow_the_shadow_order() -> None:
    assert REGISTER_COUNT == 5
    assert REGISTER_NAMES == ("COLOR0", "COLOR1", "COLOR2", "COLOR3", "COLOR4")
    assert GTIA_REGISTERS == ("COLPF0", "COLPF1", "COLPF2", "COLPF3", "COLBK")


# --- mode resolution ----------------------------------------------------


@pytest.mark.parametrize(
    ("width", "height", "mode"),
    [(160, 96, "D"), (160, 192, "E"), (160, 190, "E"), (320, 192, "F")],
)
def test_the_mode_follows_the_screen_size(width: int, height: int, mode: str) -> None:
    assert mode_for_size(width, height) == mode


@pytest.mark.parametrize(("width", "height"), [(200, 100), (100, 200), (640, 480)])
def test_an_unknown_size_cannot_be_exported(width: int, height: int) -> None:
    with pytest.raises(ValueError, match="Cannot infer ANTIC mode"):
        mode_for_size(width, height)


@pytest.mark.parametrize(
    ("given", "expected"),
    [("d", "D"), ("E", "E"), ("f", "F"), ("7", "D"), ("15", "E"), ("8", "F")],
)
def test_the_aliases_resolve_to_the_letters(given: str, expected: str) -> None:
    assert resolve_antic_mode(given, 160, 192) == expected


def test_an_invalid_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid ANTIC mode"):
        resolve_antic_mode("X", 160, 192)


def test_no_mode_falls_back_to_the_size() -> None:
    assert resolve_antic_mode(None, 320, 192) == "F"


# --- padding ------------------------------------------------------------


def test_padding_fills_up_and_truncates() -> None:
    assert pad_registers([1, 2]) == [1, 2, 0, 0, 0]
    assert pad_registers([]) == [0, 0, 0, 0, 0]
    assert pad_registers([1, 2, 3, 4, 5, 6, 7]) == [1, 2, 3, 4, 5]
    assert pad_registers([0x1FF]) == [0xFF, 0, 0, 0, 0], "values are masked to one byte"


# --- the arrangement (the B5 fix) --------------------------------------


@pytest.mark.parametrize("mode", ["D", "E"])
def test_d_and_e_put_the_background_in_colbk(mode: str) -> None:
    registers = arrange_registers([0x11, 0x22, 0x33, 0x44], mode)

    assert registers == [0x22, 0x33, 0x44, 0x00, 0x11]
    assert registers[4] == 0x11, "COLOR4 (COLBK) holds palette colour 0"
    assert registers[0] == 0x22, "COLOR0 (COLPF0) holds palette colour 1"
    assert registers[3] == 0x00, "COLPF3 is not a playfield colour in D/E"


@pytest.mark.parametrize("mode", ["D", "E"])
def test_a_two_colour_palette_leaves_the_unused_registers_empty(mode: str) -> None:
    assert arrange_registers([0x11, 0x22], mode) == [0x22, 0x00, 0x00, 0x00, 0x11]


def test_antic_f_builds_one_hue_and_two_luminances() -> None:
    background = 0x0E  # hue 0, luminance 14
    foreground = 0x3A  # hue 3, luminance 10

    registers = arrange_registers([background, foreground], "F")

    assert registers[2] == 0x3E, "COLOR2 (COLPF2): foreground hue + background luminance"
    assert registers[1] == 0x3A, "COLOR1 (COLPF1): foreground luminance"
    assert registers[4] == 0x0E, "COLOR4 (COLBK): the border stays the background colour"
    assert registers[0] == 0x00 and registers[3] == 0x00


def test_antic_f_cannot_invent_a_second_colour() -> None:
    """A one-colour palette gets luminance 0 for the foreground - never a made-up colour."""
    registers = arrange_registers([0x2E], "F")

    assert registers[1] == 0x00
    assert registers[2] & 0xF0 == 0x00, "no foreground colour, so no hue"
    assert registers[2] & 0x0F == 0x0E, "the background luminance survives"


# --- derived registers --------------------------------------------------


def test_derived_registers_map_the_palette_through_rgb2a8() -> None:
    palette = [(0x32, 0x31, 0x32), (0xFF, 0xFF, 0xFF), (0x2C, 0x8E, 0x50), (0x62, 0x0E, 0x71)]

    registers = derived_registers(palette, "E")

    mapped = [nearest_hardware_color(rgb) for rgb in palette]
    assert mapped == [0x00, 0x0F, 0xA4, 0x40]
    assert registers == [0x0F, 0xA4, 0x40, 0x00, 0x00]


def test_derived_registers_respect_the_mode_capacity() -> None:
    palette = [(0x32, 0x31, 0x32), (0xFF, 0xFF, 0xFF), (0xB8, 0x5E, 0x60), (0x2C, 0x8E, 0x50)]

    assert derived_registers(palette, "F")[1] != 0x00, "mode F keeps the foreground luminance"
    assert derived_registers(palette, "F")[3] == 0x00, "mode F has two colours"
    assert derived_registers(palette, "E")[3] == 0x00, "COLPF3 is unused in D/E"


def test_converter_indices_are_translated_through_their_rgb() -> None:
    """``0xEC`` is a converter-table position, not a hardware colour: it must not reach a register."""
    from converter import atari_palette

    indices = [0x00, 0xEC, 0xB2, 0xAB]
    registers = derived_registers_from_indices(indices, "E")

    expected = derived_registers([atari_palette.get_color(i).rgb for i in indices], "E")
    assert registers == expected
    assert registers[4] == nearest_hardware_color(atari_palette.get_color(0x00).rgb)


# --- resolved registers -------------------------------------------------


def test_manual_values_win_verbatim() -> None:
    result = _result(palette_rgb=[(0, 0, 0), (255, 255, 255)])
    config = ConversionConfig(atari_colors=[0x00, 0x04, 0x0A, 0x0E])

    assert resolved_registers(result, config) == [0x00, 0x04, 0x0A, 0x0E, 0x00]


@pytest.mark.parametrize("mode", ["D", "E", "F"])
def test_manual_values_are_not_rearranged(mode: str) -> None:
    """A user who typed COLOR0..COLOR4 gets exactly those bytes, whatever the mode is."""
    result = _result(palette_rgb=[(0, 0, 0), (255, 255, 255)])
    config = ConversionConfig(atari_colors=[1, 2, 3, 4, 5])

    assert resolved_registers(result, config, mode) == [1, 2, 3, 4, 5]


def test_without_manual_values_the_palette_is_mapped() -> None:
    result = _result(palette_rgb=[(0x32, 0x31, 0x32), (0xFF, 0xFF, 0xFF)])

    assert resolved_registers(result, ConversionConfig()) == derived_registers(
        result.palette_rgb, "E"
    )


def test_the_mode_comes_from_the_result_when_it_is_not_given() -> None:
    result = _result(width=320, height=192, palette_rgb=[(0x32, 0x31, 0x32), (0xFF, 0xFF, 0xFF)])
    config = ConversionConfig()

    assert resolved_registers(result, config) == resolved_registers(result, config, "F")
    assert resolved_registers(result, config, "F") != resolved_registers(result, config, "E")


def test_an_unknown_shape_without_manual_values_raises() -> None:
    result = _result(width=200, height=100, palette_rgb=[(0, 0, 0)])

    with pytest.raises(ValueError, match="Cannot infer ANTIC mode"):
        resolved_registers(result, ConversionConfig())


def test_an_unknown_shape_with_manual_values_is_fine() -> None:
    """Manual registers do not need a mode: they are written as they are (the format check is
    separate and refuses the export elsewhere)."""
    result = _result(width=200, height=100, palette_rgb=[(0, 0, 0)])

    assert resolved_registers(result, ConversionConfig(atari_colors=[7, 8, 9])) == [7, 8, 9, 0, 0]
