"""Tests for the hardware frame of the ``Atari result`` preview (``converter.hardware_colors``).

The preview must show what the hardware shows, so every expectation here is a colour byte turned into
the measured PAL RGB - never a converter palette colour. The consistency test pairs the frame with a
real ``.xex``: the registers the file stores and the registers the frame used have to be the same.
"""

from __future__ import annotations

import numpy as np
import pytest

from converter.atari_match import PAL_RGB
from converter.hardware_colors import (
    HardwareFrame,
    HardwareFrameRenderer,
    arrange_registers,
    derived_registers,
    pixel_colors,
    render_hardware_frame,
    resolved_registers,
)
from converter.pipeline import export_result
from converter.types import ConversionConfig, ConversionResult, ExportFormat

from xex_bytes import colour_shadows

ANTIC_D = (160, 96)
ANTIC_E = (160, 192)
ANTIC_F = (320, 192)

#: Four register values that are easy to tell apart, in COLOR0..COLOR4 order.
REGISTERS = (0x0F, 0xF0, 0x2A, 0x84, 0x00)


def _result(
    width: int = 160,
    height: int = 192,
    indexed: np.ndarray | None = None,
    palette_rgb: list[tuple[int, int, int]] | None = None,
) -> ConversionResult:
    if indexed is None:
        indexed = np.zeros((height, width), dtype=np.uint8)
    colours = palette_rgb if palette_rgb is not None else [(0, 0, 0)] * 4
    return ConversionResult(
        indexed=indexed,
        palette_indices=list(range(len(colours))),
        palette_rgb=np.array(colours, dtype=np.uint8).reshape(len(colours), 3),
        source_resized=np.zeros((indexed.shape[0], indexed.shape[1], 3), dtype=np.uint8),
    )


def _value_strip(width: int = 160, height: int = 192, values: int = 4) -> np.ndarray:
    """A bitmap where every column shows its own pixel value (0, 1, 2, 3, 0, ...)."""
    strip = np.zeros((height, width), dtype=np.uint8)
    strip[:, :] = np.arange(width, dtype=np.uint8) % values
    return strip


# --- which colour a bitmap value shows ----------------------------------


@pytest.mark.parametrize("mode", ["D", "E"])
def test_every_value_shows_the_colour_of_its_register(mode: str) -> None:
    """Value 0 -> COLBK (COLOR4), 1-3 -> COLPF0..2 (COLOR0..2) - read back through PAL_RGB."""
    indexed = _value_strip()

    frame = render_hardware_frame(indexed, REGISTERS, mode)

    expected = [
        PAL_RGB[REGISTERS[4]],  # value 0 -> COLOR4 (COLBK)
        PAL_RGB[REGISTERS[0]],  # value 1 -> COLOR0 (COLPF0)
        PAL_RGB[REGISTERS[1]],  # value 2 -> COLOR1 (COLPF1)
        PAL_RGB[REGISTERS[2]],  # value 3 -> COLOR2 (COLPF2)
    ]
    for column, colour in enumerate(expected):
        assert tuple(int(channel) for channel in frame[0, column]) == colour


@pytest.mark.parametrize("mode", ["D", "E"])
def test_the_frame_agrees_with_the_register_arrangement(mode: str) -> None:
    """Writing colours with ``arrange_registers`` and reading them back is the identity."""
    palette = [0x0A, 0x2C, 0x5E, 0x8F]
    registers = arrange_registers(palette, mode)
    indexed = _value_strip()

    frame = render_hardware_frame(indexed, registers, mode)

    for value, colour_byte in enumerate(palette):
        assert tuple(int(channel) for channel in frame[0, value]) == PAL_RGB[colour_byte]


def test_antic_f_shows_one_hue_and_two_luminances() -> None:
    background = 0x0E  # hue 0, luminance 14
    foreground = 0x3A  # hue 3, luminance 10
    registers = arrange_registers([background, foreground], "F")
    indexed = _value_strip()

    frame = render_hardware_frame(indexed, registers, "F")

    shown = [tuple(int(channel) for channel in frame[0, value]) for value in (0, 1)]
    assert shown[0] == PAL_RGB[registers[2]], "value 0: COLPF2 carries hue and background luminance"
    assert shown[1] == PAL_RGB[(registers[2] & 0xF0) | (registers[1] & 0x0F)]
    assert registers[2] >> 4 == registers[1] >> 4, "both values share the foreground hue"


def test_antic_f_has_no_independent_third_and_fourth_colour() -> None:
    base = (0x00, 0x2A, 0x3E, 0x00, 0x00)
    noisy = (0x5F, 0x2A, 0x3E, 0x9C, 0xE4)
    indexed = _value_strip()

    assert np.array_equal(
        render_hardware_frame(indexed, base, "F"), render_hardware_frame(indexed, noisy, "F")
    ), "COLOR0, COLOR3 and COLOR4 are not playfield colours in mode F"


def test_antic_f_shows_two_colours_only() -> None:
    indexed = _value_strip()

    frame = render_hardware_frame(indexed, arrange_registers([0x0E, 0x3A], "F"), "F")

    assert {tuple(int(channel) for channel in pixel) for pixel in frame[0]} == {
        PAL_RGB[0x3E],
        PAL_RGB[0x3A],
    }


@pytest.mark.parametrize(("value", "shown"), [(4, 0), (5, 1), (6, 2), (7, 3)])
def test_values_are_masked_like_the_xex_packer(value: int, shown: int) -> None:
    indexed = np.full((192, 160), value, dtype=np.uint8)

    frame = render_hardware_frame(indexed, REGISTERS, "E")

    assert (
        tuple(int(channel) for channel in frame[0, 0])
        == PAL_RGB[pixel_colors(REGISTERS, "E")[shown]]
    )


def test_antic_f_masks_to_one_bit() -> None:
    indexed = np.full((192, 320), 3, dtype=np.uint8)

    frame = render_hardware_frame(indexed, REGISTERS, "F")

    assert (
        tuple(int(channel) for channel in frame[0, 0]) == PAL_RGB[pixel_colors(REGISTERS, "F")[1]]
    )


def test_every_shown_colour_comes_from_the_pal_palette() -> None:
    frame = render_hardware_frame(_value_strip(), REGISTERS, "E")

    shown = {tuple(int(channel) for channel in pixel) for pixel in frame.reshape(-1, 3)}
    assert shown <= set(PAL_RGB)


# --- automatic and manual registers ------------------------------------


def test_manual_registers_decide_the_frame() -> None:
    result = _result(indexed=_value_strip())
    config = ConversionConfig(atari_colors=list(REGISTERS))

    frame = HardwareFrameRenderer().frame(result, config)

    assert frame is not None
    assert frame.registers == REGISTERS
    assert np.array_equal(frame.rgb, render_hardware_frame(result.indexed, REGISTERS, "E"))


def test_automatic_registers_come_from_the_converted_palette() -> None:
    palette = [(0x32, 0x31, 0x32), (0xFF, 0xFF, 0xFF), (0x2C, 0x8E, 0x50), (0x62, 0x0E, 0x71)]
    result = _result(indexed=_value_strip(), palette_rgb=palette)

    frame = HardwareFrameRenderer().frame(result, ConversionConfig())

    assert frame is not None
    assert frame.registers == tuple(resolved_registers(result, ConversionConfig(), "E"))
    assert frame.registers == tuple(derived_registers(palette, "E"))


def test_the_frame_uses_the_pal_palette_not_the_converter_colours() -> None:
    """The converter palette is only an input to the mapping; the frame must not show it."""
    palette = [(0x32, 0x31, 0x32), (0xFF, 0xFF, 0xFF), (0x2C, 0x8E, 0x50), (0x62, 0x0E, 0x71)]
    result = _result(indexed=_value_strip(), palette_rgb=palette)

    frame = HardwareFrameRenderer().frame(result, ConversionConfig())

    assert frame is not None
    expected = {PAL_RGB[byte] for byte in resolved_registers(result, ConversionConfig(), "E")}
    shown = {tuple(int(channel) for channel in pixel) for pixel in frame.rgb.reshape(-1, 3)}
    assert shown == expected, "only the mapped palette bytes are shown, nothing else"


def test_an_unknown_size_has_no_hardware_frame() -> None:
    result = _result(width=200, height=100, indexed=np.zeros((100, 200), dtype=np.uint8))

    assert HardwareFrameRenderer().frame(result, ConversionConfig()) is None


def test_the_frame_has_the_size_of_the_bitmap() -> None:
    result = _result(*ANTIC_D, indexed=np.zeros((96, 160), dtype=np.uint8))

    frame = HardwareFrameRenderer().frame(result, ConversionConfig())

    assert frame is not None
    assert frame.rgb.shape == (96, 160, 3)
    assert frame.rgb.dtype == np.uint8
    assert frame.mode == "D"


def test_the_inputs_are_not_modified() -> None:
    indexed = _value_strip()
    indexed.setflags(write=False)
    result = _result(indexed=indexed)
    config = ConversionConfig(atari_colors=list(REGISTERS))

    HardwareFrameRenderer().frame(result, config)

    assert indexed.flags.writeable is False
    assert config.atari_colors == list(REGISTERS)
    assert np.array_equal(indexed, _value_strip())


def test_the_frame_does_not_touch_the_geometry() -> None:
    """Colour interpretation and pixel aspect ratio are separate stages: only the RGB changes."""
    result = _result(indexed=_value_strip())
    with_pictures = result.source_resized

    frame = HardwareFrameRenderer().frame(result, ConversionConfig(atari_colors=list(REGISTERS)))

    assert frame is not None
    assert frame.rgb.shape == with_pictures.shape, "no scaling happens here"


# --- the exported file shows the same thing ----------------------------


@pytest.mark.parametrize(("size", "mode"), [(ANTIC_D, "D"), (ANTIC_E, "E"), (ANTIC_F, "F")])
def test_the_frame_matches_the_xex_registers(size, mode: str, tmp_path) -> None:
    """One source of truth: the file stores exactly the registers the frame was rendered from."""
    width, height = size
    depth = 2 if mode == "F" else 4
    indexed = _value_strip(width, height, values=depth)
    palette = [(0x32, 0x31, 0x32), (0xFF, 0xFF, 0xFF)]
    result = _result(width, height, indexed=indexed, palette_rgb=palette)
    config = ConversionConfig(export_formats=[ExportFormat.XEX])

    frame = HardwareFrameRenderer().frame(result, config)
    export_result(result, tmp_path / "screen.xex", config)

    assert frame is not None
    assert frame.mode == mode
    shadows = colour_shadows(tmp_path / "screen.xex")
    stored = [shadows[0x02C4 + offset] for offset in range(5)]
    assert stored == list(frame.registers)


@pytest.mark.parametrize("mode", ["D", "E", "F"])
def test_manual_registers_reach_the_file_and_the_frame(mode: str, tmp_path) -> None:
    width, height = {"D": ANTIC_D, "E": ANTIC_E, "F": ANTIC_F}[mode]
    depth = 2 if mode == "F" else 4
    result = _result(width, height, indexed=_value_strip(width, height, values=depth))
    config = ConversionConfig(
        export_formats=[ExportFormat.XEX], atari_colors=[0x11, 0x22, 0x33, 0x44, 0x55]
    )

    frame = HardwareFrameRenderer().frame(result, config, mode)
    export_result(result, tmp_path / "manual.xex", config, antic_mode=mode)

    assert frame is not None
    assert list(frame.registers) == [0x11, 0x22, 0x33, 0x44, 0x55]
    shadows = colour_shadows(tmp_path / "manual.xex")
    assert [shadows[0x02C4 + offset] for offset in range(5)] == [0x11, 0x22, 0x33, 0x44, 0x55]


# --- the cache ----------------------------------------------------------


def test_the_same_request_returns_the_same_frame() -> None:
    renderer = HardwareFrameRenderer()
    result = _result(indexed=_value_strip())
    config = ConversionConfig(atari_colors=list(REGISTERS))

    first = renderer.frame(result, config)
    second = renderer.frame(result, config)

    assert first is second, "a repeat must not render again"
    assert len(renderer) == 1


def test_a_changed_bitmap_is_rendered_again() -> None:
    renderer = HardwareFrameRenderer()
    config = ConversionConfig(atari_colors=list(REGISTERS))
    first = renderer.frame(_result(indexed=_value_strip()), config)

    other = _value_strip()
    other[0, 0] = 3
    second = renderer.frame(_result(indexed=other), config)

    assert first is not second
    assert not np.array_equal(first.rgb, second.rgb)
    assert len(renderer) == 2


def test_changed_registers_are_rendered_again() -> None:
    renderer = HardwareFrameRenderer()
    result = _result(indexed=_value_strip())

    first = renderer.frame(result, ConversionConfig(atari_colors=[0x00, 0x0F, 0x2A, 0x84, 0x00]))
    second = renderer.frame(result, ConversionConfig(atari_colors=[0x0F, 0x0F, 0x2A, 0x84, 0x00]))

    assert first is not second
    assert first.registers != second.registers
    assert not np.array_equal(first.rgb, second.rgb)


def test_a_changed_mode_is_rendered_again() -> None:
    renderer = HardwareFrameRenderer()
    result = _result(*ANTIC_E, indexed=_value_strip())
    config = ConversionConfig(atari_colors=list(REGISTERS))

    ant_e = renderer.frame(result, config, "E")
    ant_f = renderer.frame(result, config, "F")

    assert ant_e is not ant_f
    assert (ant_e.mode, ant_f.mode) == ("E", "F")
    assert len(renderer) == 2


def test_a_change_that_does_not_touch_the_colours_hits_the_cache() -> None:
    """Dither, quantizer or size settings do not change the hardware frame of the same bitmap."""
    from converter.types import DitherMethod, QuantizerMethod

    renderer = HardwareFrameRenderer()
    result = _result(indexed=_value_strip())
    registers = list(REGISTERS)

    first = renderer.frame(result, ConversionConfig(atari_colors=registers))
    second = renderer.frame(
        result,
        ConversionConfig(
            atari_colors=registers,
            dither=DitherMethod.BAYER8,
            quantizer=QuantizerMethod.WEIGHTED,
            colors=4,
            sharpen=True,
            label_name="SCREEN",
            export_formats=[ExportFormat.XEX],
        ),
    )

    assert first is second
    assert len(renderer) == 1


def test_the_cache_is_bounded() -> None:
    renderer = HardwareFrameRenderer(capacity=1)
    config = ConversionConfig(atari_colors=list(REGISTERS))

    renderer.frame(_result(indexed=_value_strip()), config)
    renderer.frame(_result(*ANTIC_D, indexed=np.zeros((96, 160), dtype=np.uint8)), config)

    assert len(renderer) == 1


def test_clearing_the_cache_forgets_the_frames() -> None:
    renderer = HardwareFrameRenderer()
    config = ConversionConfig(atari_colors=list(REGISTERS))
    first = renderer.frame(_result(indexed=_value_strip()), config)

    renderer.clear()

    assert len(renderer) == 0
    assert renderer.frame(_result(indexed=_value_strip()), config) is not first


def test_the_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="capacity"):
        HardwareFrameRenderer(capacity=0)


def test_the_frame_record_keeps_the_mode_and_the_registers() -> None:
    frame = HardwareFrameRenderer().frame(
        _result(*ANTIC_F, indexed=np.zeros((192, 320), dtype=np.uint8)),
        ConversionConfig(atari_colors=[0x0E, 0x3A, 0x00, 0x00, 0x00]),
    )

    assert isinstance(frame, HardwareFrame)
    assert frame.mode == "F"
    assert frame.registers == (0x0E, 0x3A, 0x00, 0x00, 0x00)
