"""Tests for the ``rgb2a8`` port (``converter.atari_match``).

The port has to behave like the original tool, so the tests pin the properties that matter rather
than a few magic numbers: the measured table is the one from ``rgb2a8.cpp``, every entry maps back to
itself, the CIELAB conversion matches published values, the CIE94 weighting is the graphic-arts
variant, and the result is deterministic.
"""

from __future__ import annotations

import pytest

from converter.atari_match import (
    K1,
    K2,
    PAL_RGB,
    PAL_RGB_HEX,
    cie94_distance,
    hardware_rgb,
    nearest_hardware_color,
    nearest_hardware_colors,
    srgb_to_lab,
)


# --- the measured palette ----------------------------------------------


def test_the_table_has_one_entry_per_hardware_colour() -> None:
    assert len(PAL_RGB_HEX) == 256
    assert len(PAL_RGB) == 256
    assert all(0 <= channel <= 255 for rgb in PAL_RGB for channel in rgb)


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (0x00, (0x32, 0x31, 0x32)),  # darkest grey, hue 0 luminance 0
        (0x0F, (0xFF, 0xFF, 0xFF)),  # hue 0 luminance 15 is white
        (0x1F, (0xFF, 0xFF, 0xAE)),  # hue 1, brightest
        (0xA4, (0x2C, 0x8E, 0x50)),  # hue 10, luminance 4
        (0xFF, (0xFF, 0xFF, 0xAF)),  # last entry
    ],
)
def test_the_table_entries_match_the_source(index: int, expected: tuple[int, int, int]) -> None:
    """Spot values read straight out of ``real_pal`` in ``rgb2a8.cpp``."""
    assert hardware_rgb(index) == expected


def test_the_table_is_a_multiple_of_sixteen_hues() -> None:
    """Row 0 is the grey ramp, so every entry of it must be colour-neutral."""
    for luminance in range(16):
        red, green, blue = hardware_rgb(luminance)
        assert max(red, green, blue) - min(red, green, blue) <= 1


# --- CIELAB -------------------------------------------------------------


@pytest.mark.parametrize(
    ("rgb", "expected"),
    [
        ((0, 0, 0), (0.0, 0.0, 0.0)),
        ((255, 255, 255), (100.0, 0.0, 0.0)),
        ((255, 0, 0), (53.24, 80.09, 67.20)),
        ((0, 255, 0), (87.74, -86.18, 83.18)),
        ((0, 0, 255), (32.30, 79.19, -107.86)),
    ],
)
def test_lab_matches_the_published_values(rgb, expected) -> None:
    lab = srgb_to_lab(rgb)

    for value, reference in zip(lab, expected, strict=True):
        assert value == pytest.approx(reference, abs=0.05)


def test_grey_has_no_chroma() -> None:
    for level in (0, 40, 128, 255):
        _, alpha, beta = srgb_to_lab((level, level, level))
        assert abs(alpha) < 0.05
        assert abs(beta) < 0.05


# --- CIE94 --------------------------------------------------------------


def test_the_weighting_factors_are_the_graphic_arts_ones() -> None:
    assert K1 == 0.045
    assert K2 == 0.015


def test_identical_colours_have_zero_distance() -> None:
    lab = srgb_to_lab((10, 200, 90))

    assert cie94_distance(lab, lab) == pytest.approx(0.0)


def test_cie94_matches_the_formula_by_hand() -> None:
    """Two colours on the same lightness with equal chroma: only the hue term may contribute."""
    first = (50.0, 2.5, 0.0)
    second = (50.0, 0.0, -2.5)

    expected = 12.5 / (1.0 + K2 * 2.5) ** 2  # dH^2 / sH^2, with dL = dC = 0
    assert cie94_distance(first, second) == pytest.approx(expected)


def test_a_lightness_difference_is_not_weighted_away() -> None:
    dark = srgb_to_lab((80, 80, 80))
    light = srgb_to_lab((200, 200, 200))

    assert cie94_distance(dark, light) > cie94_distance(dark, srgb_to_lab((90, 90, 90)))


def test_a_hue_difference_costs_more_than_a_small_brightness_one() -> None:
    blue = srgb_to_lab((0, 0, 255))
    green = srgb_to_lab((0, 255, 0))
    slightly_darker_blue = srgb_to_lab((0, 0, 220))

    assert cie94_distance(blue, green) > cie94_distance(blue, slightly_darker_blue)


# --- the mapping --------------------------------------------------------


def test_every_hardware_colour_maps_to_itself() -> None:
    """The strongest property of the port: the table is its own nearest neighbour."""
    failures = [index for index, rgb in enumerate(PAL_RGB) if nearest_hardware_color(rgb) != index]

    assert failures == []


@pytest.mark.parametrize("rgb", [(0, 0, 0), (255, 255, 255), (128, 128, 128), (90, 30, 30)])
def test_the_mapping_is_deterministic(rgb: tuple[int, int, int]) -> None:
    first = nearest_hardware_color(rgb)

    assert [nearest_hardware_color(rgb) for _ in range(3)] == [first, first, first]


def test_black_and_white_stay_black_and_white() -> None:
    assert nearest_hardware_color((0, 0, 0)) == 0x00
    assert nearest_hardware_color((255, 255, 255)) == 0x0F


def test_greys_stay_greys() -> None:
    """Hue 0 is the grey ramp, so a neutral input must not jump to a coloured hue."""
    for level in (20, 60, 100, 140, 180, 220, 250):
        index = nearest_hardware_color((level, level, level))
        assert index // 16 == 0, f"grey {level} mapped to hue {index // 16}"

    assert nearest_hardware_color((20, 20, 20)) < nearest_hardware_color((250, 250, 250))


def test_a_green_stays_green_and_a_blue_stays_blue() -> None:
    green = nearest_hardware_color((44, 142, 80))
    blue = nearest_hardware_color((62, 117, 164))

    assert green == 0xA4, "the measured palette has this exact entry"
    assert blue // 16 in (8, 9, 10), "hues 8-10 are the blue rows"


def test_batch_and_single_mapping_agree() -> None:
    colours = [(0, 0, 0), (255, 255, 255), (44, 142, 80), (125, 45, 85)]

    assert nearest_hardware_colors(colours) == [nearest_hardware_color(rgb) for rgb in colours]


def test_an_empty_palette_maps_to_nothing() -> None:
    assert nearest_hardware_colors([]) == []


@pytest.mark.parametrize("rgb", [(0, 0, 0), (255, 255, 255), (1, 254, 3)])
def test_the_result_is_a_valid_colour_byte(rgb: tuple[int, int, int]) -> None:
    index = nearest_hardware_color(rgb)

    assert isinstance(index, int)
    assert 0 <= index <= 0xFF
