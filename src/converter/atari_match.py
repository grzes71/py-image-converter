"""Nearest Atari hardware colour for an RGB value - a port of the ``rgb2a8`` tool.

Two colour questions must not be confused, so they live in two modules:

* :mod:`converter.atari_palette` models the palette the *image* is quantised against. Changing it
  would change every conversion and every golden file, so it stays as it is.
* this module answers the *export* question - "which colour byte makes the Atari display this RGB
  value as closely as possible?" - and is used for the hardware colour registers only.

The algorithm is a port of ``rgb2a8.cpp`` (witcher-atari-game project, ``rgb2a8`` directory):

* :data:`PAL_RGB` is that project's measured PAL palette of all 256 GTIA colours. The index is the
  hardware colour byte itself: ``hue * 16 + luminance``, i.e. ``0x3A`` is hue 3, luminance 10.
* both the input colour and every palette entry are converted to CIELAB (D65, 2 degree observer)
  with the same sRGB gamma expansion the C++ tool uses;
* the distance is **CIE94** (graphic arts variant, ``K1 = 0.045``, ``K2 = 0.015``), which weights the
  lightness, chroma and hue differences separately. That is what makes the match consider hue *and*
  luminance: a hue error is not traded away for a slightly better brightness, and the chroma
  weighting keeps nearly-grey inputs from jumping to a saturated hue.

Two properties of the original are kept on purpose:

* ties resolve to the lowest index (the C++ loop keeps the first minimum), so the result is
  deterministic for a given table;
* nothing is clipped or masked. The luminance of the returned byte is exactly what the input maps
  to - the OS VBI's ``DRKMSK`` may still drop its lowest bit on real hardware (see
  ``docs/XEX_COLOR_REGISTERS.md`` §9), but silently pre-rounding here would make the file disagree
  with the mapping the user sees.

Because the measured table only holds fully saturated hues plus a grey ramp, two converter colours
that are only slightly different can land on the same hardware colour (and a very dark colour lands on
the darkest grey, ``$00``). That is a property of the hardware palette, not of the mapping: nothing is
invented to keep palette entries apart, and the manual registers remain the way to override it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

#: Measured PAL palette of the 256 GTIA colours, one ``RRGGBB`` value per entry, in the same
#: 16-per-row layout as ``real_pal`` in ``rgb2a8.cpp`` (a row is one hue, a column one luminance).
#: Copied without changes so the port and the original agree.
_PAL_HEX = """
    323132 3F3E3F 4D4C4D 5B5B5B 6A696A 797879 888788 979797 A1A0A1 AFAFAF BEBEBE CECDCE DBDBDB EBEAEB FAFAFA FFFFFF
    612E00 6C3B00 7A4A00 885800 94670C A5761B B2842A C1943A CA9D43 DAAD53 E8BB62 F8CB72 FFD87F FFE88F FFF79F FFFFAE
    6C2400 773000 844003 924E11 9E5D22 AF6C31 BC7B41 CC8A50 D5935B E4A369 F2B179 FFC289 FFCF97 FFDFA6 FFEDB5 FFFDC4
    751618 812324 8F3134 9D4043 AA4E50 B85E60 C66D6F D57D7F DE8787 ED9596 FCA4A5 FFB4B5 FFC2C4 FFD1D3 FFE0E1 FFEFF0
    620E71 6E1B7C 7B2A8A 8A3998 9647A5 A557B5 B365C3 C375D1 CD7EDA DC8DE9 EA97F7 F9ACFF FFBAFF FFC9FF FFD9FF FFE8FF
    560F87 611D90 712C9E 7F3AAC 8D48BA 9B58C7 A967D5 B877E5 C280ED D090FC DF9FFF EEAFFF FCBDFF FFCCFF FFDBFF FFEAFF
    461695 5122A0 6032AC 6E41BB 7C4FC8 8A5ED6 996DE3 A87CF2 B185FB C095FF CFA3FF DFB3FF EEC1FF FCD0FF FFDFFF FFEFFF
    212994 2D359F 3D44AD 4B53BA 5961C7 686FD5 777EE2 878EF2 9097FA 96A6FF AEB5FF BFC4FF CDD2FF DAE3FF EAF1FF FAFEFF
    0F3584 1C418D 2C509B 3A5EAA 486CB7 587BC5 678AD2 7699E2 80A2EB 8FB2F9 9EC0FF ADD0FF BDDDFF CBECFF DBFCFF EAFFFF
    043F70 114B79 215988 2F6896 3E75A4 4D83B2 5C92C1 6CA1D2 74ABD9 83BAE7 93C9F6 A2D8FF B1E6FF C0F5FF D0FFFF DEFFFF
    005918 006526 0F7235 1D8144 2C8E50 3B9D60 4AAC6F 59BB7E 63C487 72D396 82E2A5 92F1B5 9FFEC3 AEFFD2 BEFFE2 CEFFF1
    075C00 146800 227500 328300 3F910B 4FA01B 5EAE2A 6EBD3B 77C644 87D553 96E363 A7F373 B3FE80 C3FF8F D3FFA0 E3FFB0
    1A5600 286200 367000 457E00 538C00 629B07 70A916 80B926 89C22F 99D13E A8DF4D B7EF5C C5FC6B D5FF7B E3FF8B F3FF99
    334B00 405700 4D6500 5D7300 6A8200 7A9100 889E0F 98AE1F A1B728 BAC638 BFD548 CEE458 DCF266 EBFF75 FAFF85 FFFF95
    4B3C00 584900 655700 746500 817400 908307 9F9116 AEA126 B7AA2E C7BA3E D5C74D E5D75D F2E56B FEF47A FFFF8B FFFF9A
    602E00 6D3A00 7A4900 895800 95670A A4761B B2832A C2943A CB9D44 DAAC53 E8BA62 F8CB73 FFD77F FFE791 FFF69F FFFFAF
"""

PAL_RGB_HEX: tuple[int, ...] = tuple(int(value, 16) for value in _PAL_HEX.split())
"""The table above as integers, so a value can be compared with ``real_pal`` at a glance."""


PAL_RGB: tuple[tuple[int, int, int], ...] = tuple(
    ((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF) for value in PAL_RGB_HEX
)
"""The same table as RGB triples, indexed by the hardware colour byte."""

K1 = 0.045
"""CIE94 chroma weight of the graphic-arts variant (``rgb2a8.cpp``)."""

K2 = 0.015
"""CIE94 hue weight of the graphic-arts variant (``rgb2a8.cpp``)."""

_REF_X, _REF_Y, _REF_Z = 95.047, 100.000, 108.883  # D65, 2 degree observer
_LAB_EPSILON = 0.008856
_LAB_KAPPA = 7.787
_D65 = (
    (0.4124, 0.3576, 0.1805),
    (0.2126, 0.7152, 0.0722),
    (0.0193, 0.1192, 0.9505),
)


def srgb_to_lab(rgb: Sequence[int]) -> tuple[float, float, float]:
    """CIELAB (D65, 2 degree) of an 8-bit sRGB triple - ``RGB2LAB`` from ``rgb2a8.cpp``."""
    red, green, blue = (channel / 255.0 for channel in rgb)

    def linear(value: float) -> float:
        if value > 0.04045:
            return ((value + 0.055) / 1.055) ** 2.4
        return value / 12.92

    red, green, blue = (linear(value) * 100.0 for value in (red, green, blue))
    x = red * _D65[0][0] + green * _D65[0][1] + blue * _D65[0][2]
    y = red * _D65[1][0] + green * _D65[1][1] + blue * _D65[1][2]
    z = red * _D65[2][0] + green * _D65[2][1] + blue * _D65[2][2]

    def scale(value: float, reference: float) -> float:
        normalized = value / reference
        if normalized > _LAB_EPSILON:
            return normalized ** (1.0 / 3.0)
        return _LAB_KAPPA * normalized + 16.0 / 116.0

    var_x = scale(x, _REF_X)
    var_y = scale(y, _REF_Y)
    var_z = scale(z, _REF_Z)
    return (116.0 * var_y - 16.0, 500.0 * (var_x - var_y), 200.0 * (var_y - var_z))


def cie94_distance(first: Sequence[float], second: Sequence[float]) -> float:
    """Squared CIE94 distance between two CIELAB colours, exactly as ``rgb2a8.cpp`` computes it.

    The weighting factors use the chroma of ``first``, which is what turns the plain Euclidean Lab
    distance into a perceptual one: lightness, chroma and hue are scaled independently.
    """
    delta_l = first[0] - second[0]
    chroma_first = math.hypot(first[1], first[2])
    chroma_second = math.hypot(second[1], second[2])
    delta_c = chroma_first - chroma_second
    delta_a = first[1] - second[1]
    delta_b = first[2] - second[2]
    # A tiny negative value can appear through rounding; the C++ `sqrt` would return NaN there.
    delta_h_squared = max(delta_a * delta_a + delta_b * delta_b - delta_c * delta_c, 0.0)

    s_l = 1.0
    s_c = 1.0 + K1 * chroma_first
    s_h = 1.0 + K2 * chroma_first
    return (delta_l / s_l) ** 2 + (delta_c / s_c) ** 2 + delta_h_squared / (s_h * s_h)


_PAL_LAB = np.array([srgb_to_lab(rgb) for rgb in PAL_RGB], dtype=np.float64)
"""CIELAB of the whole palette, computed once at import."""


def hardware_rgb(index: int) -> tuple[int, int, int]:
    """The measured RGB of a hardware colour byte (``0..255``)."""
    return PAL_RGB[int(index) & 0xFF]


def _nearest_indices(labs: np.ndarray) -> np.ndarray:
    """Index of the closest palette entry for every row of ``labs`` (ties go to the lowest index)."""
    lightness = labs[:, 0:1]
    alpha = labs[:, 1:2]
    beta = labs[:, 2:3]
    chroma = np.hypot(alpha, beta)
    s_c = 1.0 + K1 * chroma
    s_h = 1.0 + K2 * chroma

    pal_lightness = _PAL_LAB[np.newaxis, :, 0]
    pal_alpha = _PAL_LAB[np.newaxis, :, 1]
    pal_beta = _PAL_LAB[np.newaxis, :, 2]
    pal_chroma = np.hypot(pal_alpha, pal_beta)

    delta_l = lightness - pal_lightness
    delta_c = chroma - pal_chroma
    delta_a = alpha - pal_alpha
    delta_b = beta - pal_beta
    delta_h_squared = np.maximum(delta_a * delta_a + delta_b * delta_b - delta_c * delta_c, 0.0)

    distance = delta_l * delta_l + (delta_c / s_c) ** 2 + delta_h_squared / (s_h * s_h)
    return np.argmin(distance, axis=1)


def nearest_hardware_colors(colors: Iterable[Sequence[int]]) -> list[int]:
    """Map every RGB triple of ``colors`` to its nearest Atari hardware colour byte."""
    labs = np.array([srgb_to_lab(tuple(color)) for color in colors], dtype=np.float64)
    if labs.size == 0:
        return []
    return [int(index) for index in _nearest_indices(labs)]


def nearest_hardware_color(rgb: Sequence[int]) -> int:
    """Map one RGB triple to the nearest Atari hardware colour byte (``hue * 16 + luminance``)."""
    return nearest_hardware_colors([rgb])[0]
