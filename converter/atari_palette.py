"""Atari 8-bit hardware color palette."""

from __future__ import annotations

import colorsys
import math

import numpy as np

from converter.types import AtariColor

# Standard Atari 8-bit GTIA/ANTIC palette RGB approximations.
# 16 hues × 16 luminance levels (index = hue * 16 + luminance).
_ATARI_RGB: list[tuple[int, int, int]] = [
    (0, 0, 0),
    (64, 64, 64),
    (128, 128, 128),
    (192, 192, 192),
    (255, 255, 255),
    (255, 0, 0),
    (255, 64, 0),
    (255, 128, 0),
    (255, 192, 0),
    (255, 255, 0),
    (192, 255, 0),
    (128, 255, 0),
    (64, 255, 0),
    (0, 255, 0),
    (0, 255, 64),
    (0, 255, 128),
    (0, 255, 192),
    (0, 255, 255),
    (0, 192, 255),
    (0, 128, 255),
    (0, 64, 255),
    (0, 0, 255),
    (64, 0, 255),
    (128, 0, 255),
    (192, 0, 255),
    (255, 0, 255),
    (255, 0, 192),
    (255, 0, 128),
    (255, 0, 64),
    (128, 64, 0),
    (192, 96, 0),
    (255, 128, 64),
    (255, 160, 128),
    (255, 192, 192),
    (128, 128, 64),
    (192, 192, 96),
    (255, 255, 128),
    (64, 128, 64),
    (96, 192, 96),
    (128, 255, 128),
    (64, 64, 128),
    (96, 96, 192),
    (128, 128, 255),
    (128, 64, 64),
    (192, 96, 96),
    (255, 128, 128),
    (64, 128, 128),
    (96, 192, 192),
    (128, 255, 255),
    (96, 64, 96),
    (128, 96, 128),
    (192, 128, 192),
    (255, 192, 255),
    (128, 96, 64),
    (192, 128, 96),
    (255, 192, 128),
    (64, 96, 64),
    (96, 128, 96),
    (128, 192, 128),
    (64, 64, 96),
    (96, 96, 128),
    (128, 128, 192),
    (96, 64, 64),
    (128, 96, 96),
    (192, 128, 128),
    (64, 96, 96),
    (96, 128, 128),
    (128, 192, 192),
    (32, 32, 32),
    (48, 48, 48),
    (80, 80, 80),
    (112, 112, 112),
    (144, 144, 144),
    (176, 176, 176),
    (208, 208, 208),
    (240, 240, 240),
    (255, 80, 80),
    (255, 112, 112),
    (255, 144, 144),
    (255, 176, 176),
    (255, 208, 208),
    (80, 255, 80),
    (112, 255, 112),
    (144, 255, 144),
    (176, 255, 176),
    (208, 255, 208),
    (80, 80, 255),
    (112, 112, 255),
    (144, 144, 255),
    (176, 176, 255),
    (208, 208, 255),
    (255, 80, 255),
    (255, 112, 255),
    (255, 144, 255),
    (255, 176, 255),
    (255, 208, 255),
    (255, 255, 80),
    (255, 255, 112),
    (255, 255, 144),
    (255, 255, 176),
    (255, 255, 208),
    (160, 160, 160),
    (200, 200, 200),
    (220, 220, 220),
    (235, 235, 235),
    (245, 245, 245),
    (250, 250, 250),
    (252, 252, 252),
    (254, 254, 254),
    (180, 100, 60),
    (200, 120, 80),
    (220, 140, 100),
    (100, 180, 60),
    (120, 200, 80),
    (140, 220, 100),
    (60, 100, 180),
    (80, 120, 200),
    (100, 140, 220),
    (180, 60, 100),
    (200, 80, 120),
    (220, 100, 140),
    (100, 60, 180),
    (120, 80, 200),
    (140, 100, 220),
    (180, 180, 60),
    (200, 200, 80),
    (220, 220, 100),
    (60, 180, 180),
    (80, 200, 200),
    (100, 220, 220),
    (180, 60, 180),
    (200, 80, 200),
    (220, 100, 220),
    (210, 170, 130),
    (170, 130, 210),
    (130, 210, 170),
    (210, 130, 170),
    (170, 210, 130),
    (130, 170, 210),
    (190, 150, 110),
    (150, 110, 190),
    (110, 190, 150),
    (190, 110, 150),
    (150, 190, 110),
    (110, 150, 190),
    (175, 135, 95),
    (135, 95, 175),
    (95, 175, 135),
    (175, 95, 135),
    (135, 175, 95),
    (95, 135, 175),
    (165, 125, 85),
    (125, 85, 165),
    (85, 165, 125),
    (165, 85, 125),
    (125, 165, 85),
    (85, 125, 165),
    (155, 115, 75),
    (115, 75, 155),
    (75, 155, 115),
    (155, 75, 115),
    (115, 155, 75),
    (75, 115, 155),
    (145, 105, 65),
    (105, 65, 145),
    (65, 145, 105),
    (145, 65, 105),
    (105, 145, 65),
    (65, 105, 145),
    (135, 95, 55),
    (95, 55, 135),
    (55, 135, 95),
    (135, 55, 95),
    (95, 135, 55),
    (55, 95, 135),
    (125, 85, 45),
    (85, 45, 125),
    (45, 125, 85),
    (125, 45, 85),
    (85, 125, 45),
    (45, 85, 125),
    (115, 75, 35),
    (75, 35, 115),
    (35, 115, 75),
    (115, 35, 75),
    (75, 115, 35),
    (35, 75, 115),
    (105, 65, 25),
    (65, 25, 105),
    (25, 105, 65),
    (105, 25, 65),
    (65, 105, 25),
    (25, 65, 105),
    (95, 55, 15),
    (55, 15, 95),
    (15, 95, 55),
    (95, 15, 55),
    (55, 95, 15),
    (15, 55, 95),
    (85, 45, 5),
    (45, 5, 85),
    (5, 85, 45),
    (85, 5, 45),
    (45, 85, 5),
    (5, 45, 85),
    (75, 35, 0),
    (35, 0, 75),
    (0, 75, 35),
    (75, 0, 35),
    (35, 75, 0),
    (0, 35, 75),
    (65, 25, 0),
    (25, 0, 65),
    (0, 65, 25),
    (65, 0, 25),
    (25, 65, 0),
    (0, 25, 65),
    (55, 15, 0),
    (15, 0, 55),
    (0, 55, 15),
    (55, 0, 15),
    (15, 55, 0),
    (0, 15, 55),
    (45, 5, 0),
    (5, 0, 45),
    (0, 45, 5),
    (45, 0, 5),
    (5, 45, 0),
    (0, 5, 45),
    (35, 0, 0),
    (0, 35, 0),
    (0, 0, 35),
    (35, 35, 0),
    (35, 0, 35),
    (0, 35, 35),
    (20, 20, 20),
    (40, 40, 40),
    (60, 60, 60),
    (100, 100, 100),
    (140, 140, 140),
    (180, 180, 180),
    (220, 220, 220),
    (255, 255, 255),
]

# Pad to 256 entries (16 hues × 16 luminance levels)
while len(_ATARI_RGB) < 256:
    last = _ATARI_RGB[-1]
    _ATARI_RGB.append(last)

_HUE_NAMES = [
    "grey",
    "gold",
    "orange",
    "red-orange",
    "pink",
    "purple",
    "purple-blue",
    "blue",
    "blue-cyan",
    "cyan",
    "cyan-green",
    "green",
    "yellow-green",
    "orange-green",
    "light-orange",
    "light-yellow",
]


def _rgb_to_hsv(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    r, g, b = (c / 255.0 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return (h * 360.0, s * 100.0, v * 100.0)


def _luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _color_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def _build_palette() -> list[AtariColor]:
    colors: list[AtariColor] = []
    for index, rgb in enumerate(_ATARI_RGB[:256]):
        hue_idx = index // 16
        lum_idx = index % 16
        name = f"{_HUE_NAMES[hue_idx % len(_HUE_NAMES)]}-{lum_idx}"
        colors.append(
            AtariColor(
                index=index,
                rgb=rgb,
                hsv=_rgb_to_hsv(rgb),
                luminance=_luminance(rgb),
                name=name,
            )
        )

    for color in colors:
        color.distances = {
            other.index: _color_distance(color.rgb, other.rgb) for other in colors
        }
    return colors


PALETTE: list[AtariColor] = _build_palette()
PALETTE_RGB: list[tuple[int, int, int]] = [c.rgb for c in PALETTE]
PALETTE_ARRAY = np.array(PALETTE_RGB, dtype=np.uint8)


def get_color(index: int) -> AtariColor:
    """Return Atari color by hardware index."""
    return PALETTE[index]


def nearest_atari_index(rgb: tuple[int, int, int]) -> int:
    """Find nearest Atari palette index for an RGB color."""
    best_idx = 0
    best_dist = float("inf")
    for color in PALETTE:
        dist = color.distances.get(color.index, _color_distance(rgb, color.rgb))
        d = _color_distance(rgb, color.rgb)
        if d < best_dist:
            best_dist = d
            best_idx = color.index
    return best_idx


def nearest_atari_indices_batch(pixels: np.ndarray) -> np.ndarray:
    """Vectorized nearest Atari index lookup for N×3 pixel array."""
    palette = PALETTE_ARRAY.astype(np.float64)
    flat = pixels.reshape(-1, 3).astype(np.float64)
    diff = flat[:, np.newaxis, :] - palette[np.newaxis, :, :]
    dist = np.sum(diff * diff, axis=2)
    return np.argmin(dist, axis=1).astype(np.uint8)
