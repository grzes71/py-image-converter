"""Hardware colour registers of the exported viewer (``COLOR0`` .. ``COLOR4``).

The XEX init routine pokes five bytes into ``$02C4..$02C8``, the OS shadows of the GTIA colour
registers. The mapping from a shadow to a register is measured and documented in
``docs/XEX_COLOR_REGISTERS.md`` §5:

=========  =========  ================
Shadow     Address    GTIA register
=========  =========  ================
``COLOR0``  ``$02C4``  ``COLPF0``
``COLOR1``  ``$02C5``  ``COLPF1``
``COLOR2``  ``$02C6``  ``COLPF2``
``COLOR3``  ``$02C7``  ``COLPF3``
``COLOR4``  ``$02C8``  ``COLBK``
=========  =========  ================

Which *playfield colour* belongs in which shadow depends on the ANTIC mode, not on the palette
order, so that decision lives here and not in the exporter:

``D`` / ``E``
    Pixel values 0-3 select ``COLBK``, ``COLPF0``, ``COLPF1``, ``COLPF2`` (§7.1), i.e. the
    background (palette colour 0) has to go to ``COLOR4`` and colours 1-3 to ``COLOR0..COLOR2``.
    The old code wrote colour 0 to ``COLOR0``, which shifted every playfield colour by one register
    (§8.1).
``F``
    Mode F has no independent colours: ``COLPF2`` supplies the hue *and* the background luminance,
    ``COLPF1`` the foreground luminance (§7.2). Two arbitrary colours are impossible, so the mapping
    keeps one hue (the foreground's, because that is what draws the picture) and both luminances.
    This is option A of the report: visible and deterministic, but lossy by design.

``atari_colors`` keeps its historical meaning - the raw ``COLOR0..COLOR4`` shadow bytes:

* a list is written **verbatim**, so a user who configures registers by hand still gets exactly the
  bytes they typed (the GUI labels them ``COLOR0`` .. ``COLOR4`` for the same reason);
* ``None`` means "derive them from the converted palette": the converter's RGB colours are mapped to
  the nearest measured Atari colours (:mod:`converter.atari_match`) and then arranged as above.

Everything here is export-only: it never runs inside the image pipeline, so it cannot change a
conversion, a cache key or a golden image file.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from converter import atari_palette
from converter.atari_match import PAL_RGB, nearest_hardware_colors
from converter.types import ConversionConfig, ConversionResult, IndexedImage, RGBImage

REGISTER_COUNT = 5
"""``COLOR0`` .. ``COLOR4`` - the five shadows the XEX init routine writes."""

REGISTER_NAMES: tuple[str, ...] = ("COLOR0", "COLOR1", "COLOR2", "COLOR3", "COLOR4")

GTIA_REGISTERS: tuple[str, ...] = ("COLPF0", "COLPF1", "COLPF2", "COLPF3", "COLBK")
"""GTIA register each shadow feeds, in ``REGISTER_NAMES`` order (measured, see module docstring)."""

ANTIC_MODE_ALIASES: dict[str, str] = {
    "D": "D",
    "E": "E",
    "F": "F",
    "7": "D",
    "15": "E",
    "8": "F",
}
"""Accepted mode spellings: the ANTIC letters plus the BASIC/GTIA numbers the CLI allows."""

_MODE_CAPACITY: dict[str, int] = {"D": 4, "E": 4, "F": 2}
"""How many independent colours the mode can display (``COLPF3`` is not a playfield colour)."""

PIXEL_COLOR_REGISTERS: dict[str, tuple[int, ...]] = {
    "D": (4, 0, 1, 2),
    "E": (4, 0, 1, 2),
}
"""``COLOR`` index every bitmap value selects in modes D and E (§7.1).

Value 0 shows ``COLBK``, 1 ``COLPF0``, 2 ``COLPF1`` and 3 ``COLPF2``. The table is used in both
directions: :func:`arrange_registers` fills the registers from it and :func:`pixel_colors` reads the
displayed colour back through it, so the export and the preview cannot disagree.
"""


def mode_for_size(width: int, height: int) -> str:
    """ANTIC mode an export of this screen size uses.

    Raises ``ValueError`` with the CLI's wording when the size does not match a supported mode, so
    the message a user sees is the same whether the check happens here or in the exporter.
    """
    if (width, height) == (160, 96):
        return "D"
    if width == 160:
        return "E"
    if width == 320:
        return "F"
    raise ValueError(
        "Cannot infer ANTIC mode from image shape. Use --antic-mode with one of: D, E, F, 7, 15, 8."
    )


def resolve_antic_mode(antic_mode: str | None, width: int, height: int) -> str:
    """Normalise ``antic_mode`` (``None`` infers it from the size); ``ValueError`` when impossible."""
    if antic_mode is None:
        return mode_for_size(width, height)
    normalized = ANTIC_MODE_ALIASES.get(str(antic_mode).upper())
    if normalized is None:
        raise ValueError("Invalid ANTIC mode. Use one of: D, E, F, 7, 15, 8")
    return normalized


def pad_registers(values: Sequence[int]) -> list[int]:
    """Exactly ``REGISTER_COUNT`` bytes: the given values, then zeros."""
    padded = [int(value) & 0xFF for value in list(values)[:REGISTER_COUNT]]
    padded.extend([0] * (REGISTER_COUNT - len(padded)))
    return padded


def arrange_registers(colors: Sequence[int], mode: str) -> list[int]:
    """Place playfield colours (background first) into the ``COLOR0..COLOR4`` shadow order.

    ``colors`` are hardware colour bytes in palette order. Values the mode cannot use are dropped;
    a mode that needs more values than given gets zeros (a two-colour palette in mode E leaves
    ``COLPF1``/``COLPF2`` at ``$00``, which no pixel value ever selects).
    """
    palette = pad_registers(colors)
    registers = [0] * REGISTER_COUNT

    if mode == "F":
        background, foreground = palette[0], palette[1]
        hue = foreground & 0xF0
        registers[1] = hue | (foreground & 0x0F)  # COLOR1 -> COLPF1: foreground luminance
        registers[2] = hue | (background & 0x0F)  # COLOR2 -> COLPF2: hue + background luminance
        registers[4] = background  # COLOR4 -> COLBK stays the border colour
        return registers

    four_colour = PIXEL_COLOR_REGISTERS.get(mode)
    if four_colour is None:
        raise ValueError(f"unknown ANTIC mode {mode!r}; expected one of D, E, F")
    for value, register in enumerate(four_colour):
        registers[register] = palette[value]
    return registers


def derived_registers(palette_rgb: Sequence[Sequence[int]], mode: str) -> list[int]:
    """Registers for a converter palette: every RGB colour mapped to its nearest Atari colour."""
    hardware = nearest_hardware_colors(palette_rgb)
    return arrange_registers(hardware[: _MODE_CAPACITY.get(mode, 4)], mode)


def derived_registers_from_indices(palette_indices: Sequence[int], mode: str) -> list[int]:
    """Registers for palette *indices* of the converter table.

    Those indices (``converter.atari_palette``) are positions in an approximate colour table, **not**
    GTIA colour bytes: index ``0xEC`` means "dark grey" to the converter while the hardware reads it
    as hue 14, luminance 12. They therefore have to go through their RGB value and the measured
    palette before they can reach a register.
    """
    rgb = [
        atari_palette.get_color(int(index) % len(atari_palette.PALETTE)).rgb
        for index in palette_indices
    ]
    return derived_registers(rgb, mode)


def resolved_registers(
    result: ConversionResult,
    config: ConversionConfig,
    antic_mode: str | None = None,
) -> list[int]:
    """The five ``COLOR0..COLOR4`` bytes an export of ``result`` must write.

    A manual ``config.atari_colors`` wins and is used verbatim; ``None`` derives the values from the
    converted palette of ``result`` (the ``rgb2a8`` mapping), arranged for the mode of the export.
    """
    if config.atari_colors is not None:
        return pad_registers(config.atari_colors)
    height, width = result.indexed.shape
    mode = resolve_antic_mode(antic_mode, width, height)
    return derived_registers(result.palette_rgb, mode)


# --- what the hardware displays ----------------------------------------


def pixel_colors(registers: Sequence[int], mode: str) -> tuple[int, ...]:
    """The hardware colour byte every bitmap value displays, in pixel-value order.

    This is the reading direction of the very same rule :func:`arrange_registers` uses for writing,
    so an exported file and this frame show the same thing by construction:

    * modes D/E: value ``k`` shows ``COLOR`` ``PIXEL_COLOR_REGISTERS["E"][k]`` (0 -> ``COLBK``,
      1-3 -> ``COLPF0``-``COLPF2``);
    * mode F: value 0 shows ``COLPF2`` (its hue *and* luminance), value 1 shows the hue of ``COLPF2``
      with the luminance of ``COLPF1`` - the two-luminance limitation of the mode itself.
    """
    padded = pad_registers(registers)
    if mode == "F":
        hue = padded[2] & 0xF0
        return (padded[2], hue | (padded[1] & 0x0F))
    four_colour = PIXEL_COLOR_REGISTERS.get(mode)
    if four_colour is None:
        raise ValueError(f"unknown ANTIC mode {mode!r}; expected one of D, E, F")
    return tuple(padded[register] for register in four_colour)


def render_hardware_frame(indexed: IndexedImage, registers: Sequence[int], mode: str) -> RGBImage:
    """RGB frame of the bitmap as the GTIA colours display it (no scaling, no smoothing).

    Every bitmap value is passed through the same mask the packers in the XEX exporter apply
    (``& 3`` for 2bpp modes, ``& 1`` for ANTIC F), so a result with more colours than the mode can
    hold renders as the hardware would see it packed. The returned array is a fresh ``uint8``
    ``(H, W, 3)`` image; the inputs are never modified.
    """
    colours = pixel_colors(registers, mode)
    table = np.array([PAL_RGB[index] for index in colours], dtype=np.uint8)
    values = np.asarray(indexed, dtype=np.uint8) & (len(colours) - 1)
    return table[values]


@dataclass(frozen=True)
class HardwareFrame:
    """What ``Atari result`` shows: the mode, the registers and the RGB frame they produce."""

    mode: str
    registers: tuple[int, ...]
    rgb: RGBImage


class HardwareFrameRenderer:
    """Turns a result plus the export settings into the frame the hardware would display.

    Owns a small LRU keyed by the bitmap content, the mode and the registers, so the repeated calls a
    GUI makes (a register edit, a checkbox toggle, an unrelated parameter change) reuse the last
    render instead of recomputing it. Pure NumPy, no Qt and no global state: one renderer belongs to
    one owner, like the conversion engine does.

    ``frame()`` returns ``None`` when the screen size has no ANTIC mode: such a result cannot be
    exported as an executable at all, so there is no hardware display to show.
    """

    def __init__(self, capacity: int = 4) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1 (got {capacity})")
        self._capacity = capacity
        self._items: OrderedDict[tuple, HardwareFrame] = OrderedDict()

    def frame(
        self,
        result: ConversionResult,
        config: ConversionConfig,
        antic_mode: str | None = None,
    ) -> HardwareFrame | None:
        """The hardware frame of ``result`` for the colours ``config`` would export."""
        height, width = result.indexed.shape
        try:
            mode = resolve_antic_mode(antic_mode, width, height)
        except ValueError:
            return None
        registers = tuple(resolved_registers(result, config, mode))
        key = _frame_key(result.indexed, registers, mode)
        cached = self._items.get(key)
        if cached is not None:
            self._items.move_to_end(key)
            return cached

        frame = HardwareFrame(
            mode=mode,
            registers=registers,
            rgb=render_hardware_frame(result.indexed, registers, mode),
        )
        self._items[key] = frame
        self._items.move_to_end(key)
        while len(self._items) > self._capacity:
            self._items.popitem(last=False)
        return frame

    def clear(self) -> None:
        """Forget every rendered frame."""
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


def _frame_key(indexed: IndexedImage, registers: Sequence[int], mode: str) -> tuple:
    """Cache key of a frame: the bitmap bytes, the mode and the five register values."""
    digest = hashlib.blake2b(indexed.tobytes(), digest_size=16).digest()
    return (indexed.shape, digest, mode, tuple(registers))
