"""PAL pixel aspect ratio (geometry stage, and the shape the preview draws).

Atari 8-bit pixels are not square. This module owns the one table both places read: the conversion
geometry (:func:`plan_geometry` - "which part of the source lands where on the target grid") and the
preview (:func:`preview_pixel_aspect` - "how wide is one pixel when the result is displayed"). They are
independent mechanisms that must agree, so they share the numbers here.

The numbers come from a measurement, not from a re-derivation: a PAL Altirra run of the exported XEX
(docs/XEX_COLOR_REGISTERS.md documents the same emulator setup) renders the 336x240 Atari output into a
1.3:1 rectangle and reports

    RenderFrame: rect=(131.3,21.0,1017.4,699.0) tex=(336,240)     -> 886.1 x 678.0, aspect 1.3069

so one raster column is ``1.3 / (336/240) = 0.93`` of a raster line on screen. The Atari samples a
160-wide mode with two columns per pixel and one line per pixel row (ANTIC mode D doubles every row,
so its pixel is two lines tall), while the 320-wide mode F uses one column per pixel:

==========================  =====================================  ==========
mode                        pixel on screen                        PAR
==========================  =====================================  ==========
ANTIC D  (160x96)           2 columns x 2 lines (rows doubled)      0.93
ANTIC E  (160x192, 160x190)  2 columns x 1 line                     1.86
ANTIC F  (320x192)           1 column  x 1 line                     0.93
==========================  =====================================  ==========

All three therefore cover the same physical area (:attr:`PalScreen.display_aspect` is equal for every
mode), which is why ANTIC E and ANTIC F pictures have identical proportions and ANTIC D is simply half
as tall.

The module is pure NumPy - no Qt, no I/O - so the CLI, the engine and the GUI read the same table.

Only PAL is modelled: NTSC uses a different pixel clock and is out of scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from converter import atari_palette
from converter.resize import resize_image
from converter.types import AspectMode, ConversionConfig, RGBImage

PAL_RASTER = (336, 240)
"""Columns x lines of the PAL output the emulator produces, as measured in Altirra."""

PAL_DISPLAY_ASPECT = 1.3
"""Aspect (width/height) that output is displayed with - measured from Altirra's render rectangle."""

_RASTER_COLUMN = PAL_DISPLAY_ASPECT / (PAL_RASTER[0] / PAL_RASTER[1])
"""One raster column relative to one raster line on screen: ``1.3 / 1.4 = 0.93``."""


@dataclass(frozen=True)
class PalScreen:
    """One PAL screen geometry: the framebuffer grid and the width of its pixels.

    ``pixel_aspect`` is the width/height ratio of a single pixel on screen (see the module
    docstring for how it is measured).
    """

    name: str
    width: int
    height: int
    pixel_aspect: float

    @property
    def display_aspect(self) -> float:
        """Physical width/height ratio of the whole display area."""
        return (self.width * self.pixel_aspect) / self.height


PAL_SCREENS: tuple[PalScreen, ...] = (
    # Two columns wide, two lines tall: ANTIC mode D doubles every scanline.
    PalScreen("ANTIC D", 160, 96, _RASTER_COLUMN),
    # Two columns wide, one line tall.
    PalScreen("ANTIC E", 160, 192, 2 * _RASTER_COLUMN),
    # ANTIC E with the last two scanlines dropped: the size the BIN/ASM/DTA exporters accept. Fewer
    # rows do not change how wide a pixel is, so it keeps the ANTIC E factor.
    PalScreen("ANTIC E (190 rows)", 160, 190, 2 * _RASTER_COLUMN),
    # One column wide, one line tall.
    PalScreen("ANTIC F", 320, 192, _RASTER_COLUMN),
)
"""The single source of truth for the PAL pixel aspect ratios."""

PAL_SCREENS_BY_SIZE: dict[tuple[int, int], PalScreen] = {
    (screen.width, screen.height): screen for screen in PAL_SCREENS
}


def pal_screen(width: int, height: int) -> PalScreen | None:
    """Return the PAL screen with exactly this framebuffer size (``None`` when there is none)."""
    return PAL_SCREENS_BY_SIZE.get((int(width), int(height)))


def pixel_aspect_for(width: int, height: int, explicit: float | None = None) -> float | None:
    """PAL pixel aspect of a target grid.

    An explicit ``explicit`` factor wins over the table - that is the escape hatch for a Custom size,
    which has no hardware geometry of its own. Non-positive or non-finite values count as "not set"
    rather than as an error, so a broken configuration falls back to the historical behaviour
    instead of producing a division by zero. Returns ``None`` when the size has no known geometry.
    """
    if explicit is not None:
        factor = float(explicit)
        if math.isfinite(factor) and factor > 0.0:
            return factor
        return None
    screen = pal_screen(width, height)
    return screen.pixel_aspect if screen is not None else None


def effective_pixel_aspect(config: ConversionConfig) -> float | None:
    """PAL pixel aspect the configuration converts with (``None`` when it is unknown)."""
    return pixel_aspect_for(config.width, config.height, config.pixel_aspect)


def preview_pixel_aspect(config: ConversionConfig, size: tuple[int, int] | None = None) -> float:
    """Pixel aspect the *preview* must draw a result with.

    This is mechanism B (how wide a hardware pixel is), not mechanism A (how the source was fitted
    onto the grid): it therefore does **not** look at ``config.aspect``. Correcting the conversion
    geometry and displaying non-square pixels are independent - with ``disabled`` the bitmap is a
    stretched copy of the source, and the Atari still shows it with its own pixel shape, so the
    preview has to apply the same factor or it stops matching the hardware.

    ``size`` is the ``(width, height)`` of the result on screen; without it the configuration's size
    is used. The result's own size is what the display depends on - a Custom grid may be on screen
    while the panel already shows another target.

    Returns ``1.0`` only when that size has no known geometry (a *Custom* grid without an explicit
    pixel aspect): there is nothing to apply then, and the result has no hardware display at all.
    """
    if size is None:
        width, height = config.width, config.height
    else:
        width, height = size
    factor = pixel_aspect_for(width, height, config.pixel_aspect)
    return 1.0 if factor is None else factor


@dataclass(frozen=True)
class GeometryPlan:
    """Where the source lands in the target grid, in whole pixels.

    ``content`` is the size of the resized source, ``offset`` its top-left corner inside the target
    grid (both equal to the target for ``crop``, which trims instead of padding) and ``crop`` the
    source rectangle that survives - ``None`` for ``fit``, which keeps all of it.
    """

    mode: AspectMode
    pixel_aspect: float
    target: tuple[int, int]
    content: tuple[int, int]
    offset: tuple[int, int]
    crop: tuple[int, int, int, int] | None = None


def _round_extent(value: float, limit: int) -> int:
    """Nearest whole pixel count in ``1..limit``; halves round up, so the result never flips."""
    return max(1, min(limit, int(math.floor(value + 0.5))))


def _center(outer: int, inner: int) -> int:
    """Offset that centres ``inner`` inside ``outer``; an odd remainder stays at the far edge."""
    return (outer - inner) // 2


def _plan_fit(
    source: tuple[int, int], config: ConversionConfig, pixel_aspect: float
) -> GeometryPlan:
    """Largest content box with the source proportions, centred on a background-filled grid."""
    source_width, source_height = source
    target_width, target_height = config.width, config.height
    source_aspect = source_width / source_height
    display_aspect = (target_width * pixel_aspect) / target_height

    if source_aspect >= display_aspect:
        # Wider than the screen: the width fills it and the height follows from the proportions.
        content_width = target_width
        content_height = _round_extent(target_width * pixel_aspect / source_aspect, target_height)
    else:
        content_height = target_height
        content_width = _round_extent(target_height * source_aspect / pixel_aspect, target_width)

    return GeometryPlan(
        mode=AspectMode.FIT,
        pixel_aspect=pixel_aspect,
        target=(target_width, target_height),
        content=(content_width, content_height),
        offset=(_center(target_width, content_width), _center(target_height, content_height)),
    )


def _plan_crop(
    source: tuple[int, int], config: ConversionConfig, pixel_aspect: float
) -> GeometryPlan:
    """Source rectangle with the screen proportions, centred and resized to the whole target grid."""
    source_width, source_height = source
    target_width, target_height = config.width, config.height
    display_aspect = (target_width * pixel_aspect) / target_height
    source_aspect = source_width / source_height

    if source_aspect > display_aspect:
        crop_height = source_height
        crop_width = _round_extent(source_height * display_aspect, source_width)
    else:
        crop_width = source_width
        crop_height = _round_extent(source_width / display_aspect, source_height)

    return GeometryPlan(
        mode=AspectMode.CROP,
        pixel_aspect=pixel_aspect,
        target=(target_width, target_height),
        content=(target_width, target_height),
        offset=(0, 0),
        crop=(
            _center(source_width, crop_width),
            _center(source_height, crop_height),
            crop_width,
            crop_height,
        ),
    )


def plan_geometry(source_size: tuple[int, int], config: ConversionConfig) -> GeometryPlan | None:
    """Geometry plan for a ``(width, height)`` source, or ``None`` when nothing has to change.

    ``None`` means "scale the whole source onto the target grid", which is what ``disabled`` asks
    for and what a size without a known pixel aspect falls back to.
    """
    if config.aspect is AspectMode.DISABLED:
        return None
    pixel_aspect = effective_pixel_aspect(config)
    if pixel_aspect is None:
        return None
    source_width, source_height = int(source_size[0]), int(source_size[1])
    if source_width < 1 or source_height < 1 or config.width < 1 or config.height < 1:
        return None
    source = (source_width, source_height)
    if config.aspect is AspectMode.CROP:
        return _plan_crop(source, config, pixel_aspect)
    return _plan_fit(source, config, pixel_aspect)


def background_rgb(config: ConversionConfig) -> tuple[int, int, int]:
    """Colour the margins get: the hardware colour of ``background_index``.

    The palette selector always keeps the background index in the palette, so the quantiser maps
    this exact colour back to that index and the margins really are the configured background.
    """
    return atari_palette.get_color(config.background_index % len(atari_palette.PALETTE)).rgb


def apply_geometry(image: RGBImage, config: ConversionConfig) -> RGBImage:
    """Resize ``image`` onto the target grid, correcting the PAL pixel aspect when asked to.

    Returns a ``config.height`` x ``config.width`` RGB array. ``disabled`` (and any target without a
    known pixel aspect) resizes the whole source, i.e. the historical result.
    """
    plan = plan_geometry((image.shape[1], image.shape[0]), config)
    if plan is None:
        return resize_image(image, config.width, config.height, config.resize)

    if plan.crop is not None:
        left, top, width, height = plan.crop
        cropped = image[top : top + height, left : left + width]
        return resize_image(cropped, *plan.target, config.resize)

    content_width, content_height = plan.content
    resized = resize_image(image, content_width, content_height, config.resize)
    target_width, target_height = plan.target
    canvas = np.empty((target_height, target_width, 3), dtype=np.uint8)
    canvas[:, :] = background_rgb(config)
    left, top = plan.offset
    canvas[top : top + content_height, left : left + content_width] = resized
    return canvas
