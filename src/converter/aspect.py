"""PAL pixel aspect ratio correction (geometry stage).

Atari 8-bit pixels are not square. On a PAL screen a 160-wide ANTIC D/E pixel is roughly twice as
wide as it is tall, while the 320-wide ANTIC F pixel is (nearly) square - see the "Pixel Aspect
Ratios" chapter of the Altirra Hardware Reference Manual. Scaling a square-pixel image onto the
target grid therefore changes the proportions of everything in it, and no single factor fixes all
modes: the factor belongs to the mode, not to the converter.

This module owns the hardware facts (:data:`PAL_SCREENS`) and the geometry that follows from them.
It is a pure NumPy module - no Qt, no I/O - and it lives in the core package so the CLI, the engine
and the GUI all read the same table instead of repeating it.

The correction runs before any colour work (palette selection, quantisation, dithering): it decides
which part of the source reaches those stages, and the margins it adds are part of the converted
image, so it cannot be moved later.

Two modes correct the geometry, and one keeps the historical behaviour:

``fit``       the whole source stays visible; the target grid gains background-coloured margins
``crop``      the target grid is filled completely; the source loses outer rows/columns
``disabled``  the source is stretched onto the whole target grid (the previous behaviour)

Only PAL is modelled: NTSC uses different pixel clocks and is out of scope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from converter import atari_palette
from converter.resize import resize_image
from converter.types import AspectMode, ConversionConfig, RGBImage


@dataclass(frozen=True)
class PalScreen:
    """One PAL screen geometry: the framebuffer grid and the width of its pixels.

    ``pixel_aspect`` is the width/height ratio of a single pixel on screen: 2.08 for the 160-wide
    ANTIC D/E modes and 1.04 for the 320-wide ANTIC F mode.
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
    PalScreen("ANTIC D", 160, 96, 1.04),
    PalScreen("ANTIC E", 160, 192, 2.08),
    # ANTIC E with the last two scanlines dropped: the size the BIN/ASM/DTA exporters accept. Fewer
    # rows do not change how wide a pixel is, so it keeps the ANTIC E factor.
    PalScreen("ANTIC E (190 rows)", 160, 190, 2.08),
    PalScreen("ANTIC F", 320, 192, 1.04),
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


def preview_pixel_aspect(config: ConversionConfig) -> float:
    """Pixel aspect to display a result with: ``1.0`` when the correction is off or not applicable.

    ``disabled`` deliberately returns square pixels: the conversion is unchanged there, so the
    preview must stay unchanged too. Every other mode shows the pixels with their real shape, which
    is what makes the result look the way it will look on an Atari monitor.
    """
    if config.aspect is AspectMode.DISABLED:
        return 1.0
    factor = effective_pixel_aspect(config)
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
