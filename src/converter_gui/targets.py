"""Named conversion targets (Q6, plan §6.2).

A target is explicit GUI state on top of :class:`~converter.types.ConversionConfig`: it writes
``width``/``height`` and caps ``colors``, and it supplies the default ANTIC mode the export dialog
starts with. The PAL pixel aspect is *not* stored here - it is looked up from
:mod:`converter.aspect`, which owns the hardware facts, so the preview, the engine and this table
cannot drift apart. ``AtariMode`` and the saved JSON are left alone, so a configuration produced
here still round-trips through the CLI.

The module is plain data plus pure helpers: no Qt, no engine, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from converter.aspect import pal_screen
from converter.types import MAX_ATARI_COLORS, ConversionConfig

CUSTOM_LABEL = "Custom"
"""Label of the editable entry, which has no fixed size."""


@dataclass(frozen=True)
class TargetSpec:
    """One entry of the target combo.

    ``width``/``height`` are ``None`` for the Custom entry, whose size comes from the configuration.
    ``max_colors`` is the hardware limit of the target (4 for the 2bpp modes, 2 for ANTIC F) and
    ``default_antic_mode`` is the XEX mode the export dialog starts with.
    """

    key: str
    label: str
    width: int | None
    height: int | None
    max_colors: int
    default_antic_mode: str

    @property
    def is_custom(self) -> bool:
        """True for the editable entry."""
        return self.width is None or self.height is None

    @property
    def pixel_aspect(self) -> float | None:
        """PAL pixel aspect of the target, from :data:`converter.aspect.PAL_SCREENS`.

        ``None`` for the Custom entry: a size without a known geometry must not get an invented
        factor (``ConversionConfig.pixel_aspect`` is the explicit way to correct such a size).
        """
        if self.width is None or self.height is None:
            return None
        screen = pal_screen(self.width, self.height)
        return screen.pixel_aspect if screen is not None else None


TARGETS: tuple[TargetSpec, ...] = (
    TargetSpec("antic_e_192", "ANTIC E 160×192 (4 col)", 160, 192, MAX_ATARI_COLORS, "E"),
    TargetSpec("antic_e_190", "ANTIC E 160×190 (4 col)", 160, 190, MAX_ATARI_COLORS, "E"),
    TargetSpec("antic_d_96", "ANTIC D 160×96 (4 col)", 160, 96, MAX_ATARI_COLORS, "D"),
    TargetSpec("antic_f_192", "ANTIC F 320×192 (2 col)", 320, 192, 2, "F"),
)

CUSTOM = TargetSpec("custom", CUSTOM_LABEL, None, None, MAX_ATARI_COLORS, "E")

ALL_TARGETS: tuple[TargetSpec, ...] = (*TARGETS, CUSTOM)
"""Every combo entry, in display order; the Custom entry is last."""

TARGETS_BY_KEY: dict[str, TargetSpec] = {target.key: target for target in ALL_TARGETS}


def get_target(key: str) -> TargetSpec:
    """Return the target registered under ``key`` (``KeyError`` if there is none)."""
    return TARGETS_BY_KEY[key]


def match_target(width: int, height: int) -> TargetSpec:
    """Return the target with exactly this size, or :data:`CUSTOM`."""
    for target in TARGETS:
        if target.width == width and target.height == height:
            return target
    return CUSTOM


def apply_target(config: ConversionConfig, target: TargetSpec) -> ConversionConfig:
    """Return a copy of ``config`` set up for ``target``; the argument is never mutated.

    The size comes from the target unless it is Custom, and ``colors`` is only ever lowered to the
    target's hardware limit: switching to ANTIC F caps it at 2, and switching back does not invent
    colours the user did not ask for.
    """
    if target.is_custom:
        return replace(config, colors=min(config.colors, target.max_colors))
    assert target.width is not None and target.height is not None, "a fixed target needs a size"
    return replace(
        config,
        width=target.width,
        height=target.height,
        colors=min(config.colors, target.max_colors),
    )
