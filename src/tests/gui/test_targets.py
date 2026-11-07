"""Tests for the conversion targets (T2.7).

Pure data and pure helpers: the assertions spell out the values from plan §6.2 and the PAL pixel
aspects of §4.2, so a changed table is a visible failure rather than a silent GUI difference.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6", reason="PySide6 is not installed (install the 'gui' extra)")

from converter.aspect import pal_screen  # noqa: E402
from converter.types import MAX_ATARI_COLORS, ConversionConfig  # noqa: E402
from converter_gui.targets import (  # noqa: E402
    ALL_TARGETS,
    CUSTOM,
    TARGETS,
    TARGETS_BY_KEY,
    TargetSpec,
    apply_target,
    get_target,
    match_target,
)


def test_the_documented_targets_exist() -> None:
    assert [target.key for target in TARGETS] == [
        "antic_e_192",
        "antic_e_190",
        "antic_d_96",
        "antic_f_192",
    ]
    assert [target.label for target in TARGETS] == [
        "ANTIC E 160×192 (4 col)",
        "ANTIC E 160×190 (4 col)",
        "ANTIC D 160×96 (4 col)",
        "ANTIC F 320×192 (2 col)",
    ]


@pytest.mark.parametrize(
    ("key", "width", "height", "max_colors", "mode", "aspect"),
    [
        ("antic_e_192", 160, 192, MAX_ATARI_COLORS, "E", 1.86),
        ("antic_e_190", 160, 190, MAX_ATARI_COLORS, "E", 1.86),
        ("antic_d_96", 160, 96, MAX_ATARI_COLORS, "D", 0.93),
        ("antic_f_192", 320, 192, 2, "F", 0.93),
    ],
)
def test_target_values(
    key: str, width: int, height: int, max_colors: int, mode: str, aspect: float
) -> None:
    target = get_target(key)

    assert (target.width, target.height) == (width, height)
    assert target.max_colors == max_colors
    assert target.default_antic_mode == mode
    assert target.pixel_aspect == pytest.approx(aspect, abs=0.01)
    assert target.is_custom is False


def test_the_pixel_aspects_come_from_the_central_pal_table() -> None:
    """The GUI must not keep its own copy of the hardware facts (converter.aspect owns them)."""
    for target in TARGETS:
        screen = pal_screen(target.width, target.height)
        assert screen is not None
        assert target.pixel_aspect == screen.pixel_aspect

    assert get_target("antic_d_96").pixel_aspect != get_target("antic_e_192").pixel_aspect
    assert CUSTOM.pixel_aspect is None, "a Custom size must not get an invented factor"


def test_custom_is_last_and_has_no_size() -> None:
    assert ALL_TARGETS[-1] is CUSTOM
    assert CUSTOM.label == "Custom"
    assert CUSTOM.width is None and CUSTOM.height is None
    assert CUSTOM.is_custom is True
    assert CUSTOM.max_colors == MAX_ATARI_COLORS


def test_the_registry_is_complete() -> None:
    assert set(TARGETS_BY_KEY) == {target.key for target in ALL_TARGETS}
    assert get_target("antic_f_192") is TARGETS[-1]
    with pytest.raises(KeyError):
        get_target("does-not-exist")


@pytest.mark.parametrize(
    ("width", "height", "key"),
    [
        (160, 192, "antic_e_192"),
        (160, 190, "antic_e_190"),
        (160, 96, "antic_d_96"),
        (320, 192, "antic_f_192"),
    ],
)
def test_matching_a_size_returns_the_target(width: int, height: int, key: str) -> None:
    assert match_target(width, height) is get_target(key)


@pytest.mark.parametrize(("width", "height"), [(161, 192), (160, 191), (320, 96), (8, 8)])
def test_an_unknown_size_matches_custom(width: int, height: int) -> None:
    assert match_target(width, height) is CUSTOM


def test_applying_a_target_sets_the_size_and_does_not_mutate() -> None:
    config = ConversionConfig(width=64, height=48, colors=4)

    updated = apply_target(config, get_target("antic_f_192"))

    assert (updated.width, updated.height) == (320, 192)
    assert updated.colors == 2, "ANTIC F clamps the colour count"
    assert (config.width, config.height, config.colors) == (64, 48, 4), "the input is untouched"


def test_switching_back_does_not_invent_colours() -> None:
    config = apply_target(ConversionConfig(colors=4), get_target("antic_f_192"))
    assert config.colors == 2

    back = apply_target(config, get_target("antic_e_192"))

    assert back.colors == 2, "a narrower target only ever lowers the count"


def test_applying_custom_keeps_the_size() -> None:
    config = ConversionConfig(width=100, height=200, colors=3)

    updated = apply_target(config, CUSTOM)

    assert (updated.width, updated.height) == (100, 200)
    assert updated.colors == 3


def test_targets_are_hashable_and_comparable() -> None:
    assert TargetSpec("k", "label", 10, 20, 4, "E") == TargetSpec("k", "label", 10, 20, 4, "E")
    assert len({target.key for target in ALL_TARGETS}) == len(ALL_TARGETS)
