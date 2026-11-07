"""Tests for the PAL pixel aspect ratio correction (``converter.aspect``).

The geometry is checked against explicit numbers (content sizes, offsets, margins) instead of just
"a plan was returned": a wrong factor or a wrong rounding rule has to fail here. The integration tests
then prove the two properties the feature promises - the correction happens before any colour work,
and ``disabled`` reproduces the previous result byte for byte.
"""

from __future__ import annotations

import numpy as np
import pytest

from converter.aspect import (
    PAL_SCREENS,
    apply_geometry,
    background_rgb,
    effective_pixel_aspect,
    pal_screen,
    pixel_aspect_for,
    plan_geometry,
    preview_pixel_aspect,
)
from converter.engine import ConversionEngine
from converter.pipeline import prepare_image
from converter.resize import resize_image
from converter.sharpen import apply_sharpen
from converter.types import (
    AspectMode,
    ConversionConfig,
    DitherMethod,
    PaletteMethod,
    ResizeMethod,
)

ANTIC_D = (160, 96)
ANTIC_E = (160, 192)
ANTIC_F = (320, 192)

E_FIT = ConversionConfig(width=160, height=192, aspect=AspectMode.FIT)
E_CROP = ConversionConfig(width=160, height=192, aspect=AspectMode.CROP)


def _solid(width: int, height: int, colour: tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :] = colour
    return image


# --- the PAL table ------------------------------------------------------


def test_the_pal_table_holds_the_documented_pal_factors() -> None:
    assert pal_screen(*ANTIC_D).pixel_aspect == pytest.approx(1.04)
    assert pal_screen(*ANTIC_E).pixel_aspect == pytest.approx(2.08)
    assert pal_screen(*ANTIC_F).pixel_aspect == pytest.approx(1.04)
    assert [screen.name for screen in PAL_SCREENS][:2] == ["ANTIC D", "ANTIC E"]


def test_the_modes_do_not_share_one_factor() -> None:
    """A single 2:1 factor for every mode is exactly what the hardware does not do."""
    assert pal_screen(*ANTIC_D).pixel_aspect != pal_screen(*ANTIC_E).pixel_aspect
    assert pal_screen(*ANTIC_F).pixel_aspect != pal_screen(*ANTIC_E).pixel_aspect


def test_the_pal_screens_cover_the_same_physical_area() -> None:
    """Different pixel grids, same display aspect: this is why one factor cannot serve them all."""
    aspects = [pal_screen(*size).display_aspect for size in (ANTIC_D, ANTIC_E, ANTIC_F)]

    for aspect in aspects[1:]:
        assert aspect == pytest.approx(aspects[0], rel=1e-6)
    assert aspects[0] == pytest.approx((160 * 2.08) / 192)


def test_the_190_row_screen_keeps_the_antic_e_factor() -> None:
    screen = pal_screen(160, 190)

    assert screen is not None
    assert screen.pixel_aspect == pytest.approx(2.08), "fewer rows do not change the pixel width"


@pytest.mark.parametrize(("width", "height"), [(200, 100), (160, 100), (8, 8), (640, 480)])
def test_an_unknown_size_has_no_factor(width: int, height: int) -> None:
    assert pal_screen(width, height) is None
    assert pixel_aspect_for(width, height) is None


def test_an_explicit_factor_wins_over_the_table() -> None:
    assert pixel_aspect_for(200, 100, 1.5) == pytest.approx(1.5)
    assert pixel_aspect_for(*ANTIC_E, 1.6) == pytest.approx(1.6)


@pytest.mark.parametrize("explicit", [0.0, -1.0, float("nan"), float("inf")])
def test_an_invalid_explicit_factor_counts_as_not_set(explicit: float) -> None:
    assert pixel_aspect_for(*ANTIC_E, explicit) is None
    assert pixel_aspect_for(200, 100, explicit) is None


# --- geometry plans ----------------------------------------------------


def test_disabled_needs_no_plan() -> None:
    assert plan_geometry((640, 480), ConversionConfig(aspect=AspectMode.DISABLED)) is None


def test_an_unknown_size_needs_no_plan() -> None:
    config = ConversionConfig(width=200, height=100, aspect=AspectMode.FIT)

    assert plan_geometry((640, 480), config) is None, "no factor, no invented correction"


def test_fit_of_a_source_with_the_target_size() -> None:
    plan = plan_geometry(ANTIC_E, E_FIT)

    assert plan is not None
    assert plan.pixel_aspect == pytest.approx(2.08)
    assert plan.content == (77, 192), "160x192 square pixels are narrower than the PAL screen"
    assert plan.offset == (41, 0), "centred: (160 - 77) // 2"
    assert plan.crop is None


def test_fit_of_a_panoramic_source() -> None:
    plan = plan_geometry((400, 100), E_FIT)

    assert plan is not None
    assert plan.content == (160, 83), "round(160 * 2.08 / 4.0) = 83"
    assert plan.offset == (0, 54)
    assert 192 - 54 - 83 == 55, "an odd remainder stays at the bottom edge"


def test_fit_of_a_portrait_source() -> None:
    plan = plan_geometry((100, 400), E_FIT)

    assert plan is not None
    assert plan.content == (23, 192), "round(192 * 0.25 / 2.08) = 23"
    assert plan.offset == (68, 0), "centred: (160 - 23) // 2"


def test_fit_of_the_same_source_in_antic_d_fills_fewer_rows() -> None:
    plan = plan_geometry(ANTIC_E, ConversionConfig(width=160, height=96, aspect=AspectMode.FIT))

    assert plan is not None
    assert plan.content == (77, 96), "same display aspect, half the rows"
    assert plan.offset == (41, 0)


@pytest.mark.parametrize("target", [ANTIC_D, ANTIC_E, ANTIC_F])
@pytest.mark.parametrize(
    "source", [(160, 192), (400, 100), (100, 400), (640, 480), (17, 300), (300, 17)]
)
def test_fit_content_keeps_the_source_proportions(target: tuple[int, int], source) -> None:
    """The content box is the ideal one rounded to whole pixels - never a fixed multiplier."""
    config = ConversionConfig(width=target[0], height=target[1], aspect=AspectMode.FIT)
    plan = plan_geometry(source, config)

    assert plan is not None
    width, height = plan.content
    assert 1 <= width <= target[0]
    assert 1 <= height <= target[1]
    source_aspect = source[0] / source[1]
    if width == target[0]:
        ideal = width * plan.pixel_aspect / source_aspect
        assert abs(height - ideal) <= 0.5 + 1e-9, "height is the rounded ideal, not half of it"
    else:
        ideal = height * source_aspect / plan.pixel_aspect
        assert abs(width - ideal) <= 0.5 + 1e-9, "width is the rounded ideal, not double it"


@pytest.mark.parametrize("target", [ANTIC_D, ANTIC_E, ANTIC_F])
def test_fit_stays_inside_the_target(target: tuple[int, int]) -> None:
    config = ConversionConfig(width=target[0], height=target[1], aspect=AspectMode.FIT)

    for source in [(1, 1), (1000, 1), (1, 1000), (160, 192), (320, 240)]:
        plan = plan_geometry(source, config)
        assert plan is not None
        width, height = plan.content
        left, top = plan.offset
        assert 0 < width <= target[0] and 0 < height <= target[1]
        assert 0 <= left <= target[0] - width
        assert 0 <= top <= target[1] - height


def test_crop_trims_the_rows_of_a_narrow_source() -> None:
    plan = plan_geometry(ANTIC_E, E_CROP)

    assert plan is not None
    assert plan.crop == (0, 50, 160, 92), "round(160 / 1.7333) = 92 rows survive, centred"
    assert plan.content == ANTIC_E, "crop fills the whole grid"
    assert plan.offset == (0, 0)


def test_crop_trims_the_columns_of_a_wide_source() -> None:
    plan = plan_geometry((400, 100), E_CROP)

    assert plan is not None
    assert plan.crop == (113, 0, 173, 100), "round(100 * 1.7333) = 173 columns survive"


def test_crop_keeps_a_source_that_already_has_the_screen_proportions() -> None:
    plan = plan_geometry((173, 100), E_CROP)

    assert plan is not None
    assert plan.crop == (0, 0, 173, 100)


def test_crop_of_antic_f_uses_its_own_factor() -> None:
    config = ConversionConfig(width=320, height=192, aspect=AspectMode.CROP)
    plan = plan_geometry((160, 192), config)

    assert plan is not None
    assert plan.pixel_aspect == pytest.approx(1.04)
    assert plan.crop == (0, 50, 160, 92), "same display aspect as ANTIC E, different grid"


# --- the resulting image ------------------------------------------------


def test_the_margins_are_filled_with_the_background_colour() -> None:
    image = _solid(400, 100, (200, 30, 40))
    config = ConversionConfig(aspect=AspectMode.FIT, background_index=4)

    fitted = apply_geometry(image, config)

    assert fitted.shape == (192, 160, 3)
    background = background_rgb(config)
    assert tuple(fitted[0, 0]) == background
    assert tuple(fitted[0, 159]) == background
    assert tuple(fitted[191, 0]) == background
    assert tuple(fitted[96, 80]) == (200, 30, 40), "the content in the middle is untouched"


def test_a_different_background_index_changes_the_margins() -> None:
    image = _solid(400, 100)

    black = apply_geometry(image, ConversionConfig(aspect=AspectMode.FIT, background_index=0))
    white = apply_geometry(image, ConversionConfig(aspect=AspectMode.FIT, background_index=4))

    assert tuple(black[0, 0]) == (0, 0, 0)
    assert tuple(white[0, 0]) == (255, 255, 255)


def test_apply_geometry_returns_the_target_size() -> None:
    for target in (ANTIC_D, ANTIC_E, ANTIC_F):
        for mode in (AspectMode.FIT, AspectMode.CROP):
            config = ConversionConfig(width=target[0], height=target[1], aspect=mode)
            assert apply_geometry(_solid(320, 240), config).shape == (target[1], target[0], 3)


def test_apply_geometry_with_disabled_resizes_everything() -> None:
    image = _solid(400, 100, (10, 20, 30))
    config = ConversionConfig(aspect=AspectMode.DISABLED)

    fitted = apply_geometry(image, config)

    assert fitted.shape == (192, 160, 3)
    assert tuple(fitted[0, 0]) == (10, 20, 30), "no margins, the image is stretched"


# --- the pipeline and the engine ---------------------------------------


def test_disabled_reproduces_the_previous_prepare_result() -> None:
    """The historical behaviour, computed the way it was computed before the correction existed."""
    image = _solid(400, 100, (10, 20, 30))
    config = ConversionConfig(
        width=160, height=192, resize=ResizeMethod.BICUBIC, sharpen=True, sharpen_strength=1.5
    )

    expected = apply_sharpen(resize_image(image, 160, 192, ResizeMethod.BICUBIC), True, 1.5)
    np.testing.assert_array_equal(prepare_image(image, config), expected)


def test_prepare_image_applies_the_correction_before_sharpening() -> None:
    image = _solid(400, 100, (10, 20, 30))
    config = ConversionConfig(aspect=AspectMode.FIT, sharpen=True, sharpen_strength=2.0)

    prepared = prepare_image(image, config)

    assert prepared.shape == (192, 160, 3), "the margins survive the sharpening stage"


def test_the_correction_runs_before_the_palette_and_the_dithering() -> None:
    """The margins reach the quantiser as background colour, so they end up as background indices."""
    engine = ConversionEngine()
    engine.set_source_array(_solid(400, 100, (150, 150, 150)))
    config = ConversionConfig(
        width=160,
        height=192,
        aspect=AspectMode.FIT,
        colors=4,
        dither=DitherMethod.NONE,
        palette=PaletteMethod.POPULARITY,
        background_index=0,
    )

    result = engine.convert(config)

    assert result.source_resized.shape == (192, 160, 3)
    assert result.indexed.shape == (192, 160)
    assert tuple(result.source_resized[0, 0]) == background_rgb(config)
    assert result.indexed[0, 0] == 0, "the margin is the background index"
    assert result.indexed[96, 80] != 0, "the content is not the background"


def test_the_margins_stay_uniform_when_dithering() -> None:
    """Error diffusion must not push the flat margin into a pattern."""
    engine = ConversionEngine()
    engine.set_source_array(_solid(400, 100, (150, 150, 150)))
    config = ConversionConfig(
        width=160, height=192, aspect=AspectMode.FIT, colors=4, dither=DitherMethod.FLOYD
    )

    result = engine.convert(config)

    top_margin = result.indexed[:54, :]
    bottom_margin = result.indexed[137:, :]
    assert np.unique(top_margin).tolist() == [0]
    assert np.unique(bottom_margin).tolist() == [0]


def test_crop_feeds_a_trimmed_source_to_the_colours() -> None:
    engine = ConversionEngine()
    source = _solid(400, 100, (150, 150, 150))
    source[:, :40] = (255, 0, 0)  # left border that crop must remove
    engine.set_source_array(source)
    config = ConversionConfig(
        width=160, height=192, aspect=AspectMode.CROP, dither=DitherMethod.NONE, colors=4
    )

    result = engine.convert(config)

    assert result.source_resized.shape == (192, 160, 3)
    assert tuple(result.source_resized[96, 0]) != (255, 0, 0), "the red border was cropped away"


# --- cache, configuration and preview -----------------------------------


def test_the_engine_cache_key_covers_the_geometry() -> None:
    from converter.engine import _l1_key

    disabled = ConversionConfig()
    fit = ConversionConfig(aspect=AspectMode.FIT)
    crop = ConversionConfig(aspect=AspectMode.CROP)

    assert _l1_key(fit) != _l1_key(disabled), "enabling the correction must miss the cache"
    assert _l1_key(fit) != _l1_key(crop), "the two modes produce different pixels"
    assert _l1_key(fit) == _l1_key(ConversionConfig(aspect=AspectMode.FIT))
    assert _l1_key(fit) != _l1_key(ConversionConfig(aspect=AspectMode.FIT, pixel_aspect=1.5))


def test_the_cache_key_ignores_an_explicit_factor_while_disabled() -> None:
    from converter.engine import _l1_key

    assert _l1_key(ConversionConfig(pixel_aspect=1.5)) == _l1_key(ConversionConfig())


def test_switching_the_mode_recomputes_and_repeats_come_from_the_cache() -> None:
    engine = ConversionEngine()
    engine.set_source_array(_solid(200, 100, (120, 120, 120)))
    fit = ConversionConfig(width=160, height=192, aspect=AspectMode.FIT)

    first = engine.convert(fit)
    assert first.timings["prepare"] > 0.0

    repeated = engine.convert(fit)
    assert repeated.timings["prepare"] == 0.0, "the same geometry is cached"

    switched = engine.convert(ConversionConfig(width=160, height=192, aspect=AspectMode.DISABLED))
    assert switched.timings["prepare"] > 0.0, "another mode is another prepared image"

    assert not np.array_equal(switched.source_resized, first.source_resized)


def test_a_configuration_round_trips_the_correction() -> None:
    config = ConversionConfig(aspect=AspectMode.CROP, pixel_aspect=1.5)

    restored = ConversionConfig.from_dict(config.to_dict())

    assert restored.aspect is AspectMode.CROP
    assert restored.pixel_aspect == pytest.approx(1.5)


def test_a_default_configuration_keeps_its_saved_shape() -> None:
    """Old configurations (and `--save-config` output) must not gain keys."""
    data = ConversionConfig().to_dict()

    assert "aspect" not in data
    assert "pixel_aspect" not in data


def test_an_old_configuration_loads_with_the_correction_off() -> None:
    config = ConversionConfig.from_dict({"width": 160, "height": 96, "dither": "bayer4"})

    assert config.aspect is AspectMode.DISABLED
    assert config.pixel_aspect is None


def test_an_unknown_mode_in_a_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        ConversionConfig.from_dict({"aspect": "stretch"})


def test_effective_pixel_aspect_follows_the_configured_mode() -> None:
    assert effective_pixel_aspect(ConversionConfig(width=160, height=96)) == pytest.approx(1.04)
    assert effective_pixel_aspect(ConversionConfig(width=160, height=192)) == pytest.approx(2.08)
    assert effective_pixel_aspect(ConversionConfig(width=320, height=192)) == pytest.approx(1.04)
    assert effective_pixel_aspect(ConversionConfig(width=200, height=100)) is None


def test_the_preview_aspect_is_square_while_disabled() -> None:
    """Disabled keeps the preview as it was: the result is drawn with square pixels."""
    assert preview_pixel_aspect(ConversionConfig()) == pytest.approx(1.0)
    assert preview_pixel_aspect(ConversionConfig(width=160, height=192)) == pytest.approx(1.0)


def test_the_preview_aspect_follows_the_enabled_mode() -> None:
    fit = ConversionConfig(width=160, height=192, aspect=AspectMode.FIT)

    assert preview_pixel_aspect(fit) == pytest.approx(2.08)
    assert preview_pixel_aspect(
        ConversionConfig(width=320, height=192, aspect=AspectMode.FIT)
    ) == pytest.approx(1.04)
    assert preview_pixel_aspect(
        ConversionConfig(width=160, height=96, aspect=AspectMode.CROP)
    ) == pytest.approx(1.04)


def test_the_preview_aspect_stays_square_for_an_unknown_size() -> None:
    config = ConversionConfig(width=200, height=100, aspect=AspectMode.FIT)

    assert preview_pixel_aspect(config) == pytest.approx(1.0)


def test_the_preview_aspect_uses_the_explicit_factor() -> None:
    config = ConversionConfig(width=200, height=100, aspect=AspectMode.FIT, pixel_aspect=1.5)

    assert preview_pixel_aspect(config) == pytest.approx(1.5)
