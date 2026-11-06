"""Tests for the cached conversion engine."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import converter.engine as engine_module
import converter.pipeline as pipeline_module
from converter.engine import ConversionCancelled, ConversionEngine
from converter.pipeline import convert_image
from converter.types import (
    AtariMode,
    ConversionConfig,
    DitherMethod,
    ExportFormat,
    PaletteMethod,
    QuantizerMethod,
)

REFERENCE_IMAGE = Path(__file__).parent / "fixtures" / "reference.png"

STAGE_FUNCTIONS = (
    "prepare_image",
    "choose_palette",
    "compute_edges",
    "render_indexed",
    "build_metrics",
)
TIMING_KEYS = ("prepare", "palette", "edges", "render", "metrics")
ARRAY_FIELDS = ("indexed", "palette_rgb", "source_resized", "source")

# Representative configurations for the bit-exact comparison against `convert_image`.
MATRIX = (
    {"dither": DitherMethod.NONE},
    {"dither": DitherMethod.NONE, "palette": PaletteMethod.MEDIAN_CUT},
    {
        "dither": DitherMethod.NONE,
        "palette": PaletteMethod.MEDIAN_CUT,
        "quantizer": QuantizerMethod.ADAPTIVE,
    },
    {
        "dither": DitherMethod.NONE,
        "palette": PaletteMethod.KMEANS,
        "quantizer": QuantizerMethod.PERCEPTUAL,
    },
    {"dither": DitherMethod.NONE, "palette": PaletteMethod.PERCEPTUAL, "colors": 2},
    {"dither": DitherMethod.NONE, "palette": PaletteMethod.MEDIAN_CUT, "width": 160, "height": 96},
    {"dither": DitherMethod.BAYER2},
    {"dither": DitherMethod.BAYER4, "palette": PaletteMethod.MEDIAN_CUT},
    {"dither": DitherMethod.FLOYD},
    {"dither": DitherMethod.ATKINSON, "palette": PaletteMethod.MEDIAN_CUT},
    {"dither": DitherMethod.ADAPTIVE},
    {"dither": DitherMethod.BAYER2, "sharpen": True, "sharpen_strength": 1.5},
)


@pytest.fixture()
def engine() -> ConversionEngine:
    instance = ConversionEngine()
    instance.set_source(REFERENCE_IMAGE)
    return instance


@pytest.fixture()
def stage_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count how often the engine actually calls each stage function."""
    calls = dict.fromkeys(STAGE_FUNCTIONS, 0)
    for name in STAGE_FUNCTIONS:
        original = getattr(engine_module, name)

        def wrapper(*args, __name=name, __original=original, **kwargs):
            calls[__name] += 1
            return __original(*args, **kwargs)

        monkeypatch.setattr(engine_module, name, wrapper)
    return calls


def _reset(calls: dict[str, int]) -> None:
    for name in calls:
        calls[name] = 0


def _cancel_after(limit: int):
    """Cancel callable that reports True from its ``limit``-th invocation on."""
    state = {"calls": 0}

    def cancel() -> bool:
        state["calls"] += 1
        return state["calls"] >= limit

    return cancel


def _arrays(engine: ConversionEngine, config: ConversionConfig) -> dict[str, np.ndarray]:
    """Convert once and return every array the engine hands to a caller."""
    result = engine.convert(config)
    source = engine.source
    assert source is not None
    return {
        "indexed": result.indexed,
        "palette_rgb": result.palette_rgb,
        "source_resized": result.source_resized,
        "source": source,
    }


@pytest.mark.parametrize("overrides", MATRIX)
def test_engine_matches_convert_image(engine: ConversionEngine, overrides: dict) -> None:
    config = ConversionConfig(metrics=True, **overrides)

    result = engine.convert(config)
    expected = convert_image(REFERENCE_IMAGE, config)

    np.testing.assert_array_equal(result.indexed, expected.indexed)
    assert result.palette_indices == expected.palette_indices
    np.testing.assert_array_equal(result.palette_rgb, expected.palette_rgb)
    np.testing.assert_array_equal(result.source_resized, expected.source_resized)
    assert result.metrics == expected.metrics
    assert tuple(result.timings) == TIMING_KEYS


def test_engine_without_metrics_matches_convert_image(engine: ConversionEngine) -> None:
    config = ConversionConfig(dither=DitherMethod.BAYER2)

    result = engine.convert(config)
    expected = convert_image(REFERENCE_IMAGE, config)

    assert result.metrics is None
    np.testing.assert_array_equal(result.indexed, expected.indexed)


# --- cache behaviour ----------------------------------------------------


def test_second_conversion_hits_every_cache(
    engine: ConversionEngine, stage_calls: dict[str, int]
) -> None:
    config = ConversionConfig(dither=DitherMethod.BAYER2, metrics=True)
    engine.convert(config)

    _reset(stage_calls)
    engine.convert(config)

    assert stage_calls == dict.fromkeys(STAGE_FUNCTIONS, 0)


def test_changing_dither_reuses_prepared_image_and_palette(
    engine: ConversionEngine, stage_calls: dict[str, int]
) -> None:
    base = ConversionConfig(dither=DitherMethod.BAYER2, palette=PaletteMethod.MEDIAN_CUT)
    engine.convert(base)

    _reset(stage_calls)
    engine.convert(replace(base, dither=DitherMethod.BAYER4))

    assert stage_calls["prepare_image"] == 0
    assert stage_calls["compute_edges"] == 0
    assert stage_calls["choose_palette"] == 0
    assert stage_calls["render_indexed"] == 1


def test_changing_quantizer_while_dithering_is_free(
    engine: ConversionEngine, stage_calls: dict[str, int]
) -> None:
    base = ConversionConfig(dither=DitherMethod.BAYER2, palette=PaletteMethod.MEDIAN_CUT)
    engine.convert(base)

    _reset(stage_calls)
    engine.convert(replace(base, quantizer=QuantizerMethod.ADAPTIVE))

    assert sum(stage_calls.values()) == 0


def test_changing_palette_recomputes_palette_only(
    engine: ConversionEngine, stage_calls: dict[str, int]
) -> None:
    base = ConversionConfig(dither=DitherMethod.NONE, palette=PaletteMethod.MEDIAN_CUT, colors=4)
    engine.convert(base)

    _reset(stage_calls)
    engine.convert(replace(base, colors=3))

    assert stage_calls["prepare_image"] == 0
    assert stage_calls["compute_edges"] == 0
    assert stage_calls["choose_palette"] == 1
    assert stage_calls["render_indexed"] == 1


def test_changing_size_recomputes_prepared_image(
    engine: ConversionEngine,
    stage_calls: dict[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = ConversionConfig(dither=DitherMethod.NONE, palette=PaletteMethod.MEDIAN_CUT)
    engine.convert(base)

    resizes = {"calls": 0}
    original_resize = pipeline_module.resize_image

    def counting_resize(*args, **kwargs):
        resizes["calls"] += 1
        return original_resize(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "resize_image", counting_resize)

    _reset(stage_calls)
    engine.convert(replace(base, height=96))

    assert resizes["calls"] == 1
    assert stage_calls["prepare_image"] == 1


def test_export_only_settings_are_free(
    engine: ConversionEngine, stage_calls: dict[str, int]
) -> None:
    base = ConversionConfig(
        dither=DitherMethod.BAYER2,
        palette=PaletteMethod.MEDIAN_CUT,
        metrics=True,
    )
    engine.convert(base)

    _reset(stage_calls)
    engine.convert(
        replace(
            base,
            mode=AtariMode.ANTIC_E,
            export_formats=[ExportFormat.PNG],
            atari_colors=[0, 4, 10, 14],
            label_name="SCREEN_DEMO",
            preview=True,
        )
    )

    assert sum(stage_calls.values()) == 0


def test_lru_evicts_the_oldest_result(stage_calls: dict[str, int]) -> None:
    engine = ConversionEngine(max_results=1)
    engine.set_source(REFERENCE_IMAGE)
    first = ConversionConfig(dither=DitherMethod.NONE, palette=PaletteMethod.MEDIAN_CUT, colors=4)
    second = replace(first, dither=DitherMethod.BAYER2)

    engine.convert(first)
    engine.convert(second)

    _reset(stage_calls)
    engine.convert(first)

    assert stage_calls["render_indexed"] == 1


def test_l3_hit_and_correct_result_when_l1_was_evicted(stage_calls: dict[str, int]) -> None:
    engine = ConversionEngine()
    engine.set_source(REFERENCE_IMAGE)
    base = ConversionConfig(dither=DitherMethod.NONE, palette=PaletteMethod.MEDIAN_CUT)
    widths = [160, 158, 156, 154, 152]

    for width in widths:
        engine.convert(replace(base, width=width))

    _reset(stage_calls)
    result = engine.convert(base)

    assert stage_calls["prepare_image"] == 1  # the L1 entry for 160 was evicted
    assert stage_calls["render_indexed"] == 0  # ... but the L3 entry survived
    fresh = ConversionEngine()
    fresh.set_source(REFERENCE_IMAGE)
    np.testing.assert_array_equal(result.indexed, fresh.convert(base).indexed)


def test_max_results_must_be_positive() -> None:
    with pytest.raises(ValueError):
        ConversionEngine(max_results=0)


def test_two_engines_have_independent_caches(stage_calls: dict[str, int]) -> None:
    config = ConversionConfig(dither=DitherMethod.BAYER2)
    first = ConversionEngine()
    first.set_source(REFERENCE_IMAGE)
    first.convert(config)

    _reset(stage_calls)
    second = ConversionEngine()
    second.set_source(REFERENCE_IMAGE)
    second.convert(config)

    assert stage_calls["prepare_image"] == 1


def test_set_source_invalidates_derived_caches(
    engine: ConversionEngine, stage_calls: dict[str, int]
) -> None:
    config = ConversionConfig(dither=DitherMethod.BAYER2, metrics=True)
    engine.convert(config)

    engine.set_source(REFERENCE_IMAGE)

    _reset(stage_calls)
    engine.convert(config)

    assert stage_calls["prepare_image"] == 1
    assert stage_calls["build_metrics"] == 1


def test_clear_cache_keeps_the_source(
    engine: ConversionEngine, stage_calls: dict[str, int]
) -> None:
    config = ConversionConfig(dither=DitherMethod.BAYER2)
    engine.convert(config)

    engine.clear_cache()

    assert engine.source is not None
    _reset(stage_calls)
    engine.convert(config)
    assert stage_calls["prepare_image"] == 1


def test_convert_without_source_raises() -> None:
    with pytest.raises(RuntimeError):
        ConversionEngine().convert(ConversionConfig())


# --- cancellation -------------------------------------------------------


def test_cancel_at_entry_on_cold_cache(
    engine: ConversionEngine, stage_calls: dict[str, int]
) -> None:
    config = ConversionConfig(dither=DitherMethod.BAYER2, palette=PaletteMethod.MEDIAN_CUT)

    with pytest.raises(ConversionCancelled):
        engine.convert(config, cancel=_cancel_after(1))

    assert sum(stage_calls.values()) == 0

    _reset(stage_calls)
    engine.convert(config)
    assert stage_calls["prepare_image"] == 1


def test_cancel_at_entry_on_warm_cache(
    engine: ConversionEngine, stage_calls: dict[str, int]
) -> None:
    config = ConversionConfig(dither=DitherMethod.BAYER2, palette=PaletteMethod.MEDIAN_CUT)
    first = engine.convert(config)

    _reset(stage_calls)
    with pytest.raises(ConversionCancelled):
        engine.convert(config, cancel=_cancel_after(1))
    assert sum(stage_calls.values()) == 0

    second = engine.convert(config)
    assert sum(stage_calls.values()) == 0
    np.testing.assert_array_equal(second.indexed, first.indexed)


@pytest.mark.parametrize(
    ("boundary", "expected_calls"),
    [
        (
            2,
            {
                "prepare_image": 0,
                "compute_edges": 0,
                "choose_palette": 1,
                "render_indexed": 1,
                "build_metrics": 1,
            },
        ),
        (
            3,
            {
                "prepare_image": 0,
                "compute_edges": 0,
                "choose_palette": 0,
                "render_indexed": 1,
                "build_metrics": 1,
            },
        ),
        (
            4,
            {
                "prepare_image": 0,
                "compute_edges": 0,
                "choose_palette": 0,
                "render_indexed": 0,
                "build_metrics": 1,
            },
        ),
        (
            5,
            {
                "prepare_image": 0,
                "compute_edges": 0,
                "choose_palette": 0,
                "render_indexed": 0,
                "build_metrics": 1,
            },
        ),
    ],
)
def test_cancel_after_a_stage_keeps_the_earlier_work(
    engine: ConversionEngine,
    stage_calls: dict[str, int],
    boundary: int,
    expected_calls: dict[str, int],
) -> None:
    config = ConversionConfig(
        dither=DitherMethod.BAYER2,
        palette=PaletteMethod.MEDIAN_CUT,
        metrics=True,
    )

    with pytest.raises(ConversionCancelled):
        engine.convert(config, cancel=_cancel_after(boundary))

    _reset(stage_calls)
    result = engine.convert(config)

    assert stage_calls == expected_calls

    fresh = ConversionEngine()
    fresh.set_source(REFERENCE_IMAGE)
    expected = fresh.convert(config)
    np.testing.assert_array_equal(result.indexed, expected.indexed)
    assert result.metrics == expected.metrics


def test_cancel_callable_exception_propagates(engine: ConversionEngine) -> None:
    def exploding_cancel() -> bool:
        raise RuntimeError("cancel probe failed")

    with pytest.raises(RuntimeError, match="cancel probe failed"):
        engine.convert(ConversionConfig(dither=DitherMethod.BAYER2), cancel=exploding_cancel)


# --- immutability -------------------------------------------------------


def test_freeze_preserves_values_dtype_and_shape() -> None:
    array = np.arange(12, dtype=np.uint8).reshape(3, 4)
    frozen = engine_module._freeze_array(array)

    assert frozen.dtype == array.dtype
    assert frozen.shape == array.shape
    np.testing.assert_array_equal(frozen, array)
    assert frozen.flags.writeable is False


@pytest.mark.parametrize(
    "array",
    [
        pytest.param(np.array([object()], dtype=object), id="plain-object"),
        pytest.param(
            np.zeros(2, dtype=np.dtype([("value", object), ("index", np.uint8)])),
            id="structured-with-object",
        ),
        pytest.param(
            np.zeros(2, dtype=np.dtype([("nested", [("inner", object)])])),
            id="nested-structured-with-object",
        ),
    ],
)
def test_freeze_rejects_object_dtype(array: np.ndarray) -> None:
    with pytest.raises(TypeError):
        engine_module._freeze_array(array)


def test_freeze_accepts_a_structured_dtype_without_objects() -> None:
    array = np.zeros(3, dtype=np.dtype([("x", np.uint8), ("y", "<u2")]))

    frozen = engine_module._freeze_array(array)

    assert frozen.dtype == array.dtype
    np.testing.assert_array_equal(frozen, array)
    assert frozen.flags.writeable is False


@pytest.mark.parametrize("name", ARRAY_FIELDS)
def test_returned_arrays_are_read_only(engine: ConversionEngine, name: str) -> None:
    array = _arrays(
        engine, ConversionConfig(dither=DitherMethod.NONE, palette=PaletteMethod.MEDIAN_CUT)
    )[name]

    assert array.flags.writeable is False
    with pytest.raises(ValueError):
        array[0] = 0


@pytest.mark.parametrize("name", ARRAY_FIELDS)
def test_setflags_write_true_is_refused(engine: ConversionEngine, name: str) -> None:
    array = _arrays(
        engine, ConversionConfig(dither=DitherMethod.NONE, palette=PaletteMethod.MEDIAN_CUT)
    )[name]

    with pytest.raises(ValueError):
        array.setflags(write=True)
    assert array.flags.writeable is False


@pytest.mark.parametrize("name", ARRAY_FIELDS)
def test_base_escalation_is_refused(engine: ConversionEngine, name: str) -> None:
    array = _arrays(
        engine, ConversionConfig(dither=DitherMethod.NONE, palette=PaletteMethod.MEDIAN_CUT)
    )[name]

    assert array.flags.owndata is False
    with pytest.raises(ValueError):
        array.base.setflags(write=True)


def test_views_stay_read_only(engine: ConversionEngine) -> None:
    indexed = _arrays(
        engine, ConversionConfig(dither=DitherMethod.NONE, palette=PaletteMethod.MEDIAN_CUT)
    )["indexed"]

    for view in (indexed.reshape(-1), indexed.T, indexed[0:1]):
        assert view.flags.writeable is False
        with pytest.raises(ValueError):
            view.setflags(write=True)


def test_copy_is_writeable_and_independent(engine: ConversionEngine) -> None:
    config = ConversionConfig(dither=DitherMethod.NONE, palette=PaletteMethod.MEDIAN_CUT)
    first = engine.convert(config)

    copy = first.indexed.copy()
    assert copy.flags.writeable is True
    copy[0, 0] = 3

    second = engine.convert(config)
    np.testing.assert_array_equal(second.indexed, first.indexed)


def test_cache_survives_failed_write_attempts(engine: ConversionEngine) -> None:
    config = ConversionConfig(dither=DitherMethod.NONE, palette=PaletteMethod.MEDIAN_CUT)
    first = engine.convert(config)

    for array in (first.indexed, first.palette_rgb, first.source_resized):
        with pytest.raises(ValueError):
            array[0] = 0
        with pytest.raises(ValueError):
            array.setflags(write=True)

    fresh = ConversionEngine()
    fresh.set_source(REFERENCE_IMAGE)
    second = engine.convert(config)
    np.testing.assert_array_equal(second.indexed, fresh.convert(config).indexed)


def test_set_source_array_copies_and_freezes() -> None:
    original = np.zeros((24, 32, 3), dtype=np.uint8)
    original[:, :, 1] = 60

    engine = ConversionEngine()
    engine.set_source_array(original)
    expected_engine = ConversionEngine()
    expected_engine.set_source_array(original.copy())

    assert engine.source is not None
    assert engine.source.flags.writeable is False

    original[:, :, 1] = 250  # the engine must not see this

    config = ConversionConfig(
        width=32,
        height=24,
        dither=DitherMethod.NONE,
        palette=PaletteMethod.MEDIAN_CUT,
    )
    np.testing.assert_array_equal(
        engine.convert(config).source_resized,
        expected_engine.convert(config).source_resized,
    )


def test_metadata_is_isolated(engine: ConversionEngine) -> None:
    config = ConversionConfig(
        dither=DitherMethod.NONE,
        palette=PaletteMethod.MEDIAN_CUT,
        metrics=True,
    )
    first = engine.convert(config)
    assert first.metrics is not None
    original_indices = list(first.palette_indices)
    original_mse = first.metrics.mse

    first.palette_indices.append(999)
    first.metrics.mse = -1.0

    second = engine.convert(config)
    assert second.palette_indices == original_indices
    assert second.metrics is not None
    assert second.metrics.mse == pytest.approx(original_mse)


# --- timings ------------------------------------------------------------


def test_timings_have_five_keys_in_order(engine: ConversionEngine) -> None:
    result = engine.convert(ConversionConfig(dither=DitherMethod.BAYER2, metrics=True))

    assert tuple(result.timings) == TIMING_KEYS
    assert all(isinstance(value, float) for value in result.timings.values())
    assert all(value >= 0.0 for value in result.timings.values())


def test_cold_conversion_reports_time_for_every_stage(engine: ConversionEngine) -> None:
    config = ConversionConfig(
        dither=DitherMethod.NONE,
        palette=PaletteMethod.MEDIAN_CUT,
        metrics=True,
    )
    result = engine.convert(config)

    for key in TIMING_KEYS:
        assert result.timings[key] > 0.0, key


def test_warm_conversion_reports_zero_for_every_stage(engine: ConversionEngine) -> None:
    config = ConversionConfig(
        dither=DitherMethod.NONE,
        palette=PaletteMethod.MEDIAN_CUT,
        metrics=True,
    )
    engine.convert(config)

    result = engine.convert(config)

    assert result.timings == dict.fromkeys(TIMING_KEYS, 0.0)


def test_metrics_disabled_keeps_the_key_at_zero(engine: ConversionEngine) -> None:
    result = engine.convert(ConversionConfig(dither=DitherMethod.BAYER2, metrics=False))

    assert result.metrics is None
    assert tuple(result.timings) == TIMING_KEYS
    assert result.timings["metrics"] == 0.0


def test_compute_metrics_override(engine: ConversionEngine) -> None:
    base = ConversionConfig(dither=DitherMethod.NONE, palette=PaletteMethod.MEDIAN_CUT)

    forced = engine.convert(base, compute_metrics=True)
    assert forced.metrics is not None
    assert forced.timings["metrics"] > 0.0

    disabled = engine.convert(base, compute_metrics=False)
    assert disabled.metrics is None
    assert disabled.timings["metrics"] == 0.0

    cached_metrics = engine.convert(base, compute_metrics=True)
    assert cached_metrics.metrics is not None
    assert cached_metrics.timings["metrics"] == 0.0


def test_timings_sum_does_not_exceed_the_total_time(engine: ConversionEngine) -> None:
    config = ConversionConfig(
        dither=DitherMethod.NONE,
        palette=PaletteMethod.MEDIAN_CUT,
        metrics=True,
    )

    started = time.perf_counter()
    result = engine.convert(config)
    elapsed = time.perf_counter() - started

    assert sum(result.timings.values()) <= elapsed + 1e-6


def test_palette_timing_covers_palette_rgb(
    engine: ConversionEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    delay = 0.05
    original_palette_rgb = engine_module.palette_rgb

    def slow_palette_rgb(indices: list[int]) -> np.ndarray:
        time.sleep(delay)
        return original_palette_rgb(indices)

    monkeypatch.setattr(engine_module, "palette_rgb", slow_palette_rgb)
    config = ConversionConfig(
        width=64,
        height=48,
        dither=DitherMethod.NONE,
        palette=PaletteMethod.MEDIAN_CUT,
    )

    result = engine.convert(config)

    assert result.timings["palette"] >= delay


def test_stage_timings_exclude_cache_bookkeeping(
    engine: ConversionEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    delay = 0.1
    original_freeze = engine_module._freeze_array
    freezes = {"count": 0}

    def slow_freeze(array: np.ndarray) -> np.ndarray:
        freezes["count"] += 1
        time.sleep(delay)
        return original_freeze(array)

    monkeypatch.setattr(engine_module, "_freeze_array", slow_freeze)
    config = ConversionConfig(
        width=64,
        height=48,
        dither=DitherMethod.NONE,
        palette=PaletteMethod.MEDIAN_CUT,
        metrics=True,
    )

    result = engine.convert(config)

    assert freezes["count"] >= 4  # prepare, edges, palette_rgb, indexed
    for key in ("prepare", "palette", "edges", "render"):
        assert result.timings[key] < delay, key
