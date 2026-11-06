"""Cache-integrity tests for the conversion engine (plan §8.1).

Cached NumPy arrays are treated as immutable. The engine does not merely clear the ``writeable`` flag
(which the caller can flip back with ``setflags(write=True)``): every cached array is a read-only view
over an immutable ``bytes`` buffer, so an assignment, a flag flip and a ``.base`` escalation all raise
``ValueError``. These tests check the real data the engine returns, not only the flag of one object.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import converter.engine as engine_module
from converter.engine import ConversionEngine
from converter.types import ConversionConfig, DitherMethod, PaletteMethod

REFERENCE_IMAGE = Path(__file__).parent / "fixtures" / "reference.png"

ARRAY_FIELDS = ("indexed", "palette_rgb", "source_resized", "source")


def _config() -> ConversionConfig:
    """Cheap configuration used by the behavioural integrity checks."""
    return ConversionConfig(dither=DitherMethod.NONE, palette=PaletteMethod.MEDIAN_CUT)


@pytest.fixture()
def engine() -> ConversionEngine:
    instance = ConversionEngine()
    instance.set_source(REFERENCE_IMAGE)
    return instance


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


def _cached_l1(engine: ConversionEngine, config: ConversionConfig) -> tuple:
    """The ``(prepared image, edge mask)`` tuple the engine stored in its L1 cache.

    Reaching into the cache is the only way to inspect the nested structure: the edge mask is never
    handed to a caller, which is exactly why its freezing needs its own assertion.
    """
    found, cached = engine._l1.get(engine_module._l1_key(config))
    assert found, "the conversion should have populated L1"
    return cached


# --- freezing -----------------------------------------------------------


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


def test_freeze_recurses_into_tuples() -> None:
    """§8.1: freezing is recursive, so the L1 ``(prepared, edge_mask)`` tuple is covered."""
    prepared = np.arange(6, dtype=np.uint8).reshape(2, 3)
    mask = np.linspace(0.0, 1.0, 6, dtype=np.float64).reshape(2, 3)

    frozen = engine_module._freeze((prepared, mask))

    assert isinstance(frozen, tuple) and len(frozen) == 2
    for original, item in zip((prepared, mask), frozen):
        np.testing.assert_array_equal(item, original)
        assert item.flags.writeable is False
        with pytest.raises(ValueError):
            item[0] = item[0]


def test_cached_l1_tuple_is_frozen_recursively(engine: ConversionEngine) -> None:
    """Both arrays of the cached L1 tuple are immutable, including the hidden edge mask."""
    config = _config()
    result = engine.convert(config)

    prepared, edge_mask = _cached_l1(engine, config)

    for array in (prepared, edge_mask):
        assert array.flags.writeable is False
        with pytest.raises(ValueError):
            array.flat[0] = 0
        with pytest.raises(ValueError):
            array.setflags(write=True)
    np.testing.assert_array_equal(prepared, result.source_resized)
    assert edge_mask.shape == result.indexed.shape
    assert edge_mask.dtype == np.float64


# --- reads --------------------------------------------------------------


@pytest.mark.parametrize("name", ARRAY_FIELDS)
def test_returned_arrays_are_read_only(engine: ConversionEngine, name: str) -> None:
    array = _arrays(engine, _config())[name]

    assert array.flags.writeable is False
    with pytest.raises(ValueError):
        array[0] = 0


@pytest.mark.parametrize("name", ARRAY_FIELDS)
def test_setflags_write_true_is_refused(engine: ConversionEngine, name: str) -> None:
    array = _arrays(engine, _config())[name]

    with pytest.raises(ValueError):
        array.setflags(write=True)
    assert array.flags.writeable is False


@pytest.mark.parametrize("name", ARRAY_FIELDS)
def test_base_escalation_is_refused(engine: ConversionEngine, name: str) -> None:
    array = _arrays(engine, _config())[name]

    assert array.flags.owndata is False
    with pytest.raises(ValueError):
        array.base.setflags(write=True)


def test_views_stay_read_only(engine: ConversionEngine) -> None:
    indexed = _arrays(engine, _config())["indexed"]

    for view in (indexed.reshape(-1), indexed.T, indexed[0:1]):
        assert view.flags.writeable is False
        with pytest.raises(ValueError):
            view.setflags(write=True)


def test_reads_return_the_cached_buffers_without_copying(engine: ConversionEngine) -> None:
    """§8.1: no defensive copies on read - the caller receives the cached object itself."""
    config = _config()

    result = engine.convert(config)

    l1_key = engine_module._l1_key(config)
    l2_key = l1_key + engine_module._palette_key(config)
    l3_key = l2_key + engine_module._dither_key(config)
    found_l1, cached_l1 = engine._l1.get(l1_key)
    found_l2, cached_l2 = engine._l2.get(l2_key)
    found_l3, cached_l3 = engine._l3.get(l3_key)

    assert found_l1 and found_l2 and found_l3
    assert result.source_resized is cached_l1[0]
    assert result.palette_rgb is cached_l2[1]
    assert result.indexed is cached_l3


def test_copy_is_writeable_and_independent(engine: ConversionEngine) -> None:
    config = _config()
    first = engine.convert(config)

    copy = first.indexed.copy()
    assert copy.flags.writeable is True
    copy[:] = 0

    second = engine.convert(config)
    assert second.indexed is first.indexed
    np.testing.assert_array_equal(second.indexed, first.indexed)
    assert second.indexed.any(), "the cache must still hold the converted image, not zeros"


# --- failed writes ------------------------------------------------------


def test_failed_write_attempts_leave_the_cache_intact(engine: ConversionEngine) -> None:
    """A refused write must not corrupt a cached buffer nor change a later conversion."""
    config = _config()
    first = engine.convert(config)
    before = {name: array.tobytes() for name, array in _arrays(engine, config).items()}

    for name, array in _arrays(engine, config).items():
        with pytest.raises(ValueError):
            array[0] = 0
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array.base.setflags(write=True)
        assert array.tobytes() == before[name], f"{name} changed after a refused write"

    fresh = ConversionEngine()
    fresh.set_source(REFERENCE_IMAGE)
    second = engine.convert(config)

    assert second.indexed is first.indexed, "a cache hit must return the same buffer"
    np.testing.assert_array_equal(second.indexed, fresh.convert(config).indexed)


# --- boundaries ---------------------------------------------------------


def test_set_source_array_copies_and_freezes() -> None:
    original = np.zeros((24, 32, 3), dtype=np.uint8)
    original[:, :, 1] = 60

    engine = ConversionEngine()
    engine.set_source_array(original)
    expected_engine = ConversionEngine()
    expected_engine.set_source_array(original.copy())

    assert engine.source is not None
    assert engine.source.flags.writeable is False
    with pytest.raises(ValueError):
        engine.source.setflags(write=True)

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
