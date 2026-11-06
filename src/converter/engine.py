"""Staged, cached conversion engine.

The engine wraps the behaviour-preserving stage functions from :mod:`converter.pipeline` in a
per-stage LRU cache. It is pure Python + NumPy: no Qt import, no threads and no global state, so it
can drive the GUI conversion worker (T3.1) and a future process-based backend alike.

Cached arrays are read-only views over immutable ``bytes`` buffers: a plain
``array.setflags(write=False)`` can be reverted by the caller with ``setflags(write=True)``, while a
``bytes``-backed view refuses both the assignment and the flag flip (see the plan, §8.1).
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np

from converter.image_loader import load_image
from converter.palette import palette_rgb
from converter.pipeline import (
    build_metrics,
    choose_palette,
    compute_edges,
    prepare_image,
    render_indexed,
)
from converter.types import (
    ConversionConfig,
    ConversionResult,
    DitherMethod,
    EdgeMask,
    IndexedImage,
    MetricResult,
    RGBImage,
)

_L1_CAPACITY = 4  # prepared images (plus their edge masks), plan §8
_L2_CAPACITY = 8  # palettes, plan §8

_TIMING_KEYS = ("prepare", "palette", "edges", "render", "metrics")


def _freeze_array(array: np.ndarray) -> np.ndarray:
    """Return a read-only view over an immutable ``bytes`` buffer with the same values."""
    if array.dtype.hasobject:
        raise TypeError("object arrays cannot be cached: the buffer is not safely immutable")
    # ``tobytes()`` copies once (and normalises to C order); the resulting array owns no data, so
    # neither a write nor ``setflags(write=True)`` nor a ``.base`` escalation can succeed.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _freeze(value: object) -> object:
    """Return read-only views over immutable buffers (arrays, or tuples of arrays)."""
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        return _freeze_array(value)
    return value


class _LRU:
    """Minimal LRU cache keyed by hashable tuples."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._items: OrderedDict[tuple, object] = OrderedDict()

    def get(self, key: tuple) -> tuple[bool, object]:
        """Return ``(hit, value)``, refreshing the recency of a hit."""
        if key in self._items:
            self._items.move_to_end(key)
            return True, self._items[key]
        return False, None

    def put(self, key: tuple, value: object) -> None:
        """Insert a value, evicting the least recently used entry when full."""
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self._capacity:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


def _l1_key(config: ConversionConfig) -> tuple:
    """Cache key of the prepared image and its edge mask."""
    strength = round(config.sharpen_strength, 6) if config.sharpen else None
    return (config.width, config.height, config.resize.value, config.sharpen, strength)


def _palette_key(config: ConversionConfig) -> tuple:
    """Key fragment of the palette stage (appended to the L1 key)."""
    return (config.palette.value, config.colors, config.background_index)


def _dither_key(config: ConversionConfig) -> tuple:
    """Key fragment of the render stage.

    The quantizer only matters when no dither is active, so it stays out of the key otherwise and
    changing it never invalidates a dithered result.
    """
    quantizer = config.quantizer.value if config.dither == DitherMethod.NONE else None
    return (config.dither.value, quantizer)


class ConversionCancelled(Exception):
    """Raised when a conversion was cancelled between stages."""


class ConversionEngine:
    """Cache-backed conversion front-end for a single worker thread.

    Not thread-safe by design: create one instance per worker thread (T3.1 owns it inside the
    conversion worker) and never call it from the GUI thread. Instances share no state.
    """

    def __init__(self, max_results: int = 32) -> None:
        """Create an engine.

        ``max_results`` bounds the L3 (indexed images) and L4 (metrics) caches; the prepared-image
        cache and the palette cache use the fixed capacities from the plan (4 and 8).
        """
        if max_results < 1:
            raise ValueError(f"max_results must be >= 1 (got {max_results})")
        self._source_key: tuple | None = None
        self._source: RGBImage | None = None
        self._array_token = 0
        self._l1 = _LRU(_L1_CAPACITY)
        self._l2 = _LRU(_L2_CAPACITY)
        self._l3 = _LRU(max_results)
        self._l4 = _LRU(max_results)

    def set_source(self, path: str | Path) -> None:
        """Load an image file and drop every cached stage."""
        resolved = Path(path).resolve()
        frozen = _freeze_array(load_image(resolved).data)
        stat = resolved.stat()
        self._source_key = ("path", str(resolved), stat.st_size, stat.st_mtime_ns)
        self._source = frozen
        self._clear_derived()

    def set_source_array(self, image: RGBImage) -> None:
        """Use an in-memory image; the caller's array is copied and never aliased."""
        copy = np.array(image, dtype=np.uint8, copy=True)
        self._array_token += 1
        self._source_key = ("array", self._array_token)
        self._source = _freeze_array(copy)
        self._clear_derived()

    @property
    def source(self) -> RGBImage | None:
        """The loaded source image as a read-only view (``None`` before ``set_source``)."""
        return self._source

    def clear_cache(self) -> None:
        """Drop the derived caches (L1-L4); the loaded source is kept."""
        self._clear_derived()

    def _clear_derived(self) -> None:
        for level in (self._l1, self._l2, self._l3, self._l4):
            level.clear()

    def convert(
        self,
        config: ConversionConfig,
        *,
        compute_metrics: bool | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> ConversionResult:
        """Run (or reuse) the pipeline for ``config``.

        Raises ``ConversionCancelled`` when ``cancel()`` returns ``True`` at one of the five stage
        boundaries. Completed stages stay cached, so a later retry resumes from there.

        ``timings`` measures the stage computations only: the palette stage covers both
        ``choose_palette`` and ``palette_rgb``, while cache bookkeeping (freezing, insertion) is
        never included.
        """
        if self._source is None:
            raise RuntimeError("set_source() must be called before convert()")

        wants_metrics = config.metrics if compute_metrics is None else compute_metrics
        timings: dict[str, float] = dict.fromkeys(_TIMING_KEYS, 0.0)

        self._check_cancel(cancel)  # boundary 1: before any work

        prepared: RGBImage
        edge_mask: EdgeMask
        l1_key = _l1_key(config)
        found, cached = self._l1.get(l1_key)
        if found:
            prepared, edge_mask = cached  # type: ignore[misc]
        else:
            started = time.perf_counter()
            raw_prepared = prepare_image(self._source, config)
            timings["prepare"] = time.perf_counter() - started

            started = time.perf_counter()
            raw_edges = compute_edges(raw_prepared)
            timings["edges"] = time.perf_counter() - started

            prepared, edge_mask = _freeze((raw_prepared, raw_edges))
            self._l1.put(l1_key, (prepared, edge_mask))
        self._check_cancel(cancel)  # boundary 2: prepared image and edge mask are cached

        palette_indices: tuple
        pal_rgb: RGBImage
        l2_key = l1_key + _palette_key(config)
        found, cached = self._l2.get(l2_key)
        if found:
            palette_indices, pal_rgb = cached  # type: ignore[misc]
        else:
            started = time.perf_counter()
            indices = choose_palette(prepared, config)
            raw_palette_rgb = palette_rgb(indices)
            timings["palette"] = time.perf_counter() - started

            palette_indices = tuple(indices)
            pal_rgb = _freeze_array(raw_palette_rgb)
            self._l2.put(l2_key, (palette_indices, pal_rgb))
        self._check_cancel(cancel)  # boundary 3: palette is cached

        indexed: IndexedImage
        l3_key = l2_key + _dither_key(config)
        found, cached = self._l3.get(l3_key)
        if found:
            indexed = cached  # type: ignore[assignment]
        else:
            started = time.perf_counter()
            rendered = render_indexed(prepared, list(palette_indices), edge_mask, config)
            timings["render"] = time.perf_counter() - started

            indexed = _freeze_array(rendered)
            self._l3.put(l3_key, indexed)
        self._check_cancel(cancel)  # boundary 4: indexed image is cached

        metrics: MetricResult | None = None
        if wants_metrics:
            self._check_cancel(cancel)  # boundary 5: before the metrics stage
            found, cached = self._l4.get(l3_key)
            if found:
                metrics = replace(cached)  # type: ignore[arg-type]
            else:
                started = time.perf_counter()
                computed = build_metrics(prepared, indexed, list(palette_indices))
                timings["metrics"] = time.perf_counter() - started
                self._l4.put(l3_key, computed)
                metrics = replace(computed)

        return ConversionResult(
            indexed=indexed,
            palette_indices=list(palette_indices),
            palette_rgb=pal_rgb,
            source_resized=prepared,
            metrics=metrics,
            timings=timings,
        )

    @staticmethod
    def _check_cancel(cancel: Callable[[], bool] | None) -> None:
        """Raise ``ConversionCancelled`` when the caller asked to stop."""
        if cancel is not None and cancel():
            raise ConversionCancelled()
