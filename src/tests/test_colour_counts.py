"""Colour-count capability audit (plan task T1.7).

The audit proves that the pipeline is correct for N in {2, 3, 4, 8, 16} — every palette selector,
every quantizer and every dithering algorithm — and pins the real limits of each exporter by
*running* the exporter rather than restating its conditions.

Two distinctions matter throughout:

* **declared vs used colours** — ``colors=N`` guarantees a palette of exactly N unique indices, but
  an algorithm may use fewer of them (``adaptive`` classifies 16x16 tiles and can stay below N);
* **technical vs product limits** — PNG itself supports 256 colours; the GUI cap of 16 is a
  product decision documented in the implementation plan (§3.4), not a format limit.

Every rendering here uses a 64x48 crop of ``fixtures/reference.png`` (17x cheaper than 160x192);
exporter tests use the real shapes their formats require.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from converter import atari_palette
from converter.exporters.asm_export import export_asm
from converter.exporters.atari_bin_export import export_bin
from converter.exporters.atari_dta_export import export_dta
from converter.exporters.png_export import export_png
from converter.exporters.xex_export import export_xex
from converter.image_loader import load_image
from converter.metrics import compute_metrics
from converter.pipeline import (
    build_metrics,
    choose_palette,
    compute_edges,
    convert_image,
    prepare_image,
    render_indexed,
)
from converter.preview import error_heatmap
from converter.quantizer import indexed_to_rgb
from converter.types import (
    ConversionConfig,
    ConversionResult,
    DitherMethod,
    PaletteMethod,
    QuantizerMethod,
)

REFERENCE_IMAGE = Path(__file__).parent / "fixtures" / "reference.png"
N_VALUES = (2, 3, 4, 8, 16)
ALL_PALETTES = tuple(PaletteMethod)
ALL_QUANTIZERS = tuple(QuantizerMethod)
ALL_DITHERS = tuple(d for d in DitherMethod if d is not DitherMethod.NONE)
DITHERS_WITHOUT_ADAPTIVE = tuple(d for d in ALL_DITHERS if d is not DitherMethod.ADAPTIVE)
BIN_EXPORTERS = (("bin", export_bin), ("asm", export_asm), ("dta", export_dta))


def _config(n: int, **overrides: object) -> ConversionConfig:
    """Small-image conversion config with the requested number of colours."""
    base = ConversionConfig(
        width=64,
        height=48,
        colors=n,
        palette=PaletteMethod.MEDIAN_CUT,
        dither=DitherMethod.NONE,
    )
    return replace(base, **overrides) if overrides else base


@pytest.fixture(scope="module")
def prepared() -> np.ndarray:
    """The 64x48 prepared image shared by every pipeline test."""
    return prepare_image(load_image(REFERENCE_IMAGE).data, _config(4))


@pytest.fixture(scope="module")
def edges(prepared: np.ndarray) -> np.ndarray:
    """Edge mask of the shared prepared image."""
    return compute_edges(prepared)


@pytest.fixture(scope="module")
def dither_cache(prepared: np.ndarray, edges: np.ndarray) -> dict:
    """Render every (N, dither) combination once and share it between the assertions."""
    cache: dict = {}
    for n in N_VALUES:
        palette_indices = choose_palette(prepared, _config(n))
        for dither in ALL_DITHERS:
            config = _config(n, dither=dither)
            cache[(n, dither)] = render_indexed(prepared, palette_indices, edges, config)
    return cache


@pytest.fixture(scope="module")
def quantizer_cache(prepared: np.ndarray, edges: np.ndarray) -> dict:
    """Render every (N, quantizer) combination once (``dither=none``)."""
    cache: dict = {}
    for n in N_VALUES:
        palette_indices = choose_palette(prepared, _config(n))
        for quantizer in ALL_QUANTIZERS:
            config = _config(n, quantizer=quantizer)
            cache[(n, quantizer)] = render_indexed(prepared, palette_indices, edges, config)
    return cache


# --- palette selection --------------------------------------------------


@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("method", ALL_PALETTES)
def test_palette_has_exactly_n_unique_indices(
    prepared: np.ndarray, method: PaletteMethod, n: int
) -> None:
    config = _config(n, palette=method, background_index=6)

    palette_indices = choose_palette(prepared, config)

    assert len(palette_indices) == n
    assert len(set(palette_indices)) == n
    assert all(0 <= index <= 255 for index in palette_indices)
    assert palette_indices[0] == 6


def test_palette_is_filled_even_for_a_flat_image() -> None:
    """A flat image cannot supply N distinct colours, so ``_fill_palette`` tops the palette up."""
    flat = np.full((24, 32, 3), 90, dtype=np.uint8)

    palette_indices = choose_palette(flat, _config(16))

    assert len(palette_indices) == 16
    assert len(set(palette_indices)) == 16


def test_palette_request_above_256_is_capped_by_the_pipeline(prepared: np.ndarray) -> None:
    """Low-level pipeline behaviour **without** ``validate_config`` - not a valid configuration.

    ``validate_config`` restricts ``colors`` to 2..256, but the pipeline itself silently caps larger
    requests at the 256 entries of the hardware palette.
    """
    palette_indices = choose_palette(prepared, _config(300))

    assert len(palette_indices) == 256
    assert len(set(palette_indices)) == 256


# --- quantizers (dither = none) -----------------------------------------


@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("quantizer", ALL_QUANTIZERS)
def test_quantizer_index_range(quantizer_cache: dict, quantizer: QuantizerMethod, n: int) -> None:
    indexed = quantizer_cache[(n, quantizer)]

    assert indexed.dtype == np.uint8
    assert int(indexed.min()) >= 0
    assert int(indexed.max()) <= n - 1


@pytest.mark.parametrize("n", N_VALUES)
def test_quantizer_reaches_the_last_palette_entry(quantizer_cache: dict, n: int) -> None:
    indexed = quantizer_cache[(n, QuantizerMethod.NEAREST)]

    assert int(indexed.max()) == n - 1


# --- dithering (all 17 algorithms) --------------------------------------


@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("dither", ALL_DITHERS)
def test_dither_index_range(dither_cache: dict, dither: DitherMethod, n: int) -> None:
    indexed = dither_cache[(n, dither)]

    assert indexed.dtype == np.uint8
    assert int(indexed.min()) >= 0
    assert int(indexed.max()) <= n - 1


@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("dither", DITHERS_WITHOUT_ADAPTIVE)
def test_dither_reaches_the_last_palette_entry(
    dither_cache: dict, dither: DitherMethod, n: int
) -> None:
    indexed = dither_cache[(n, dither)]

    assert int(indexed.max()) == n - 1


@pytest.mark.parametrize(
    ("n", "dither"),
    [(8, dither) for dither in DITHERS_WITHOUT_ADAPTIVE] + [(16, dither) for dither in ALL_DITHERS],
)
def test_dither_uses_colours_above_the_atari_limit(
    dither_cache: dict, dither: DitherMethod, n: int
) -> None:
    """Regression guard against a hard-coded 0..3 clamp anywhere in the dithering path.

    At N=8 ``adaptive`` is excluded (it classifies tiles and stays within indices 0-3); at N=16 all
    seventeen algorithms must actually use indices >= 4 somewhere in the image.
    """
    indexed = dither_cache[(n, dither)]

    assert int(np.count_nonzero(indexed >= 4)) > 0


def test_adaptive_dither_may_use_fewer_colours_than_requested(dither_cache: dict) -> None:
    """`adaptive` picks a dither per 16x16 tile, so it can stay well below N used colours.

    Measured on this fixture: at N=8 it never leaves indices 0-3, while at N=16 it reaches index 15
    but only for a small part of the image (175 of 3072 pixels).
    """
    eight = dither_cache[(8, DitherMethod.ADAPTIVE)]
    assert int(eight.max()) <= 7
    assert int(np.count_nonzero(eight >= 4)) == 0

    sixteen = dither_cache[(16, DitherMethod.ADAPTIVE)]
    assert int(sixteen.max()) <= 15
    above = int(np.count_nonzero(sixteen >= 4))
    assert 0 < above < sixteen.size


# --- metrics, heatmap and RGB reconstruction ----------------------------


@pytest.mark.parametrize("n", N_VALUES)
def test_metrics_are_consistent_with_the_indexed_image(
    prepared: np.ndarray, edges: np.ndarray, n: int
) -> None:
    palette_indices = choose_palette(prepared, _config(n))
    config = _config(n, dither=DitherMethod.BAYER2)
    indexed = render_indexed(prepared, palette_indices, edges, config)

    metrics = build_metrics(prepared, indexed, palette_indices)
    manual = compute_metrics(prepared, indexed, palette_indices)

    assert metrics == manual
    assert metrics.colors_used == len(np.unique(indexed))
    assert metrics.colors_used <= n
    assert metrics.mse >= 0.0
    assert 0.0 <= metrics.ssim <= 1.0
    assert 0.0 <= metrics.histogram_coverage <= 1.0


def test_mse_regression_control_for_the_reference_fixture(
    prepared: np.ndarray, edges: np.ndarray
) -> None:
    """Fixture-specific regression control - **not** a general MSE monotonicity guarantee.

    For this fixture (64x48 crop of ``fixtures/reference.png``) with ``median_cut`` and ``bayer2``
    the measured MSE is about 2823 for N=2 and 570 for N=16. Pinning the relationship catches a
    palette/index mix-up that flattens or inverts the curve; the exact values are not asserted
    because a different fixture or dither would legitimately produce different numbers.
    """

    def mse_for(n: int) -> float:
        palette_indices = choose_palette(prepared, _config(n))
        config = _config(n, dither=DitherMethod.BAYER2)
        indexed = render_indexed(prepared, palette_indices, edges, config)
        return build_metrics(prepared, indexed, palette_indices).mse

    assert mse_for(2) > mse_for(16)


@pytest.mark.parametrize("n", N_VALUES)
def test_heatmap_works_for_every_n(prepared: np.ndarray, edges: np.ndarray, n: int) -> None:
    palette_indices = choose_palette(prepared, _config(n))
    config = _config(n, dither=DitherMethod.BAYER2)
    indexed = render_indexed(prepared, palette_indices, edges, config)

    heatmap = error_heatmap(prepared, indexed_to_rgb(indexed, palette_indices))

    assert heatmap.shape == (prepared.shape[0], prepared.shape[1], 3)
    assert heatmap.dtype == np.uint8


@pytest.mark.parametrize("n", N_VALUES)
def test_converted_rgb_uses_the_selected_palette(
    prepared: np.ndarray, edges: np.ndarray, n: int
) -> None:
    palette_indices = choose_palette(prepared, _config(n))
    config = _config(n, dither=DitherMethod.BAYER2)
    indexed = render_indexed(prepared, palette_indices, edges, config)

    converted = indexed_to_rgb(indexed, palette_indices)
    palette_table = np.array(
        [atari_palette.get_color(i).rgb for i in palette_indices], dtype=np.uint8
    )

    np.testing.assert_array_equal(converted, palette_table[indexed])
    assert set(map(tuple, converted.reshape(-1, 3).tolist())) <= set(
        map(tuple, palette_table.tolist())
    )


@pytest.mark.parametrize("dither", [DitherMethod.FLOYD, DitherMethod.BAYER4, DitherMethod.ADAPTIVE])
def test_results_are_deterministic(
    prepared: np.ndarray, edges: np.ndarray, dither: DitherMethod
) -> None:
    config = _config(8, dither=dither)

    first_palette = choose_palette(prepared, config)
    first = render_indexed(prepared, first_palette, edges, config)
    second_palette = choose_palette(prepared, config)
    second = render_indexed(prepared, second_palette, edges, config)

    assert first_palette == second_palette
    np.testing.assert_array_equal(first, second)
    assert build_metrics(prepared, first, first_palette) == build_metrics(
        prepared, second, second_palette
    )


@pytest.mark.parametrize("n", N_VALUES)
def test_convert_image_end_to_end(n: int) -> None:
    config = ConversionConfig(
        width=64,
        height=48,
        colors=n,
        palette=PaletteMethod.MEDIAN_CUT,
        dither=DitherMethod.BAYER2,
        metrics=True,
    )

    result = convert_image(REFERENCE_IMAGE, config)

    assert len(result.palette_indices) == n
    assert len(set(result.palette_indices)) == n
    assert result.indexed.shape == (48, 64)
    assert int(result.indexed.min()) >= 0
    assert int(result.indexed.max()) <= n - 1
    assert result.metrics is not None
    assert result.metrics.colors_used <= n
    assert result.converted_rgb.shape == (48, 64, 3)


# --- exporters: verified limits -----------------------------------------


def _direct_result(indexed: np.ndarray, palette_size: int) -> ConversionResult:
    """Hand-built result for exporter-limit checks (no pipeline run, no k-means)."""
    return ConversionResult(
        indexed=indexed,
        palette_indices=list(range(palette_size)),
        palette_rgb=np.array(atari_palette.PALETTE_RGB[:palette_size], dtype=np.uint8),
        source_resized=np.zeros((*indexed.shape, 3), dtype=np.uint8),
    )


@pytest.mark.parametrize("palette_size", [2, 16, 256])
def test_png_supports_up_to_256_colours(tmp_path: Path, palette_size: int) -> None:
    """PNG's technical limit is 256 colours; 16 is only the product cap (plan §3.4)."""
    indexed = np.zeros((4, 8), dtype=np.uint8)
    indexed[:, 4:] = palette_size - 1
    result = _direct_result(indexed, palette_size)

    path = tmp_path / f"palette_{palette_size}.png"
    export_png(result.indexed, result.palette_rgb, path)

    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_png_rejects_more_than_256_colours(tmp_path: Path) -> None:
    """Documents the real PNG limit; the pipeline itself never exceeds 256 entries."""
    indexed = np.zeros((4, 8), dtype=np.uint8)
    oversized_palette = np.zeros((257, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        export_png(indexed, oversized_palette, tmp_path / "too_many.png")


@pytest.mark.parametrize(("name", "exporter"), BIN_EXPORTERS)
@pytest.mark.parametrize("height", [190, 192])
@pytest.mark.parametrize("colors", [2, 4])
def test_atari_bin_formats_export_both_supported_heights(
    tmp_path: Path, name: str, exporter: Callable[..., None], height: int, colors: int
) -> None:
    indexed = np.zeros((height, 160), dtype=np.uint8)
    indexed[0, 0] = colors - 1

    path = tmp_path / f"screen_{name}_{height}_{colors}"
    exporter(indexed, path)

    assert path.read_bytes()


@pytest.mark.parametrize(("name", "exporter"), BIN_EXPORTERS)
@pytest.mark.parametrize("colors", [8, 16])
def test_atari_bin_formats_reject_more_than_four_colours(
    tmp_path: Path, name: str, exporter: Callable[..., None], colors: int
) -> None:
    indexed = np.zeros((192, 160), dtype=np.uint8)
    indexed[0, 0] = colors - 1

    with pytest.raises(ValueError, match="0-3"):
        exporter(indexed, tmp_path / f"screen_{name}_{colors}")


@pytest.mark.parametrize(("name", "exporter"), BIN_EXPORTERS)
@pytest.mark.parametrize("height", [96, 100, 191])
def test_atari_bin_formats_reject_unsupported_heights(
    tmp_path: Path, name: str, exporter: Callable[..., None], height: int
) -> None:
    indexed = np.zeros((height, 160), dtype=np.uint8)

    with pytest.raises(ValueError, match="160x190"):
        exporter(indexed, tmp_path / f"screen_{name}_{height}")


@pytest.mark.parametrize("antic_mode", ["D", "E"])
@pytest.mark.parametrize("colors", [2, 4])
def test_xex_2bpp_modes_export_up_to_four_colours(
    tmp_path: Path, antic_mode: str, colors: int
) -> None:
    indexed = np.zeros((192, 160), dtype=np.uint8)
    indexed[0, 0] = colors - 1

    path = tmp_path / f"screen_{antic_mode}_{colors}.xex"
    export_xex(indexed, path, antic_mode=antic_mode)

    assert path.read_bytes()[:2] == b"\xff\xff"


@pytest.mark.parametrize("antic_mode", ["D", "E"])
def test_xex_2bpp_modes_reject_more_than_four_colours(tmp_path: Path, antic_mode: str) -> None:
    indexed = np.zeros((192, 160), dtype=np.uint8)
    indexed[0, 0] = 7

    with pytest.raises(ValueError, match="0-3"):
        export_xex(indexed, tmp_path / f"screen_{antic_mode}.xex", antic_mode=antic_mode)


def test_xex_1bpp_mode_accepts_two_colours(tmp_path: Path) -> None:
    indexed = np.zeros((192, 320), dtype=np.uint8)
    indexed[:, ::2] = 1

    path = tmp_path / "screen_f.xex"
    export_xex(indexed, path, antic_mode="F")

    assert path.read_bytes()[:2] == b"\xff\xff"


def test_xex_1bpp_mode_rejects_more_than_two_colours(tmp_path: Path) -> None:
    indexed = np.zeros((192, 320), dtype=np.uint8)
    indexed[0, 0] = 3

    with pytest.raises(ValueError, match="0-1"):
        export_xex(indexed, tmp_path / "screen_f.xex", antic_mode="F")


@pytest.mark.parametrize(
    ("shape", "colors", "inferred_mode"),
    [((192, 160), 4, "E"), ((96, 160), 4, "D"), ((192, 320), 2, "F")],
)
def test_xex_infers_the_mode_from_the_shape(
    tmp_path: Path, shape: tuple[int, int], colors: int, inferred_mode: str
) -> None:
    """The inferred output must be byte-identical to the explicit mode it stands for."""
    indexed = np.zeros(shape, dtype=np.uint8)
    indexed[0, 0] = colors - 1

    inferred = tmp_path / "inferred.xex"
    export_xex(indexed, inferred)
    explicit = tmp_path / f"{inferred_mode}.xex"
    export_xex(indexed, explicit, antic_mode=inferred_mode)

    assert inferred.read_bytes() == explicit.read_bytes()


def test_xex_inferred_f_rejects_more_than_two_colours(tmp_path: Path) -> None:
    indexed = np.zeros((192, 320), dtype=np.uint8)
    indexed[0, 0] = 3

    with pytest.raises(ValueError, match="0-1"):
        export_xex(indexed, tmp_path / "inferred_f.xex")


def test_xex_reports_when_the_mode_cannot_be_inferred(tmp_path: Path) -> None:
    indexed = np.zeros((100, 200), dtype=np.uint8)

    with pytest.raises(ValueError, match="Cannot infer ANTIC mode"):
        export_xex(indexed, tmp_path / "no_mode.xex")


def test_xex_rejects_an_invalid_explicit_mode(tmp_path: Path) -> None:
    indexed = np.zeros((192, 160), dtype=np.uint8)

    with pytest.raises(ValueError, match="Invalid ANTIC mode"):
        export_xex(indexed, tmp_path / "bad_mode.xex", antic_mode="X")


def test_xex_explicit_f_works_at_160_wide(tmp_path: Path) -> None:
    """Width 320 is required only for *inference*; an explicit mode F packs any width % 8."""
    indexed = np.zeros((192, 160), dtype=np.uint8)
    indexed[:, ::2] = 1

    path = tmp_path / "f_160.xex"
    export_xex(indexed, path, antic_mode="F")

    assert path.read_bytes()[:2] == b"\xff\xff"


@pytest.mark.parametrize(("antic_mode", "width"), [("D", 158), ("E", 158), ("F", 316)])
def test_xex_width_must_fit_the_packing_unit(tmp_path: Path, antic_mode: str, width: int) -> None:
    """Known defect documented by T1.7 (not fixed): a bad width raises a bare ``IndexError``.

    D and E pack four pixels per byte (``width % 4``) and F packs eight (``width % 8``); the packers
    index one element past the end of the row instead of validating the width first.
    """
    indexed = np.zeros((192, width), dtype=np.uint8)

    with pytest.raises(IndexError):
        export_xex(indexed, tmp_path / f"width_{width}.xex", antic_mode=antic_mode)


@pytest.mark.parametrize(("antic_mode", "width"), [("D", 156), ("F", 320)])
def test_xex_accepts_widths_that_fit_the_packing_unit(
    tmp_path: Path, antic_mode: str, width: int
) -> None:
    indexed = np.zeros((192, width), dtype=np.uint8)

    path = tmp_path / f"ok_{antic_mode}_{width}.xex"
    export_xex(indexed, path, antic_mode=antic_mode)

    assert path.read_bytes()[:2] == b"\xff\xff"
