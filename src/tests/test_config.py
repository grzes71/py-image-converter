"""Tests for config module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from converter.config import load_config, save_config, validate_config
from converter.quantizer import indexed_to_rgb
from converter.types import (
    MAX_ATARI_COLORS,
    ConversionConfig,
    ConversionResult,
    DitherMethod,
    ExportFormat,
    PaletteMethod,
)


def _make_result(indexed: np.ndarray, palette_indices: list[int]) -> ConversionResult:
    """Build a minimal conversion result for the result-type tests."""
    return ConversionResult(
        indexed=indexed,
        palette_indices=list(palette_indices),
        palette_rgb=np.zeros((len(palette_indices), 3), dtype=np.uint8),
        source_resized=np.zeros((*indexed.shape, 3), dtype=np.uint8),
    )


def test_config_round_trip(tmp_path: Path) -> None:
    config = ConversionConfig(
        width=160,
        height=192,
        dither=DitherMethod.FLOYD,
        export_formats=[ExportFormat.BIN, ExportFormat.PNG],
    )
    path = tmp_path / "config.json"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.width == 160
    assert loaded.height == 192
    assert loaded.dither == DitherMethod.FLOYD
    assert ExportFormat.BIN in loaded.export_formats

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["mode"] == "ANTIC_E"
    assert data["dither"] == "floyd"


def test_validate_config_accepts_the_default_config() -> None:
    assert validate_config(ConversionConfig()) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 1},
        {"height": 1},
        {"colors": 2},
        {"colors": 4},
        {"colors": 256},
        {"background_index": 0},
        {"background_index": 255},
        {"width": 320, "height": 192, "colors": 16},
    ],
)
def test_validate_config_accepts_supported_values(kwargs: dict) -> None:
    assert validate_config(ConversionConfig(**kwargs)) == []


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"width": 0}, "width"),
        ({"width": -1}, "width"),
        ({"height": 0}, "height"),
        ({"height": -3}, "height"),
        ({"colors": 1}, "colors"),
        ({"colors": 0}, "colors"),
        ({"colors": 257}, "colors"),
        ({"background_index": -1}, "background_index"),
        ({"background_index": 256}, "background_index"),
    ],
)
def test_validate_config_reports_a_blocking_problem(kwargs: dict, expected: str) -> None:
    problems = validate_config(ConversionConfig(**kwargs))
    assert len(problems) == 1
    assert expected in problems[0]


def test_validate_config_reports_every_problem_in_a_stable_order() -> None:
    config = ConversionConfig(width=0, height=0, colors=1, background_index=999)
    problems = validate_config(config)

    assert len(problems) == 4
    assert problems[0].startswith("width")
    assert problems[1].startswith("height")
    assert problems[2].startswith("colors")
    assert problems[3].startswith("background_index")
    assert validate_config(config) == problems


def test_validate_config_does_not_modify_the_config() -> None:
    config = ConversionConfig(width=0, colors=1, background_index=999)
    before = config.to_dict()
    validate_config(config)
    assert config.to_dict() == before


def test_validate_config_ignores_sharpen_strength() -> None:
    """A negative strength is a documented no-op, not an error."""
    assert validate_config(ConversionConfig(sharpen=True, sharpen_strength=-2.0)) == []


def test_validate_config_does_not_apply_the_atari_colour_limit() -> None:
    assert validate_config(ConversionConfig(colors=MAX_ATARI_COLORS + 12)) == []


def test_max_atari_colors_is_four() -> None:
    assert MAX_ATARI_COLORS == 4


def test_conversion_result_timings_default_to_an_empty_dict() -> None:
    result = _make_result(np.zeros((2, 2), dtype=np.uint8), [0, 1])
    assert result.timings == {}
    assert isinstance(result.timings, dict)


def test_conversion_result_timings_are_independent_per_instance() -> None:
    first = _make_result(np.zeros((2, 2), dtype=np.uint8), [0, 1])
    second = _make_result(np.zeros((2, 2), dtype=np.uint8), [0, 1])

    first.timings["palette"] = 0.5
    assert second.timings == {}


def test_conversion_result_timings_can_be_set() -> None:
    result = _make_result(np.zeros((2, 2), dtype=np.uint8), [0, 1])
    result.timings = {"resize": 0.001, "palette": 0.35}
    assert result.timings["palette"] == pytest.approx(0.35)


def test_conversion_result_converted_rgb_matches_indexed_to_rgb() -> None:
    indexed = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    result = _make_result(indexed, [0, 1])
    np.testing.assert_array_equal(result.converted_rgb, indexed_to_rgb(indexed, [0, 1]))


def test_conversion_result_converted_rgb_does_not_modify_the_result() -> None:
    indexed = np.array([[0, 2], [1, 0]], dtype=np.uint8)
    result = _make_result(indexed, [0, 100, 200])
    expected_indexed = indexed.copy()

    result.converted_rgb

    np.testing.assert_array_equal(result.indexed, expected_indexed)
    assert result.palette_indices == [0, 100, 200]


def test_config_to_dict_keys_are_frozen() -> None:
    """Presets and `--config` files must keep their exact shape."""
    assert set(ConversionConfig().to_dict()) == {
        "background_index",
        "colors",
        "dither",
        "export",
        "height",
        "metrics",
        "mode",
        "palette",
        "preview",
        "quantizer",
        "resize",
        "sharpen",
        "sharpen_strength",
        "width",
    }


def test_config_to_dict_includes_optional_keys_only_when_set() -> None:
    default = ConversionConfig().to_dict()
    assert "atari_colors" not in default
    assert "label_name" not in default

    custom = ConversionConfig(atari_colors=[0, 4, 10, 14], label_name="SCREEN").to_dict()
    assert custom["atari_colors"] == [0, 4, 10, 14]
    assert custom["label_name"] == "SCREEN"


def test_config_from_dict_accepts_a_legacy_export_string() -> None:
    config = ConversionConfig.from_dict({"export": "png"})
    assert config.export_formats == [ExportFormat.PNG]


def test_config_from_dict_rejects_unknown_enum_values() -> None:
    with pytest.raises(ValueError):
        ConversionConfig.from_dict({"palette": "does-not-exist"})


def test_config_round_trip_preserves_atari_colors_and_label(tmp_path: Path) -> None:
    config = ConversionConfig(
        palette=PaletteMethod.KMEANS,
        atari_colors=[0, 4, 10, 14],
        label_name="SCREEN_DEMO",
        preview=True,
        metrics=True,
    )
    path = tmp_path / "preset.json"
    save_config(config, path)

    loaded = load_config(path)
    assert loaded.atari_colors == [0, 4, 10, 14]
    assert loaded.label_name == "SCREEN_DEMO"
    assert loaded.preview is True
    assert loaded.metrics is True
    assert loaded.palette == PaletteMethod.KMEANS
