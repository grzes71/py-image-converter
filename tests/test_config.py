"""Tests for config module."""

import json
from pathlib import Path

from converter.config import load_config, save_config
from converter.types import ConversionConfig, DitherMethod, ExportFormat


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
