"""Configuration load/save utilities."""

from __future__ import annotations

import json
from pathlib import Path

from converter.types import ConversionConfig


def load_config(path: str | Path) -> ConversionConfig:
    """Load conversion settings from a JSON file."""
    with open(path, encoding="utf-8") as f:
        return ConversionConfig.from_dict(json.load(f))


def save_config(config: ConversionConfig, path: str | Path) -> None:
    """Save conversion settings to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=4)
