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


def validate_config(config: ConversionConfig) -> list[str]:
    """Return every blocking problem of a configuration, in a stable order.

    The CLI never calls this function - it exists for the GUI and for preset loading. An empty
    list means the configuration is valid. Only values that break the converter itself are
    reported: `sharpen_strength` (negative values are a documented no-op) and `atari_colors`
    (validated with the export formats) are deliberately not checked.
    """
    problems: list[str] = []
    if config.width < 1:
        problems.append(f"width must be >= 1 (got {config.width})")
    if config.height < 1:
        problems.append(f"height must be >= 1 (got {config.height})")
    if not 2 <= config.colors <= 256:
        problems.append(f"colors must be between 2 and 256 (got {config.colors})")
    if not 0 <= config.background_index <= 255:
        problems.append(
            f"background_index must be between 0 and 255 (got {config.background_index})"
        )
    return problems
