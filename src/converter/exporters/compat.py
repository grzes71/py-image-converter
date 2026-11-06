"""Export compatibility checks shared by the CLI and the GUI.

``check_export_compatibility`` answers "would this export succeed?" *before* anything is written, so
the GUI can disable an incompatible format instead of showing an error. The verdicts mirror the real
exporter validations: they reuse the exporter constants and the public ``normalize_antic_mode``
helper instead of restating its rules, and ``test_exports.py`` compares every verdict against an
actual export attempt.
"""

from __future__ import annotations

from converter.exporters.atari_bin_export import ANTIC_E_HEIGHT, ANTIC_E_WIDTH
from converter.exporters.xex_export import normalize_antic_mode
from converter.types import MAX_ATARI_COLORS, ConversionResult, ExportFormat, IndexedImage

# ``pack_antic_e`` accepts a 192-line screen plus the 190-line variant used by some viewers.
_ANTIC_E_HEIGHTS = (190, ANTIC_E_HEIGHT)
# A PNG palette chunk stores 256 RGB triples.
_PNG_MAX_COLORS = 256
_BIN_LIKE_FORMATS = (ExportFormat.BIN, ExportFormat.ASM, ExportFormat.DTA)


def check_export_compatibility(
    result: ConversionResult,
    fmt: ExportFormat,
    *,
    antic_mode: str | None = None,
) -> str | None:
    """Return ``None`` when ``fmt`` can be exported, or a readable reason when it cannot.

    The verdict is based on the real result - the indexed shape and the index values that actually
    occur - not on the requested colour count. ``antic_mode`` applies to XEX only (``None`` keeps the
    shape-based inference). Nothing is written and no exporter is called.
    """
    if fmt is ExportFormat.PNG:
        return _check_png(result)
    if fmt in _BIN_LIKE_FORMATS:
        return _check_antic_e(result)
    if fmt is ExportFormat.XEX:
        return _check_xex(result, antic_mode)
    return None


def _check_png(result: ConversionResult) -> str | None:
    palette_size = len(result.palette_rgb)
    if palette_size > _PNG_MAX_COLORS:
        return f"PNG supports at most {_PNG_MAX_COLORS} colours (got {palette_size})"
    return None


def _check_antic_e(result: ConversionResult) -> str | None:
    height, width = result.indexed.shape
    if width != ANTIC_E_WIDTH or height not in _ANTIC_E_HEIGHTS:
        allowed = " or ".join(f"{ANTIC_E_WIDTH}x{line_count}" for line_count in _ANTIC_E_HEIGHTS)
        return f"ANTIC E requires {allowed} (got {width}x{height})"
    return _check_indices(result.indexed, bits=2)


def _check_xex(result: ConversionResult, antic_mode: str | None) -> str | None:
    indexed = result.indexed
    try:
        mode = normalize_antic_mode(antic_mode, indexed)
    except ValueError as exc:
        return str(exc)

    bits = 1 if mode == "F" else 2
    index_problem = _check_indices(indexed, bits=bits)
    if index_problem is not None:
        return index_problem

    pixels_per_byte = 8 // bits
    width = indexed.shape[1]
    if width % pixels_per_byte != 0:
        return (
            f"mode {mode} requires a width divisible by {pixels_per_byte} (got {width}); "
            "export_xex would fail with IndexError"
        )
    return None


def _check_indices(indexed: IndexedImage, *, bits: int) -> str | None:
    limit = (1 << bits) - 1
    maximum = int(indexed.max()) if indexed.size else 0
    if maximum <= limit:
        return None
    detail = f"0-{limit} for {bits}bpp mode (found {maximum})"
    if bits == 2:
        detail += f"; at most {MAX_ATARI_COLORS} colours"
    return f"indexed values must be {detail}"
