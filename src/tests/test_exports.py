"""Tests for the export formats, the compatibility helper and ``export_result``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import converter.exporters as exporters_package
from converter.exporters.asm_export import export_asm
from converter.exporters.atari_bin_export import export_bin
from converter.exporters.atari_dta_export import export_dta
from converter.exporters.compat import check_export_compatibility
from converter.exporters.png_export import export_png
from converter.exporters.xex_export import export_xex, normalize_antic_mode
from converter.hardware_colors import resolved_registers
from converter.pipeline import convert_image, export_result
from converter.types import (
    ConversionConfig,
    ConversionResult,
    DitherMethod,
    ExportFormat,
    PaletteMethod,
)

REFERENCE_IMAGE = Path(__file__).parent / "fixtures" / "reference.png"
BIN_EXPORTERS = {
    ExportFormat.BIN: export_bin,
    ExportFormat.ASM: export_asm,
    ExportFormat.DTA: export_dta,
}


def test_asm_export(tmp_path: Path) -> None:
    indexed = np.zeros((192, 160), dtype=np.uint8)
    path = tmp_path / "screen.s"
    export_asm(indexed, path)
    text = path.read_text(encoding="utf-8")
    assert "SCREEN_DATA" in text
    assert ".byte" in text


def test_dta_export(tmp_path: Path) -> None:
    indexed = np.zeros((192, 160), dtype=np.uint8)
    path = tmp_path / "screen.dta"
    export_dta(indexed, path)
    assert path.exists()
    assert ".byte" in path.read_text(encoding="utf-8")


def test_xex_export(tmp_path: Path) -> None:
    indexed = np.zeros((192, 160), dtype=np.uint8)
    path = tmp_path / "screen.xex"
    export_xex(indexed, path, palette_indices=[0x00, 0x22, 0x44, 0x66, 0x88], antic_mode="E")
    data = path.read_bytes()
    assert data[:2] == bytes([0xFF, 0xFF])


def _parse_xex_segments(data: bytes) -> list[tuple[int, int, bytes]]:
    assert data[:2] == bytes([0xFF, 0xFF])
    i = 2
    out: list[tuple[int, int, bytes]] = []
    while i < len(data):
        start = data[i] | (data[i + 1] << 8)
        end = data[i + 2] | (data[i + 3] << 8)
        i += 4
        size = end - start + 1
        payload = data[i : i + size]
        out.append((start, end, payload))
        i += size
    return out


def test_xex_segments_and_dlist_mode_d(tmp_path: Path) -> None:
    indexed = np.zeros((96, 160), dtype=np.uint8)
    path = tmp_path / "screen_d.xex"
    export_xex(indexed, path, antic_mode="D")

    data = path.read_bytes()
    segments = _parse_xex_segments(data)
    assert len(segments) == 4

    s0_start, s0_end, s0_payload = segments[0]
    assert s0_start == 0x4000
    assert s0_end == 0x4EFF
    assert len(s0_payload) == 3840

    s1_start, _, dlist = segments[1]
    assert s1_start == 0x3E00
    assert dlist[:3] == bytes([0x70, 0x70, 0x70])
    assert dlist[3:6] == bytes([0x4D, 0x00, 0x40])
    assert dlist[6 : 6 + 95] == bytes([0x0D] * 95)
    assert dlist[-3:] == bytes([0x41, 0x00, 0x3E])

    s2_start, _, init_code = segments[2]
    assert s2_start == 0x3800
    assert init_code[:5] == bytes([0xA9, 0x22, 0x8D, 0x29, 0x02])

    s3_start, s3_end, runad = segments[3]
    assert s3_start == 0x02E0
    assert s3_end == 0x02E1
    assert runad == bytes([0x00, 0x38])


def test_xex_segments_and_dlist_mode_e(tmp_path: Path) -> None:
    indexed = np.zeros((192, 160), dtype=np.uint8)
    path = tmp_path / "screen_e.xex"
    export_xex(indexed, path, antic_mode="E")

    data = path.read_bytes()
    segments = _parse_xex_segments(data)
    assert len(segments) == 4

    s0_start, s0_end, s0_payload = segments[0]
    assert s0_start == 0x4000
    assert s0_end == 0x5E0F
    assert len(s0_payload) == 7696

    _, _, dlist = segments[1]
    assert dlist[:3] == bytes([0x70, 0x70, 0x70])
    assert dlist[3:6] == bytes([0x4E, 0x00, 0x40])
    assert dlist[6 : 6 + 101] == bytes([0x0E] * 101)
    second_lms = 6 + 101
    assert dlist[second_lms : second_lms + 3] == bytes([0x4E, 0x00, 0x50])
    assert dlist[second_lms + 3 : second_lms + 3 + 89] == bytes([0x0E] * 89)
    assert dlist[-3:] == bytes([0x41, 0x00, 0x3E])


def test_xex_segments_and_dlist_mode_e_160x190(tmp_path: Path) -> None:
    indexed = np.zeros((190, 160), dtype=np.uint8)
    path = tmp_path / "screen_e_190.xex"
    export_xex(indexed, path, antic_mode="E")

    data = path.read_bytes()
    segments = _parse_xex_segments(data)
    assert len(segments) == 4

    s0_start, s0_end, s0_payload = segments[0]
    assert s0_start == 0x4000
    assert s0_end == 0x5DBF
    assert len(s0_payload) == 7616

    _, _, dlist = segments[1]
    assert dlist[:3] == bytes([0x70, 0x70, 0x70])
    assert dlist[3:6] == bytes([0x4E, 0x00, 0x40])
    assert dlist[6 : 6 + 101] == bytes([0x0E] * 101)
    second_lms = 6 + 101
    assert dlist[second_lms : second_lms + 3] == bytes([0x4E, 0x00, 0x50])
    assert dlist[second_lms + 3 : second_lms + 3 + 87] == bytes([0x0E] * 87)
    assert dlist[-3:] == bytes([0x41, 0x00, 0x3E])


def test_xex_segments_and_dlist_mode_f(tmp_path: Path) -> None:
    indexed = np.zeros((192, 320), dtype=np.uint8)
    indexed[:, ::2] = 1
    path = tmp_path / "screen_f.xex"
    export_xex(indexed, path, antic_mode="F")

    data = path.read_bytes()
    segments = _parse_xex_segments(data)
    assert len(segments) == 4

    s0_start, s0_end, s0_payload = segments[0]
    assert s0_start == 0x4000
    assert s0_end == 0x5E0F
    assert len(s0_payload) == 7696

    _, _, dlist = segments[1]
    assert dlist[:3] == bytes([0x70, 0x70, 0x70])
    assert dlist[3:6] == bytes([0x4F, 0x00, 0x40])
    assert dlist[6 : 6 + 101] == bytes([0x0F] * 101)
    second_lms = 6 + 101
    assert dlist[second_lms : second_lms + 3] == bytes([0x4F, 0x00, 0x50])
    assert dlist[second_lms + 3 : second_lms + 3 + 89] == bytes([0x0F] * 89)
    assert dlist[-3:] == bytes([0x41, 0x00, 0x3E])


# --- export compatibility helper ----------------------------------------


def _result(
    shape: tuple[int, int],
    max_index: int,
    *,
    palette_size: int | None = None,
) -> ConversionResult:
    """Hand-built result: only the shape and the real index values matter for a check."""
    size = palette_size if palette_size is not None else max_index + 1
    indexed = np.zeros(shape, dtype=np.uint8)
    indexed[0, 0] = max_index
    return ConversionResult(
        indexed=indexed,
        palette_indices=list(range(min(size, 256))),
        palette_rgb=np.zeros((size, 3), dtype=np.uint8),
        source_resized=np.zeros((*shape, 3), dtype=np.uint8),
    )


def _run_export(
    result: ConversionResult,
    fmt: ExportFormat,
    antic_mode: str | None,
    directory: Path,
) -> Path:
    """Run the real exporter so a helper verdict can be compared with reality."""
    directory.mkdir(parents=True, exist_ok=True)
    if fmt is ExportFormat.PNG:
        path = directory / "screen.png"
        export_png(result.indexed, result.palette_rgb, path)
    elif fmt in BIN_EXPORTERS:
        path = directory / f"screen.{fmt.value}"
        BIN_EXPORTERS[fmt](result.indexed, path)
    else:
        path = directory / "screen.xex"
        export_xex(result.indexed, path, antic_mode=antic_mode)
    return path


# (shape, max_index, palette_size, format, antic_mode)
COMPATIBILITY_CASES = [
    # BIN / ASM / DTA: ANTIC E shape and 0-3 indices
    ((192, 160), 3, 4, ExportFormat.BIN, None),
    ((190, 160), 3, 4, ExportFormat.BIN, None),
    ((192, 160), 3, 4, ExportFormat.ASM, None),
    ((190, 160), 3, 4, ExportFormat.DTA, None),
    ((192, 160), 7, 8, ExportFormat.BIN, None),
    ((192, 160), 7, 8, ExportFormat.ASM, None),
    ((192, 160), 7, 8, ExportFormat.DTA, None),
    ((96, 160), 3, 4, ExportFormat.BIN, None),
    ((191, 160), 3, 4, ExportFormat.ASM, None),
    ((192, 320), 3, 4, ExportFormat.DTA, None),
    # PNG: any shape, up to 256 palette entries
    ((192, 160), 3, 4, ExportFormat.PNG, None),
    ((192, 320), 15, 16, ExportFormat.PNG, None),
    ((192, 160), 255, 256, ExportFormat.PNG, None),
    ((192, 160), 255, 257, ExportFormat.PNG, None),
    # XEX: inferred mode
    ((192, 160), 3, 4, ExportFormat.XEX, None),
    ((96, 160), 3, 4, ExportFormat.XEX, None),
    ((192, 320), 1, 2, ExportFormat.XEX, None),
    ((192, 320), 3, 4, ExportFormat.XEX, None),
    ((100, 200), 3, 4, ExportFormat.XEX, None),
    # XEX: explicit mode and aliases
    ((192, 160), 3, 4, ExportFormat.XEX, "D"),
    ((192, 160), 3, 4, ExportFormat.XEX, "E"),
    ((192, 160), 1, 2, ExportFormat.XEX, "F"),
    ((192, 160), 2, 4, ExportFormat.XEX, "F"),
    ((192, 160), 3, 4, ExportFormat.XEX, "7"),
    ((192, 160), 3, 4, ExportFormat.XEX, "15"),
    ((192, 160), 3, 4, ExportFormat.XEX, "8"),
    ((192, 160), 3, 4, ExportFormat.XEX, "d"),
    ((192, 160), 3, 4, ExportFormat.XEX, "X"),
    # XEX: packing-unit widths (the exporter raises IndexError for the bad ones)
    ((192, 158), 3, 4, ExportFormat.XEX, "D"),
    ((192, 158), 3, 4, ExportFormat.XEX, "E"),
    ((192, 316), 1, 2, ExportFormat.XEX, "F"),
    ((192, 156), 3, 4, ExportFormat.XEX, "D"),
    ((192, 320), 3, 4, ExportFormat.XEX, "E"),
]


@pytest.mark.parametrize(
    ("shape", "max_index", "palette_size", "fmt", "antic_mode"),
    COMPATIBILITY_CASES,
)
def test_helper_verdict_matches_the_real_export(
    tmp_path: Path,
    shape: tuple[int, int],
    max_index: int,
    palette_size: int,
    fmt: ExportFormat,
    antic_mode: str | None,
) -> None:
    """``None`` must mean "the exporter succeeds"; a message must mean "it raises"."""
    result = _result(shape, max_index, palette_size=palette_size)

    reason = check_export_compatibility(result, fmt, antic_mode=antic_mode)

    if reason is None:
        assert _run_export(result, fmt, antic_mode, tmp_path / "compatible").read_bytes()
    else:
        with pytest.raises((ValueError, IndexError)):  # the packers raise IndexError for bad widths
            _run_export(result, fmt, antic_mode, tmp_path / "incompatible")


@pytest.mark.parametrize(
    ("result", "fmt", "antic_mode", "needle"),
    [
        (_result((192, 160), 7, palette_size=8), ExportFormat.BIN, None, "0-3"),
        (_result((96, 160), 3), ExportFormat.BIN, None, "160x190"),
        (_result((192, 160), 255, palette_size=257), ExportFormat.PNG, None, "256"),
        (_result((192, 160), 2, palette_size=4), ExportFormat.XEX, "F", "0-1"),
        (_result((100, 200), 3), ExportFormat.XEX, None, "Cannot infer"),
        (_result((192, 160), 3), ExportFormat.XEX, "X", "Invalid ANTIC mode"),
        (_result((192, 158), 3), ExportFormat.XEX, "D", "divisible by 4"),
        (_result((192, 316), 1), ExportFormat.XEX, "F", "divisible by 8"),
    ],
)
def test_helper_messages_name_the_violated_limit(
    result: ConversionResult, fmt: ExportFormat, antic_mode: str | None, needle: str
) -> None:
    reason = check_export_compatibility(result, fmt, antic_mode=antic_mode)

    assert reason is not None
    assert needle in reason


def test_helper_returns_none_for_a_compatible_result() -> None:
    result = _result((192, 160), 3, palette_size=4)

    assert check_export_compatibility(result, ExportFormat.BIN) is None
    assert check_export_compatibility(result, ExportFormat.PNG) is None
    assert check_export_compatibility(result, ExportFormat.XEX) is None


def test_compatibility_helper_is_exported_from_the_package() -> None:
    assert "check_export_compatibility" in exporters_package.__all__
    assert exporters_package.check_export_compatibility is check_export_compatibility


# --- normalize_antic_mode -----------------------------------------------


@pytest.mark.parametrize(
    ("antic_mode", "expected"),
    [
        ("D", "D"),
        ("E", "E"),
        ("F", "F"),
        ("7", "D"),
        ("15", "E"),
        ("8", "F"),
        ("d", "D"),
        ("e", "E"),
        ("f", "F"),
    ],
)
def test_normalize_antic_mode_accepts_modes_and_aliases(antic_mode: str, expected: str) -> None:
    assert normalize_antic_mode(antic_mode, np.zeros((192, 160), dtype=np.uint8)) == expected


@pytest.mark.parametrize(
    ("shape", "expected"),
    [((192, 160), "E"), ((96, 160), "D"), ((192, 320), "F")],
)
def test_normalize_antic_mode_infers_when_none(shape: tuple[int, int], expected: str) -> None:
    assert normalize_antic_mode(None, np.zeros(shape, dtype=np.uint8)) == expected


def test_normalize_antic_mode_rejects_invalid_and_uninferable_modes() -> None:
    with pytest.raises(ValueError, match="Invalid ANTIC mode"):
        normalize_antic_mode("X", np.zeros((192, 160), dtype=np.uint8))
    with pytest.raises(ValueError, match="Cannot infer"):
        normalize_antic_mode(None, np.zeros((100, 200), dtype=np.uint8))


# --- export_result(antic_mode=...) --------------------------------------


@pytest.fixture(scope="module")
def xex_conversion() -> tuple[ConversionResult, ConversionConfig]:
    """One 160x192 conversion (inferred mode E) shared by the ``export_result`` tests."""
    config = ConversionConfig(
        dither=DitherMethod.NONE,
        palette=PaletteMethod.MEDIAN_CUT,
        export_formats=[ExportFormat.BIN, ExportFormat.XEX],
    )
    return convert_image(REFERENCE_IMAGE, config), config


def test_export_result_default_path_is_unchanged(
    tmp_path: Path, xex_conversion: tuple[ConversionResult, ConversionConfig]
) -> None:
    result, config = xex_conversion

    export_result(result, tmp_path / "default" / "screen.bin", config)
    export_result(result, tmp_path / "explicit" / "screen.bin", config, antic_mode=None)

    for name in ("screen.bin", "screen.xex"):
        assert (tmp_path / "default" / name).read_bytes() == (
            tmp_path / "explicit" / name
        ).read_bytes()


def test_export_result_forwards_antic_mode_to_xex(
    tmp_path: Path, xex_conversion: tuple[ConversionResult, ConversionConfig]
) -> None:
    result, config = xex_conversion

    export_result(result, tmp_path / "explicit" / "screen.bin", config, antic_mode="D")
    direct = tmp_path / "direct.xex"
    export_xex(
        result.indexed,
        direct,
        palette_indices=result.palette_indices,
        antic_mode="D",
        # ``export_result`` resolves the registers itself (manual values or the rgb2a8 mapping).
        atari_colors=resolved_registers(result, config, "D"),
    )
    assert (tmp_path / "explicit" / "screen.xex").read_bytes() == direct.read_bytes()

    export_result(result, tmp_path / "inferred" / "screen.bin", config)
    assert (tmp_path / "explicit" / "screen.xex").read_bytes() != (
        tmp_path / "inferred" / "screen.xex"
    ).read_bytes()


def test_export_result_ignores_antic_mode_for_other_formats(tmp_path: Path) -> None:
    config = ConversionConfig(
        dither=DitherMethod.NONE,
        palette=PaletteMethod.MEDIAN_CUT,
        export_formats=[ExportFormat.BIN, ExportFormat.PNG, ExportFormat.ASM, ExportFormat.DTA],
    )
    result = convert_image(REFERENCE_IMAGE, config)

    export_result(result, tmp_path / "plain" / "screen.bin", config)
    export_result(result, tmp_path / "directed" / "screen.bin", config, antic_mode="F")

    for name in ("screen.bin", "screen.png", "screen.s", "screen.dta"):
        assert (tmp_path / "plain" / name).read_bytes() == (
            tmp_path / "directed" / name
        ).read_bytes()


def test_export_result_propagates_an_invalid_antic_mode(
    tmp_path: Path, xex_conversion: tuple[ConversionResult, ConversionConfig]
) -> None:
    result, config = xex_conversion

    with pytest.raises(ValueError, match="Invalid ANTIC mode"):
        export_result(result, tmp_path / "bad" / "screen.bin", config, antic_mode="X")
