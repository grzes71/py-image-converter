"""Conversion pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

from converter.dithering import apply_dithering
from converter.edge_detect import detect_edges
from converter.exporters.asm_export import export_asm
from converter.exporters.atari_bin_export import export_bin
from converter.exporters.atari_dta_export import export_dta
from converter.exporters.png_export import export_png
from converter.exporters.xex_export import export_xex
from converter.histogram import analyze_histogram
from converter.image_loader import load_image
from converter.metrics import compute_metrics, format_metrics
from converter.palette import palette_rgb, select_palette
from converter.preview import generate_previews
from converter.quantizer import indexed_to_rgb, quantize
from converter.resize import resize_image
from converter.sharpen import apply_sharpen
from converter.types import ConversionConfig, ConversionResult, DitherMethod, ExportFormat


def convert_image(input_path: str | Path, config: ConversionConfig) -> ConversionResult:
    """Run the full conversion pipeline."""
    buffer = load_image(input_path)
    image = buffer.data

    resized = resize_image(image, config.width, config.height, config.resize)
    sharpened = apply_sharpen(resized, config.sharpen, config.sharpen_strength)

    hist_report = analyze_histogram(sharpened, full=config.dither == DitherMethod.ADAPTIVE)

    palette_indices = select_palette(
        sharpened,
        config.colors,
        config.palette,
        config.background_index,
    )
    pal_rgb = palette_rgb(palette_indices)

    edge_mask = detect_edges(sharpened)

    if config.dither == DitherMethod.NONE:
        indexed = quantize(
            sharpened,
            list(range(len(palette_indices))),
            config.quantizer,
            hist_report.local_variance,
        )
    else:
        indexed = apply_dithering(sharpened, pal_rgb, config.dither, edge_mask)

    metrics = None
    if config.metrics:
        metrics = compute_metrics(sharpened, indexed, list(range(len(palette_indices))))

    return ConversionResult(
        indexed=indexed,
        palette_indices=palette_indices,
        palette_rgb=pal_rgb,
        source_resized=sharpened,
        metrics=metrics,
    )


def export_result(
    result: ConversionResult,
    output_path: str | Path,
    config: ConversionConfig,
) -> list[Path]:
    """Export conversion result to requested formats."""
    output_path = Path(output_path)
    stem = output_path.stem
    parent = output_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []

    formats = config.export_formats
    if not formats:
        formats = [ExportFormat.BIN]

    for fmt in formats:
        if fmt == ExportFormat.BIN:
            p = parent / f"{stem}.bin"
            export_bin(result.indexed, p)
            exported.append(p)
        elif fmt == ExportFormat.PNG:
            p = parent / f"{stem}.png"
            export_png(result.indexed, result.palette_rgb, p)
            exported.append(p)
        elif fmt == ExportFormat.ASM:
            p = parent / f"{stem}.s"
            export_asm(result.indexed, p)
            exported.append(p)
        elif fmt == ExportFormat.DTA:
            p = parent / f"{stem}.dta"
            export_dta(result.indexed, p)
            exported.append(p)
        elif fmt == ExportFormat.XEX:
            p = parent / f"{stem}.xex"
            export_xex(result.indexed, p)
            exported.append(p)

    if config.preview:
        preview_dir = parent / f"{stem}_preview"
        exported.extend(
            generate_previews(
                result.source_resized,
                result.indexed,
                list(range(len(result.palette_indices))),
                preview_dir,
                stem,
            )
        )

    return exported


def run_conversion(
    input_path: str | Path,
    output_path: str | Path,
    config: ConversionConfig,
) -> ConversionResult:
    """Convert and export an image."""
    result = convert_image(input_path, config)
    export_result(result, output_path, config)
    if result.metrics:
        print(format_metrics(result.metrics))
    return result
