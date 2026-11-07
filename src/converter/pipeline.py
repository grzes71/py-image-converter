"""Conversion pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

from converter.aspect import apply_geometry
from converter.dithering import apply_dithering
from converter.edge_detect import detect_edges
from converter.exporters.asm_export import export_asm
from converter.exporters.atari_bin_export import export_bin
from converter.exporters.atari_dta_export import export_dta
from converter.exporters.png_export import export_png
from converter.exporters.xex_export import export_xex
from converter.histogram import compute_local_variance
from converter.image_loader import load_image
from converter.metrics import compute_metrics, format_metrics
from converter.palette import palette_rgb, select_palette
from converter.preview import generate_previews
from converter.quantizer import indexed_to_rgb, quantize
from converter.sharpen import apply_sharpen
from converter.types import (
    ConversionConfig,
    ConversionResult,
    DitherMethod,
    EdgeMask,
    ExportFormat,
    IndexedImage,
    MetricResult,
    QuantizerMethod,
    RGBImage,
)


def prepare_image(image: RGBImage, config: ConversionConfig) -> RGBImage:
    """Fit the source into the target grid and optionally sharpen it.

    The geometry stage runs first (``converter.aspect``): the PAL pixel aspect correction decides
    which part of the source reaches the palette and dithering stages, and the margins it adds are
    part of the converted image, so the order is not incidental.
    """
    fitted = apply_geometry(image, config)
    return apply_sharpen(fitted, config.sharpen, config.sharpen_strength)


def choose_palette(image: RGBImage, config: ConversionConfig) -> list[int]:
    """Select the Atari palette indices for a prepared image."""
    return select_palette(image, config.colors, config.palette, config.background_index)


def compute_edges(image: RGBImage) -> EdgeMask:
    """Detect the edge mask used by the dithering algorithms."""
    return detect_edges(image)


def render_indexed(
    image: RGBImage,
    palette_indices: list[int],
    edge_mask: EdgeMask,
    config: ConversionConfig,
) -> IndexedImage:
    """Render the indexed image: quantize for `dither=none`, dither for every other method."""
    if config.dither == DitherMethod.NONE:
        # The adaptive quantizer needs a local variance map. The histogram report that used to be
        # computed here only carried one in "full" mode (which `dither=none` never requested), so
        # the map never reached the quantizer and it silently behaved like `nearest` (B4).
        variance_map = None
        if config.quantizer == QuantizerMethod.ADAPTIVE:
            variance_map = compute_local_variance(image)
        return quantize(image, palette_indices, config.quantizer, variance_map)
    return apply_dithering(image, palette_rgb(palette_indices), config.dither, edge_mask)


def build_metrics(
    image: RGBImage,
    indexed: IndexedImage,
    palette_indices: list[int],
) -> MetricResult:
    """Compute the quality metrics of a rendered image."""
    return compute_metrics(image, indexed, palette_indices)


def convert_image(input_path: str | Path, config: ConversionConfig) -> ConversionResult:
    """Run the full conversion pipeline."""
    image = prepare_image(load_image(input_path).data, config)
    palette_indices = choose_palette(image, config)
    edge_mask = compute_edges(image)
    indexed = render_indexed(image, palette_indices, edge_mask, config)
    metrics = build_metrics(image, indexed, palette_indices) if config.metrics else None

    return ConversionResult(
        indexed=indexed,
        palette_indices=palette_indices,
        palette_rgb=palette_rgb(palette_indices),
        source_resized=image,
        metrics=metrics,
    )


def export_result(
    result: ConversionResult,
    output_path: str | Path,
    config: ConversionConfig,
    *,
    antic_mode: str | None = None,
) -> list[Path]:
    """Export conversion result to requested formats.

    ``antic_mode`` is forwarded to the XEX exporter only (``None`` keeps the shape-based inference);
    the other formats ignore it.
    """
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
            export_asm(
                result.indexed, p,
                label=config.label_name or "SCREEN",
                atari_colors=config.atari_colors,
            )
            exported.append(p)
        elif fmt == ExportFormat.DTA:
            p = parent / f"{stem}.dta"
            export_dta(result.indexed, p)
            exported.append(p)
        elif fmt == ExportFormat.XEX:
            p = parent / f"{stem}.xex"
            export_xex(
                result.indexed, p,
                palette_indices=result.palette_indices,
                antic_mode=antic_mode,
                atari_colors=config.atari_colors,
            )
            exported.append(p)

    if config.preview:
        preview_dir = parent / f"{stem}_preview"
        exported.extend(
            generate_previews(
                result.source_resized,
                result.indexed,
                result.palette_indices,
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
