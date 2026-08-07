"""CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from converter.config import load_config, save_config
from converter.exporters.xex_export import export_xex
from converter.pipeline import run_conversion
from converter.types import (
    ConversionConfig,
    DitherMethod,
    ExportFormat,
    PaletteMethod,
    QuantizerMethod,
    ResizeMethod,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert images to Atari 8-bit graphics (ANTIC mode E)",
    )
    parser.add_argument("input", help="Input image path (PNG/BMP/GIF/JPG)")
    parser.add_argument("output", help="Output path (extension determines format if --export not set)")
    parser.add_argument("--width", type=int, default=160, help="Output width (default: 160)")
    parser.add_argument("--height", type=int, default=192, help="Output height (default: 192)")
    parser.add_argument("--mode", default="ANTIC_E", help="Atari mode (default: ANTIC_E)")
    parser.add_argument(
        "-x",
        "--xex",
        help="Explicit output path for Atari executable preview (.xex)",
    )
    parser.add_argument(
        "-m",
        "--antic-mode",
        choices=["D", "E", "F", "7", "15", "8"],
        help="ANTIC mode for XEX preview (D/E/F or BASIC equivalents 7/15/8)",
    )
    parser.add_argument(
        "--palette",
        choices=[p.value for p in PaletteMethod],
        default="popularity",
        help="Palette selection algorithm",
    )
    parser.add_argument("--colors", type=int, default=4, help="Number of colors (default: 4)")
    parser.add_argument(
        "--dither",
        choices=[d.value for d in DitherMethod],
        default="floyd",
        help="Dithering algorithm",
    )
    parser.add_argument(
        "--quantizer",
        choices=[q.value for q in QuantizerMethod],
        default="nearest",
        help="Quantization method",
    )
    parser.add_argument(
        "--resize",
        choices=[r.value for r in ResizeMethod],
        default="lanczos",
        help="Resize method",
    )
    parser.add_argument("--sharpen", action="store_true", help="Apply unsharp mask")
    parser.add_argument("--sharpen-strength", type=float, default=1.0, help="Sharpen strength")
    parser.add_argument("--background-index", type=int, default=0, help="Background Atari color index")
    parser.add_argument("--preview", action="store_true", help="Generate preview images")
    parser.add_argument("--metrics", action="store_true", help="Print quality metrics")
    parser.add_argument(
        "--export",
        action="append",
        choices=[e.value for e in ExportFormat],
        help="Export format (repeat for multiple, e.g. --export asm --export xex; default: inferred from output extension)",
    )
    parser.add_argument("--config", help="Load settings from JSON config file")
    parser.add_argument("--save-config", help="Save current settings to JSON file")
    parser.add_argument(
        "--atari-colors",
        type=str,
        default=None,
        help="Override color registers in XEX output (comma-separated hex, e.g. 0x00,0x04,0x0a,0x0e)",
    )
    return parser


def _infer_export_format(output: Path) -> ExportFormat:
    ext_map = {
        ".png": ExportFormat.PNG,
        ".bin": ExportFormat.BIN,
        ".s": ExportFormat.ASM,
        ".asm": ExportFormat.ASM,
        ".dta": ExportFormat.DTA,
        ".xex": ExportFormat.XEX,
    }
    return ext_map.get(output.suffix.lower(), ExportFormat.BIN)


def _parse_atari_colors(raw: str | None) -> list[int] | None:
    """Parse a comma-separated hex color string like '0x00,0x04,0x0a,0x0e'."""
    if raw is None:
        return None
    colors: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            raise ValueError("Empty color value in --atari-colors")
        colors.append(int(token, 16))
    if not colors:
        raise ValueError("--atari-colors requires at least one value")
    return colors


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    atari_colors = None
    if args.atari_colors is not None:
        try:
            atari_colors = _parse_atari_colors(args.atari_colors)
        except ValueError as exc:
            print(f"Error parsing --atari-colors: {exc}", file=sys.stderr)
            return 1

    if args.config:
        config = load_config(args.config)
        if atari_colors is not None:
            config.atari_colors = atari_colors
    else:
        export_formats = [ExportFormat(e) for e in args.export] if args.export else None
        if export_formats is None:
            export_formats = [_infer_export_format(Path(args.output))]

        config = ConversionConfig(
            width=args.width,
            height=args.height,
            palette=PaletteMethod(args.palette),
            colors=args.colors,
            dither=DitherMethod(args.dither),
            quantizer=QuantizerMethod(args.quantizer),
            resize=ResizeMethod(args.resize),
            sharpen=args.sharpen,
            sharpen_strength=args.sharpen_strength,
            background_index=args.background_index,
            preview=args.preview,
            metrics=args.metrics,
            export_formats=export_formats,
            atari_colors=atari_colors,
        )

    if args.save_config:
        save_config(config, args.save_config)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    result = run_conversion(input_path, args.output, config)

    if args.xex:
        export_xex(
            result.indexed,
            args.xex,
            palette_indices=result.palette_indices,
            antic_mode=args.antic_mode,
            atari_colors=atari_colors,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
