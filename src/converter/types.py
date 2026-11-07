"""Shared types and dataclasses for the conversion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class AtariMode(str, Enum):
    """Supported Atari display modes."""

    ANTIC_E = "ANTIC_E"


class ResizeMethod(str, Enum):
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"


class AspectMode(str, Enum):
    """How the PAL pixel aspect ratio is corrected (see :mod:`converter.aspect`)."""

    DISABLED = "disabled"
    FIT = "fit"
    CROP = "crop"


class DitherMethod(str, Enum):
    NONE = "none"
    FLOYD = "floyd"
    JARVIS = "jarvis"
    STUCKI = "stucki"
    SIERRA = "sierra"
    SIERRA_LITE = "sierra_lite"
    BURKES = "burkes"
    ATKINSON = "atkinson"
    BAYER2 = "bayer2"
    BAYER4 = "bayer4"
    BAYER8 = "bayer8"
    ADAPTIVE = "adaptive"
    ATARI_FALSE_FLOYD = "atari_false_floyd"
    ATARI_TWO_ROW_SIERRA = "atari_two_row_sierra"
    ATARI_DIAGONAL = "atari_diagonal"
    ATARI_BAYER2V2 = "atari_bayer2v2"
    ATARI_BAYER2X4 = "atari_bayer2x4"
    ATARI_NON_RECTANGULAR8 = "atari_non_rectangular8"


class PaletteMethod(str, Enum):
    POPULARITY = "popularity"
    KMEANS = "kmeans"
    MEDIAN_CUT = "median_cut"
    OCTREE = "octree"
    PERCEPTUAL = "perceptual"


class QuantizerMethod(str, Enum):
    NEAREST = "nearest"
    PERCEPTUAL = "perceptual"
    WEIGHTED = "weighted"
    ADAPTIVE = "adaptive"


class ExportFormat(str, Enum):
    PNG = "png"
    BIN = "bin"
    ASM = "asm"
    DTA = "dta"
    XEX = "xex"


RGBImage = NDArray[np.uint8]
FloatImage = NDArray[np.float64]
IndexedImage = NDArray[np.uint8]
EdgeMask = NDArray[np.float64]

MAX_ATARI_COLORS = 4
"""Hardware limit of the ANTIC 2bpp playfield (four colours in modes D/E).

This is an export-only limit: the converter core and the PNG output accept more colours.
"""


@dataclass
class ImageBuffer:
    """Wrapper for H×W×3 RGB image data."""

    data: RGBImage

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, 3)


@dataclass
class AtariColor:
    """Single entry in the Atari hardware palette."""

    index: int
    rgb: tuple[int, int, int]
    hsv: tuple[float, float, float]
    luminance: float
    name: str
    distances: dict[int, float] = field(default_factory=dict)


@dataclass
class HistogramReport:
    """Histogram analysis results."""

    rgb_histogram: dict[str, NDArray[np.int64]]
    hsv_histogram: dict[str, NDArray[np.int64]] | None = None
    luminance_histogram: NDArray[np.int64] | None = None
    local_variance: FloatImage | None = None
    gradient_magnitude: FloatImage | None = None
    detail_score: float = 0.0


@dataclass
class MetricResult:
    """Quality metrics comparing source and converted images."""

    mse: float
    psnr: float
    ssim: float
    mean_rgb_error: float
    mean_luminance_error: float
    colors_used: int
    histogram_coverage: float


@dataclass
class ConversionResult:
    """Output of a full conversion run."""

    indexed: IndexedImage
    palette_indices: list[int]
    palette_rgb: NDArray[np.uint8]
    source_resized: RGBImage
    metrics: MetricResult | None = None
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def converted_rgb(self) -> RGBImage:
        """Indexed image rendered with the selected palette colours."""
        # Imported lazily: `converter.quantizer` imports this module (through `atari_palette`),
        # so a module-level import here would create an import cycle.
        from converter.quantizer import indexed_to_rgb

        return indexed_to_rgb(self.indexed, self.palette_indices)


@dataclass
class ConversionConfig:
    """Configuration for image conversion."""

    mode: AtariMode = AtariMode.ANTIC_E
    width: int = 160
    height: int = 192
    palette: PaletteMethod = PaletteMethod.POPULARITY
    colors: int = 4
    dither: DitherMethod = DitherMethod.FLOYD
    quantizer: QuantizerMethod = QuantizerMethod.NEAREST
    resize: ResizeMethod = ResizeMethod.LANCZOS
    aspect: AspectMode = AspectMode.DISABLED
    """PAL pixel aspect correction. ``DISABLED`` keeps the historical stretch-to-fit result."""
    pixel_aspect: float | None = None
    """Explicit pixel aspect override, for sizes the PAL table does not know (Custom targets)."""

    sharpen: bool = False
    sharpen_strength: float = 1.0
    background_index: int = 0
    preview: bool = False
    metrics: bool = False
    export_formats: list[ExportFormat] = field(default_factory=lambda: [ExportFormat.BIN])
    atari_colors: list[int] | None = None
    label_name: str | None = None

    def to_dict(self) -> dict:
        result = {
            "mode": self.mode.value,
            "width": self.width,
            "height": self.height,
            "palette": self.palette.value,
            "colors": self.colors,
            "dither": self.dither.value,
            "quantizer": self.quantizer.value,
            "resize": self.resize.value,
            "sharpen": self.sharpen,
            "sharpen_strength": self.sharpen_strength,
            "background_index": self.background_index,
            "preview": self.preview,
            "metrics": self.metrics,
            "export": [f.value for f in self.export_formats],
        }
        if self.atari_colors is not None:
            result["atari_colors"] = self.atari_colors
        if self.label_name is not None:
            result["label_name"] = self.label_name
        # Only written when they differ from the historical behaviour, so a saved configuration of an
        # untouched converter keeps its exact shape (and `--save-config` output stays byte-identical).
        if self.aspect is not AspectMode.DISABLED:
            result["aspect"] = self.aspect.value
        if self.pixel_aspect is not None:
            result["pixel_aspect"] = self.pixel_aspect
        return result

    @classmethod
    def from_dict(cls, data: dict) -> ConversionConfig:
        export_raw = data.get("export", ["bin"])
        if isinstance(export_raw, str):
            export_raw = [export_raw]
        return cls(
            mode=AtariMode(data.get("mode", "ANTIC_E")),
            width=int(data.get("width", 160)),
            height=int(data.get("height", 192)),
            palette=PaletteMethod(data.get("palette", "popularity")),
            colors=int(data.get("colors", 4)),
            dither=DitherMethod(data.get("dither", "floyd")),
            quantizer=QuantizerMethod(data.get("quantizer", "nearest")),
            resize=ResizeMethod(data.get("resize", "lanczos")),
            aspect=AspectMode(data.get("aspect", "disabled")),
            pixel_aspect=(
                float(data["pixel_aspect"]) if data.get("pixel_aspect") is not None else None
            ),
            sharpen=bool(data.get("sharpen", False)),
            sharpen_strength=float(data.get("sharpen_strength", 1.0)),
            background_index=int(data.get("background_index", 0)),
            preview=bool(data.get("preview", False)),
            metrics=bool(data.get("metrics", False)),
            export_formats=[ExportFormat(f) for f in export_raw],
            atari_colors=data.get("atari_colors"),
            label_name=data.get("label_name"),
        )


@dataclass(frozen=True)
class SourceIdentity:
    """Immutable identity of the loaded source image (the L0 key, plan §5.2 and §8).

    A file source is identified by its resolved path, its size and its ``mtime_ns`` - cheap enough
    for the interactive path. ``sha256`` is filled in only when the source was loaded with
    ``verify_content=True``: hashing reads the whole file, and the digest is the only field that
    notices a rewrite that keeps both the size and the modification time.

    A source installed through ``set_source_array()`` has no path, size or mtime; it is identified by
    ``array_token``, which changes on every call because the engine keeps a private frozen copy of the
    caller's array.

    Every field is an immutable scalar, so instances are hashable and usable as cache keys.
    """

    path: str | None = None
    size: int | None = None
    mtime_ns: int | None = None
    sha256: str | None = None
    array_token: int | None = None
