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
    sharpen: bool = False
    sharpen_strength: float = 1.0
    background_index: int = 0
    preview: bool = False
    metrics: bool = False
    export_formats: list[ExportFormat] = field(default_factory=lambda: [ExportFormat.BIN])

    def to_dict(self) -> dict:
        return {
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
            sharpen=bool(data.get("sharpen", False)),
            sharpen_strength=float(data.get("sharpen_strength", 1.0)),
            background_index=int(data.get("background_index", 0)),
            preview=bool(data.get("preview", False)),
            metrics=bool(data.get("metrics", False)),
            export_formats=[ExportFormat(f) for f in export_raw],
        )
