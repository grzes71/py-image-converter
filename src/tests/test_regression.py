"""Reference image regression tests."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from converter.pipeline import convert_image
from converter.types import ConversionConfig, DitherMethod

FIXTURES_DIR = Path(__file__).parent / "fixtures"
BASELINES_FILE = FIXTURES_DIR / "baselines.json"


def _create_reference_image(path: Path) -> None:
    """Create a simple gradient reference image."""
    w, h = 160, 192
    pixels = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            pixels[y, x] = [x, y, (x + y) // 2]
    Image.fromarray(pixels).save(path)


@pytest.fixture(scope="module")
def reference_image(tmp_path_factory) -> Path:
    FIXTURES_DIR.mkdir(exist_ok=True)
    path = FIXTURES_DIR / "reference.png"
    if not path.exists():
        _create_reference_image(path)
    return path


def test_reference_conversion_metrics(reference_image: Path) -> None:
    config = ConversionConfig(dither=DitherMethod.FLOYD, metrics=True)
    result = convert_image(reference_image, config)
    assert result.metrics is not None
    assert result.metrics.psnr > 0
    assert result.metrics.colors_used <= 4


def test_baseline_regression(reference_image: Path) -> None:
    """Compare metrics against stored baseline (within tolerance)."""
    config = ConversionConfig(dither=DitherMethod.FLOYD, metrics=True)
    result = convert_image(reference_image, config)
    assert result.metrics is not None

    current = {
        "mse": round(result.metrics.mse, 2),
        "psnr": round(result.metrics.psnr, 2),
        "ssim": round(result.metrics.ssim, 4),
        "colors_used": result.metrics.colors_used,
    }

    if not BASELINES_FILE.exists():
        FIXTURES_DIR.mkdir(exist_ok=True)
        BASELINES_FILE.write_text(json.dumps(current, indent=2), encoding="utf-8")
        return

    baseline = json.loads(BASELINES_FILE.read_text(encoding="utf-8"))
    assert current["colors_used"] == baseline["colors_used"]
    assert abs(current["mse"] - baseline["mse"]) < 50
    assert abs(current["ssim"] - baseline["ssim"]) < 0.05
