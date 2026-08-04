"""Integration test for full pipeline."""

from pathlib import Path

import numpy as np
from PIL import Image

from converter.exporters.atari_bin_export import ANTIC_E_BYTES, unpack_bin
from converter.pipeline import convert_image, export_result
from converter.types import ConversionConfig, DitherMethod, ExportFormat


def test_full_pipeline(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    img = Image.new("RGB", (320, 240))
    pixels = np.zeros((240, 320, 3), dtype=np.uint8)
    for y in range(240):
        for x in range(320):
            pixels[y, x] = [x % 256, y % 256, (x + y) % 256]
    Image.fromarray(pixels).save(input_path)

    config = ConversionConfig(
        width=160,
        height=192,
        dither=DitherMethod.FLOYD,
        metrics=True,
        export_formats=[ExportFormat.BIN, ExportFormat.PNG],
    )
    result = convert_image(input_path, config)
    assert result.indexed.shape == (192, 160)
    assert len(result.palette_indices) == 4
    assert result.metrics is not None
    assert result.metrics.colors_used <= 4

    output = tmp_path / "output.bin"
    paths = export_result(result, output, config)
    assert any(p.suffix == ".bin" for p in paths)
    bin_path = next(p for p in paths if p.suffix == ".bin")
    data = bin_path.read_bytes()
    assert len(data) == ANTIC_E_BYTES
    restored = unpack_bin(data)
    np.testing.assert_array_equal(restored, result.indexed)
