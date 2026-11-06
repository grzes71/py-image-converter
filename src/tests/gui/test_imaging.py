"""Tests for the NumPy -> QImage helper (T2.2).

The pixel comparisons read the bytes Qt actually stored (``constBits()``, with Qt's row padding
removed) instead of going through ``pixelColor``, so a channel swap such as RGB/BGR is caught as well.
Skipped when PySide6 is missing, like the rest of ``src/tests/gui``.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="PySide6 is not installed (install the 'gui' extra)")

from PySide6.QtGui import QImage  # noqa: E402

from converter.engine import ConversionEngine  # noqa: E402
from converter.types import ConversionConfig, DitherMethod, PaletteMethod  # noqa: E402
from converter_gui.imaging import ndarray_to_qimage  # noqa: E402

REFERENCE_IMAGE = Path(__file__).resolve().parents[1] / "fixtures" / "reference.png"

# Deliberately asymmetric values: a channel swap or a stray alpha handling cannot go unnoticed.
RGB_SAMPLE = np.array(
    [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[200, 100, 50], [0, 255, 128], [17, 33, 49]],
    ],
    dtype=np.uint8,
)
RGBA_SAMPLE = np.array(
    [
        [[10, 20, 30, 40], [50, 60, 70, 255], [80, 90, 100, 0]],
        [[110, 120, 130, 128], [140, 150, 160, 1], [170, 180, 190, 254]],
    ],
    dtype=np.uint8,
)


def _image_pixels(image: QImage) -> np.ndarray:
    """The logical pixel bytes of a QImage as an ``(H, W, channels)`` array.

    ``bytesPerLine()`` can exceed ``width * channels`` (Qt pads its rows), so the rows are re-packed
    before the comparison.
    """
    channels = 3 if image.format() == QImage.Format.Format_RGB888 else 4
    width, height, stride = image.width(), image.height(), image.bytesPerLine()
    raw = bytes(image.constBits())
    rows = [raw[y * stride : y * stride + width * channels] for y in range(height)]
    return np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(height, width, channels)


# --- pixel equality -----------------------------------------------------


def test_rgb_pixels_are_exact() -> None:
    image = ndarray_to_qimage(RGB_SAMPLE)

    assert image.format() == QImage.Format.Format_RGB888
    assert (image.width(), image.height()) == (3, 2)
    assert not image.isNull()
    np.testing.assert_array_equal(_image_pixels(image), RGB_SAMPLE)


def test_rgba_pixels_are_exact_including_alpha() -> None:
    image = ndarray_to_qimage(RGBA_SAMPLE)

    assert image.format() == QImage.Format.Format_RGBA8888
    assert (image.width(), image.height()) == (3, 2)
    np.testing.assert_array_equal(_image_pixels(image), RGBA_SAMPLE)
    assert image.pixelColor(1, 0).alpha() == 255
    assert image.pixelColor(2, 0).alpha() == 0
    assert image.pixelColor(0, 1).alpha() == 128


def test_rgb_channels_keep_their_order() -> None:
    """``pixelColor`` must report the same R, G and B values that NumPy stored."""
    image = ndarray_to_qimage(RGB_SAMPLE)

    colour = image.pixelColor(0, 0)
    assert (colour.red(), colour.green(), colour.blue()) == (1, 2, 3)
    assert colour.alpha() == 255, "an RGB image is opaque"


@pytest.mark.parametrize("sample", [RGB_SAMPLE, RGBA_SAMPLE], ids=["rgb", "rgba"])
def test_dimensions_match_the_input(sample: np.ndarray) -> None:
    image = ndarray_to_qimage(sample)

    assert image.width() == sample.shape[1]
    assert image.height() == sample.shape[0]
    assert image.bytesPerLine() >= sample.shape[1] * sample.shape[2]


# --- non-contiguous arrays ----------------------------------------------

_GRID = np.arange(8 * 6 * 3, dtype=np.uint8).reshape(8, 6, 3)


def _view(name: str) -> np.ndarray:
    """Return a named non-contiguous view of a 8x6 RGB grid."""
    views = {
        "strided-both-axes": _GRID[::2, ::2],
        "reversed-columns": _GRID[:, ::-1],
        "reversed-rows": _GRID[::-1],
        "transposed": _GRID.transpose(1, 0, 2),
        "sub-window": _GRID[1:5, 2:5],
        "single-pixel-stride": _GRID[::3, ::3],
    }
    return views[name]


@pytest.mark.parametrize(
    "name",
    [
        "strided-both-axes",
        "reversed-columns",
        "reversed-rows",
        "transposed",
        "sub-window",
        "single-pixel-stride",
    ],
)
def test_non_contiguous_views_are_converted_exactly(name: str) -> None:
    view = _view(name)
    assert not view.flags["C_CONTIGUOUS"], "the test is pointless for a contiguous array"
    expected = view.copy()

    image = ndarray_to_qimage(view)

    assert (image.height(), image.width()) == expected.shape[:2]
    assert image.format() == QImage.Format.Format_RGB888
    np.testing.assert_array_equal(_image_pixels(image), expected)


def test_reversed_rows_are_not_flipped_back() -> None:
    """A reversed view must be displayed exactly as indexed, not silently normalised."""
    view = _GRID[::-1]

    image = ndarray_to_qimage(view)

    first_row = _image_pixels(image)[0]
    np.testing.assert_array_equal(first_row, _GRID[-1])


# --- ownership and memory isolation -------------------------------------


def test_image_survives_deleting_the_source_array() -> None:
    array = RGB_SAMPLE.copy()
    expected = array.copy()

    image = ndarray_to_qimage(array)
    del array
    gc.collect()

    np.testing.assert_array_equal(_image_pixels(image), expected)


def test_mutating_the_source_after_conversion_does_not_change_the_image() -> None:
    array = RGB_SAMPLE.copy()
    expected = array.copy()

    image = ndarray_to_qimage(array)
    array[:] = 0  # the caller keeps working with its own buffer
    array[0, 0] = (255, 255, 255)
    gc.collect()

    np.testing.assert_array_equal(_image_pixels(image), expected)


def test_conversion_does_not_modify_the_input() -> None:
    array = RGBA_SAMPLE.copy()
    before = array.tobytes()

    ndarray_to_qimage(array)
    ndarray_to_qimage(np.ascontiguousarray(array)).width()  # a second, already contiguous call

    assert array.tobytes() == before
    np.testing.assert_array_equal(array, RGBA_SAMPLE)


def test_the_image_does_not_alias_the_source_buffer() -> None:
    array = RGB_SAMPLE.copy()

    image = ndarray_to_qimage(array)

    qt_pixels = np.frombuffer(image.constBits(), dtype=np.uint8)
    assert not np.shares_memory(qt_pixels, array), "QImage must own its pixels, not borrow them"


def test_read_only_arrays_are_accepted() -> None:
    frozen = np.frombuffer(RGB_SAMPLE.tobytes(), dtype=np.uint8).reshape(RGB_SAMPLE.shape)
    assert frozen.flags.writeable is False

    image = ndarray_to_qimage(frozen)

    np.testing.assert_array_equal(_image_pixels(image), RGB_SAMPLE)
    assert frozen.flags.writeable is False
    with pytest.raises(ValueError):
        frozen[0, 0, 0] = 0


def test_read_only_input_is_not_unlocked_by_the_conversion() -> None:
    """Freezing stays intact: the helper copies instead of asking Qt to write in place."""
    frozen = np.frombuffer(RGBA_SAMPLE.tobytes(), dtype=np.uint8).reshape(RGBA_SAMPLE.shape)

    ndarray_to_qimage(frozen)

    with pytest.raises(ValueError):
        frozen.setflags(write=True)


# --- invalid input ------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "exception", "needle"),
    [
        pytest.param([[0, 0, 0]], TypeError, "numpy.ndarray", id="not-an-array"),
        pytest.param(None, TypeError, "numpy.ndarray", id="none"),
        pytest.param(np.zeros((2, 2, 3), dtype=np.float32), TypeError, "float32", id="float32"),
        pytest.param(np.zeros((2, 2, 3), dtype=np.uint16), TypeError, "uint16", id="uint16"),
        pytest.param(np.zeros((2, 2), dtype=np.uint8), ValueError, "dimensions", id="two-dims"),
        pytest.param(
            np.zeros((2, 2, 3, 1), dtype=np.uint8), ValueError, "dimensions", id="four-dims"
        ),
        pytest.param(
            np.zeros((2, 2, 2), dtype=np.uint8), ValueError, "channels", id="two-channels"
        ),
        pytest.param(
            np.zeros((2, 2, 5), dtype=np.uint8), ValueError, "channels", id="five-channels"
        ),
        pytest.param(
            np.zeros((0, 4, 3), dtype=np.uint8), ValueError, "non-empty", id="zero-height"
        ),
        pytest.param(np.zeros((4, 0, 3), dtype=np.uint8), ValueError, "non-empty", id="zero-width"),
        pytest.param(np.zeros((4, 4, 3), dtype=bool), TypeError, "bool", id="bool"),
    ],
)
def test_invalid_input_is_rejected(value: object, exception: type[Exception], needle: str) -> None:
    with pytest.raises(exception, match=needle):
        ndarray_to_qimage(value)  # type: ignore[arg-type]


def test_invalid_input_is_never_converted_implicitly() -> None:
    """A float array must fail loudly instead of being rounded into a uint8 image."""
    floats = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float64)

    with pytest.raises(TypeError):
        ndarray_to_qimage(floats)

    assert floats.dtype == np.float64, "the input is left untouched"


def test_a_two_channel_array_is_not_treated_as_greyscale() -> None:
    with pytest.raises(ValueError, match="3 .*RGB.* or 4 .*RGBA"):
        ndarray_to_qimage(np.zeros((3, 3, 2), dtype=np.uint8))


# --- integration with the conversion engine -----------------------------


def test_engine_output_can_be_displayed_without_touching_the_engine() -> None:
    engine = ConversionEngine()
    engine.set_source(REFERENCE_IMAGE)
    config = ConversionConfig(
        width=160,
        height=192,
        dither=DitherMethod.NONE,
        palette=PaletteMethod.MEDIAN_CUT,
    )
    result = engine.convert(config)
    frozen = result.source_resized
    assert frozen.flags.writeable is False, "the engine hands out frozen buffers"
    before = frozen.tobytes()

    image = ndarray_to_qimage(frozen)

    assert (image.width(), image.height()) == (160, 192)
    assert image.format() == QImage.Format.Format_RGB888
    np.testing.assert_array_equal(_image_pixels(image), np.ascontiguousarray(frozen))
    assert frozen.tobytes() == before, "the conversion must not modify the engine buffer"
    with pytest.raises(ValueError):
        frozen[0, 0, 0] = 0


def test_engine_buffer_can_be_released_after_conversion() -> None:
    """Displaying engine output must not extend the life of the engine's arrays."""
    engine = ConversionEngine()
    engine.set_source_array(np.random.default_rng(7).integers(0, 256, (16, 24, 3), dtype=np.uint8))
    config = ConversionConfig(
        width=24,
        height=16,
        dither=DitherMethod.NONE,
        palette=PaletteMethod.MEDIAN_CUT,
    )
    frozen = engine.convert(config).source_resized
    expected = np.array(frozen)

    image = ndarray_to_qimage(frozen)
    del frozen
    del engine
    gc.collect()

    np.testing.assert_array_equal(_image_pixels(image), expected)
