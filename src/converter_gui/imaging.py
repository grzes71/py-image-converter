"""Helpers that turn conversion output into Qt images.

The conversion engine hands out frozen, read-only NumPy arrays and Qt must never write into them, so
the image returned by :func:`ndarray_to_qimage` owns its pixels: Qt is given a contiguous view of the
array, the data is then deep-copied by ``QImage.copy()`` and only that copy is returned. Consequently
neither the lifetime nor a later mutation of the source array can affect a QImage that is already on
screen (plan §6.1 and §8.1).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage

#: Channel count -> explicit QImage format. RGB888 stores R, G, B and RGBA8888 stores R, G, B, A,
#: which is exactly the byte order of a NumPy ``uint8`` array with three or four channels.
CHANNELS_TO_FORMAT = {
    3: QImage.Format.Format_RGB888,
    4: QImage.Format.Format_RGBA8888,
}


def ndarray_to_qimage(array: np.ndarray) -> QImage:
    """Return a QImage with the exact pixels of a ``uint8`` RGB or RGBA array.

    Supported inputs (nothing is rescaled, converted or reordered):

    ==================  =================  ============================
    shape               NumPy dtype        resulting QImage format
    ==================  =================  ============================
    ``(H, W, 3)``       ``uint8``          ``QImage.Format_RGB888``
    ``(H, W, 4)``       ``uint8``          ``QImage.Format_RGBA8888``
    ==================  =================  ============================

    Non-contiguous views (slices, reversed rows, transposes) are accepted: the data is made contiguous
    before Qt sees it. The input array is only ever read, and the returned image owns its pixels, so
    the caller may drop or modify the array afterwards.

    Raises ``TypeError`` for a non-array argument or a dtype other than ``uint8`` (values are never
    converted implicitly) and ``ValueError`` for a wrong number of dimensions, an unsupported channel
    count or an empty image.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"expected a numpy.ndarray, got {type(array).__name__}")
    if array.dtype != np.uint8:
        raise TypeError(
            f"expected a uint8 array, got {array.dtype}; pixels are never converted implicitly"
        )
    if array.ndim != 3:
        raise ValueError(
            f"expected a (height, width, channels) array, got {array.ndim} dimensions "
            f"with shape {array.shape}"
        )

    height, width, channels = array.shape
    image_format = CHANNELS_TO_FORMAT.get(channels)
    if image_format is None:
        raise ValueError(f"expected 3 (RGB) or 4 (RGBA) channels, got {channels}")
    if height == 0 or width == 0:
        raise ValueError(f"expected a non-empty image, got {width}x{height}")

    contiguous = np.ascontiguousarray(array)
    image = QImage(
        contiguous.data,
        width,
        height,
        int(contiguous.strides[0]),  # bytesPerLine of the source buffer
        image_format,
    )
    if image.isNull():
        raise ValueError(f"Qt rejected a {width}x{height} image with {channels} channels")
    # ``copy()`` deep-copies the pixels into a Qt-owned buffer, so the image no longer aliases the
    # NumPy memory (which may be a read-only engine buffer) and may be kept and painted indefinitely.
    return image.copy()
