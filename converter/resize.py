"""Image resize methods."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from converter.types import ResizeMethod, RGBImage


def _lanczos_kernel(x: NDArray[np.float64], a: int = 3) -> NDArray[np.float64]:
    x = np.abs(x)
    result = np.sinc(x) * np.sinc(x / a)
    result[x >= a] = 0.0
    result[x == 0] = 1.0
    return result


def _resize_axis(
    image: NDArray[np.float64],
    target_size: int,
    axis: int,
    kernel_fn,
    kernel_radius: int,
) -> NDArray[np.float64]:
    src_size = image.shape[axis]
    if src_size == target_size:
        return image

    scale = src_size / target_size
    coords = (np.arange(target_size) + 0.5) * scale - 0.5
    output = np.zeros(
        list(image.shape[:axis]) + [target_size] + list(image.shape[axis + 1 :]),
        dtype=np.float64,
    )

    for i in range(target_size):
        center = coords[i]
        left = max(0, int(np.floor(center - kernel_radius)))
        right = min(src_size, int(np.ceil(center + kernel_radius)) + 1)
        indices = np.arange(left, right)
        weights = kernel_fn(center - indices)
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights /= weight_sum
        slice_src = np.take(image, indices, axis=axis)
        weighted = np.tensordot(weights, slice_src, axes=(0, axis))
        if axis == 0:
            output[i] = weighted
        elif axis == 1:
            output[:, i] = weighted
        else:
            output[:, :, i] = weighted
    return output


def _resize_1d_nearest(src: NDArray[np.float64], target: int, axis: int) -> NDArray[np.float64]:
    src_size = src.shape[axis]
    if src_size == target:
        return src
    indices = (np.arange(target) * src_size / target).astype(int)
    indices = np.clip(indices, 0, src_size - 1)
    return np.take(src, indices, axis=axis)


def _resize_separable(
    image: RGBImage,
    width: int,
    height: int,
    kernel_fn,
    kernel_radius: int,
) -> RGBImage:
    img = image.astype(np.float64)
    img = _resize_axis(img, height, 0, kernel_fn, kernel_radius)
    img = _resize_axis(img, width, 1, kernel_fn, kernel_radius)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def resize_nearest(image: RGBImage, width: int, height: int) -> RGBImage:
    """Nearest-neighbor resize."""
    img = image.astype(np.float64)
    img = _resize_1d_nearest(img, height, 0)
    img = _resize_1d_nearest(img, width, 1)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def resize_bilinear(image: RGBImage, width: int, height: int) -> RGBImage:
    """Bilinear interpolation resize."""

    def kernel(x: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.abs(x)
        return np.maximum(0, 1 - x)

    return _resize_separable(image, width, height, kernel, 1)


def resize_bicubic(image: RGBImage, width: int, height: int) -> RGBImage:
    """Bicubic interpolation resize."""

    def kernel(x: NDArray[np.float64]) -> NDArray[np.float64]:
        x = np.abs(x)
        result = np.zeros_like(x)
        mask = x <= 1
        result[mask] = 1.5 * x[mask] ** 3 - 2.5 * x[mask] ** 2 + 1
        mask2 = (x > 1) & (x < 2)
        result[mask2] = -0.5 * x[mask2] ** 3 + 2.5 * x[mask2] ** 2 - 4 * x[mask2] + 2
        return result

    return _resize_separable(image, width, height, kernel, 2)


def resize_lanczos(image: RGBImage, width: int, height: int, a: int = 3) -> RGBImage:
    """Lanczos resampling (default quality resize)."""

    def kernel(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return _lanczos_kernel(x, a)

    return _resize_separable(image, width, height, kernel, a)


_RESIZE_REGISTRY: dict[ResizeMethod, callable] = {
    ResizeMethod.NEAREST: resize_nearest,
    ResizeMethod.BILINEAR: resize_bilinear,
    ResizeMethod.BICUBIC: resize_bicubic,
    ResizeMethod.LANCZOS: resize_lanczos,
}


def resize_image(image: RGBImage, width: int, height: int, method: ResizeMethod) -> RGBImage:
    """Resize image using the specified method."""
    fn = _RESIZE_REGISTRY.get(method, resize_lanczos)
    return fn(image, width, height)
