"""Optional unsharp mask sharpening."""

from __future__ import annotations

import numpy as np

from converter.types import RGBImage


def _box_blur_channel(channel: np.ndarray, radius: int = 1) -> np.ndarray:
    """Simple box blur for unsharp mask."""
    padded = np.pad(channel, radius, mode="edge")
    kernel_size = 2 * radius + 1
    result = np.zeros_like(channel, dtype=np.float64)
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            result += padded[dy : dy + channel.shape[0], dx : dx + channel.shape[1]]
    return result / (kernel_size * kernel_size)


def unsharp_mask(image: RGBImage, strength: float = 1.0, radius: int = 1) -> RGBImage:
    """Apply unsharp mask sharpening with configurable strength."""
    if strength <= 0:
        return image

    img = image.astype(np.float64)
    blurred = np.stack([_box_blur_channel(img[:, :, c], radius) for c in range(3)], axis=2)
    sharpened = img + strength * (img - blurred)
    return np.clip(np.round(sharpened), 0, 255).astype(np.uint8)


def apply_sharpen(image: RGBImage, enabled: bool, strength: float = 1.0) -> RGBImage:
    """Apply sharpening if enabled."""
    if not enabled:
        return image
    return unsharp_mask(image, strength=strength)
