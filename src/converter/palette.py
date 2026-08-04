"""Palette selection algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
from numpy.typing import NDArray

from converter import atari_palette
from converter.types import PaletteMethod, RGBImage


class PaletteSelector(ABC):
    """Base class for palette selection strategies."""

    @abstractmethod
    def select(self, image: RGBImage, count: int, background_index: int) -> list[int]:
        """Return list of Atari palette indices."""


def _map_to_atari(indices_or_rgb: NDArray[np.int64]) -> list[int]:
    """Ensure unique Atari indices."""
    return list(dict.fromkeys(int(i) for i in indices_or_rgb))


def _fill_palette(
    selected: list[int],
    count: int,
    background_index: int,
) -> list[int]:
    """Ensure exactly `count` unique indices with background first."""
    result = [background_index]
    for idx in selected:
        if idx not in result:
            result.append(idx)
        if len(result) >= count:
            break
    for color in atari_palette.PALETTE:
        if len(result) >= count:
            break
        if color.index not in result:
            result.append(color.index)
    return result[:count]


class PopularitySelector(PaletteSelector):
    """Select most frequent colors mapped to Atari palette."""

    def select(self, image: RGBImage, count: int, background_index: int) -> list[int]:
        flat = image.reshape(-1, 3)
        atari_idx = atari_palette.nearest_atari_indices_batch(flat)
        counts = Counter(atari_idx.tolist())
        sorted_colors = [idx for idx, _ in counts.most_common()]
        if background_index in sorted_colors:
            sorted_colors.remove(background_index)
        sorted_colors.insert(0, background_index)
        return _fill_palette(sorted_colors, count, background_index)


class KMeansSelector(PaletteSelector):
    """K-Means clustering mapped to nearest Atari colors."""

    def select(self, image: RGBImage, count: int, background_index: int) -> list[int]:
        flat = image.reshape(-1, 3).astype(np.float64)
        if flat.shape[0] < count:
            return _fill_palette([background_index], count, background_index)

        rng = np.random.default_rng(42)
        indices = rng.choice(flat.shape[0], size=count, replace=False)
        centers = flat[indices].copy()

        for _ in range(20):
            diff = flat[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.sum(diff * diff, axis=2)
            labels = np.argmin(dist, axis=1)
            new_centers = np.zeros_like(centers)
            for k in range(count):
                mask = labels == k
                if mask.any():
                    new_centers[k] = flat[mask].mean(axis=0)
                else:
                    new_centers[k] = flat[rng.integers(0, flat.shape[0])]
            if np.allclose(centers, new_centers):
                break
            centers = new_centers

        atari_indices = []
        for center in centers:
            rgb = tuple(int(c) for c in np.clip(np.round(center), 0, 255))
            atari_indices.append(atari_palette.nearest_atari_index(rgb))
        return _fill_palette(atari_indices, count, background_index)


class MedianCutSelector(PaletteSelector):
    """Median cut color quantization mapped to Atari palette."""

    def select(self, image: RGBImage, count: int, background_index: int) -> list[int]:
        pixels = image.reshape(-1, 3).tolist()
        boxes = [pixels]

        while len(boxes) < count:
            boxes.sort(key=len, reverse=True)
            box = boxes.pop(0)
            if len(box) <= 1:
                boxes.append(box)
                break
            arr = np.array(box, dtype=np.float64)
            ranges = arr.max(axis=0) - arr.min(axis=0)
            channel = int(np.argmax(ranges))
            arr = arr[arr[:, channel].argsort()]
            mid = len(arr) // 2
            boxes.append(arr[:mid].tolist())
            boxes.append(arr[mid:].tolist())

        centers = []
        for box in boxes[:count]:
            arr = np.array(box, dtype=np.float64)
            mean = arr.mean(axis=0)
            rgb = tuple(int(c) for c in np.clip(np.round(mean), 0, 255))
            centers.append(atari_palette.nearest_atari_index(rgb))
        return _fill_palette(centers, count, background_index)


class OctreeSelector(PaletteSelector):
    """Octree color quantization mapped to Atari palette."""

    class _Node:
        def __init__(self) -> None:
            self.children: dict[tuple[int, int, int], OctreeSelector._Node] = {}
            self.pixels: list[tuple[int, int, int]] = []
            self.is_leaf = False

    def _insert(self, node: _Node, pixel: tuple[int, int, int], depth: int) -> None:
        if depth >= 4 or len(node.pixels) < 8:
            node.pixels.append(pixel)
            node.is_leaf = True
            return
        shift = 7 - depth
        key = ((pixel[0] >> shift) & 1, (pixel[1] >> shift) & 1, (pixel[2] >> shift) & 1)
        if key not in node.children:
            node.children[key] = self._Node()
        self._insert(node.children[key], pixel, depth + 1)

    def _collect_leaves(self, node: _Node, leaves: list[_Node]) -> None:
        if node.is_leaf or not node.children:
            leaves.append(node)
            return
        for child in node.children.values():
            self._collect_leaves(child, leaves)

    def select(self, image: RGBImage, count: int, background_index: int) -> list[int]:
        root = self._Node()
        pixels = [tuple(map(int, p)) for p in image.reshape(-1, 3)]
        for pixel in pixels:
            self._insert(root, pixel, 0)

        leaves: list[OctreeSelector._Node] = []
        self._collect_leaves(root, leaves)
        leaves.sort(key=lambda n: len(n.pixels), reverse=True)

        centers = []
        for leaf in leaves[:count]:
            arr = np.array(leaf.pixels, dtype=np.float64)
            mean = arr.mean(axis=0)
            rgb = tuple(int(c) for c in np.clip(np.round(mean), 0, 255))
            centers.append(atari_palette.nearest_atari_index(rgb))
        return _fill_palette(centers, count, background_index)


class PerceptualSelector(PaletteSelector):
    """Perceptual optimization with luminance-weighted Atari mapping."""

    def select(self, image: RGBImage, count: int, background_index: int) -> list[int]:
        flat = image.reshape(-1, 3).astype(np.float64)
        lum = 0.2126 * flat[:, 0] + 0.7152 * flat[:, 1] + 0.0722 * flat[:, 2]

        palette_arr = atari_palette.PALETTE_ARRAY.astype(np.float64)
        pal_lum = (
            0.2126 * palette_arr[:, 0]
            + 0.7152 * palette_arr[:, 1]
            + 0.0722 * palette_arr[:, 2]
        )

        scores = np.zeros(len(atari_palette.PALETTE), dtype=np.float64)
        for i, color in enumerate(atari_palette.PALETTE):
            rgb = palette_arr[i]
            diff = flat - rgb
            rgb_dist = np.sum(diff * diff, axis=1)
            lum_diff = (lum - pal_lum[i]) ** 2
            weighted = rgb_dist + 2.0 * lum_diff
            scores[i] = np.sum(1.0 / (1.0 + weighted))

        ranked = np.argsort(-scores)
        selected = [int(i) for i in ranked[: count * 2]]
        return _fill_palette(selected, count, background_index)


_SELECTOR_REGISTRY: dict[PaletteMethod, PaletteSelector] = {
    PaletteMethod.POPULARITY: PopularitySelector(),
    PaletteMethod.KMEANS: KMeansSelector(),
    PaletteMethod.MEDIAN_CUT: MedianCutSelector(),
    PaletteMethod.OCTREE: OctreeSelector(),
    PaletteMethod.PERCEPTUAL: PerceptualSelector(),
}


def select_palette(
    image: RGBImage,
    count: int,
    method: PaletteMethod,
    background_index: int = 0,
) -> list[int]:
    """Select Atari palette indices for the image."""
    selector = _SELECTOR_REGISTRY.get(method, PopularitySelector())
    return selector.select(image, count, background_index)


def palette_rgb(indices: list[int]) -> NDArray[np.uint8]:
    """Return N×3 RGB array for selected palette indices."""
    return np.array([atari_palette.get_color(i).rgb for i in indices], dtype=np.uint8)
