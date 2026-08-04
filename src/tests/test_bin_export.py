"""Tests for ANTIC E BIN export."""

import numpy as np

from converter.exporters.atari_bin_export import ANTIC_E_BYTES, pack_antic_e, unpack_bin


def test_bin_round_trip() -> None:
    indexed = np.random.default_rng(42).integers(0, 4, (192, 160), dtype=np.uint8)
    packed = pack_antic_e(indexed)
    assert len(packed) == ANTIC_E_BYTES
    restored = unpack_bin(packed)
    np.testing.assert_array_equal(restored, indexed)


def test_pack_specific_pixels() -> None:
    indexed = np.zeros((192, 160), dtype=np.uint8)
    indexed[0, 0] = 3
    indexed[0, 1] = 2
    indexed[0, 2] = 1
    indexed[0, 3] = 0
    packed = pack_antic_e(indexed)
    assert packed[0] == 0b11100100  # pixels 3,2,1,0 MSB first
