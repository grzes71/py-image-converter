"""Source identity and content-verification tests (plan §5.2, §8 and T1.6).

Timestamps are always set explicitly with ``os.utime(..., ns=...)``, so every check is deterministic:
nothing here depends on filesystem timestamp granularity, on how fast the test runs, or on a sleep.
"""

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import converter.engine as engine_module
from converter.engine import ConversionEngine
from converter.types import ConversionConfig, DitherMethod, PaletteMethod, SourceIdentity

REFERENCE_IMAGE = Path(__file__).parent / "fixtures" / "reference.png"

# One fixed modification time (~2023-11-14T22:13:20Z), in nanoseconds.
STAMP = 1_700_000_000_000_000_000
LATER = STAMP + 5_000_000_000  # five seconds later


def _image(offset: int, size: int = 16) -> np.ndarray:
    """A small deterministic colour gradient; different offsets give different pixels."""
    ramp = np.arange(size, dtype=np.uint16)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :, 0] = (ramp * 8 + offset) % 256
    image[:, :, 1] = (ramp * 3) % 256
    image[:, :, 2] = offset % 256
    return image


def _write_bmp(path: Path, image: np.ndarray, *, mtime_ns: int = STAMP) -> None:
    """Write an uncompressed BMP and force a known mtime (equal dimensions give equal file sizes)."""
    Image.fromarray(image, "RGB").save(path, format="BMP")
    os.utime(path, ns=(mtime_ns, mtime_ns))


def _config() -> ConversionConfig:
    return ConversionConfig(
        width=16,
        height=16,
        dither=DitherMethod.NONE,
        palette=PaletteMethod.MEDIAN_CUT,
    )


# --- the recorded identity ----------------------------------------------


def test_file_identity_is_recorded_and_stable_across_conversions(tmp_path: Path) -> None:
    path = tmp_path / "source.bmp"
    _write_bmp(path, _image(0))
    engine = ConversionEngine()

    engine.set_source(path)

    identity = engine.source_identity
    assert identity is not None
    assert identity.path == str(path.resolve())
    assert identity.size == path.stat().st_size
    assert identity.mtime_ns == STAMP
    assert identity.sha256 is None, "the interactive path must not hash the file"
    assert identity.array_token is None

    config = _config()
    engine.convert(config)
    engine.convert(config)

    assert engine.source_identity is identity, "conversions must not recompute the identity"


def test_reloading_the_same_file_keeps_the_identity(tmp_path: Path) -> None:
    path = tmp_path / "source.bmp"
    _write_bmp(path, _image(0))
    engine = ConversionEngine()

    engine.set_source(path)
    first = engine.source_identity
    engine.set_source(path)

    assert engine.source_identity == first


def test_identity_changes_when_the_source_changes(tmp_path: Path) -> None:
    first_path = tmp_path / "first.bmp"
    second_path = tmp_path / "second.bmp"
    _write_bmp(first_path, _image(0))
    _write_bmp(second_path, _image(0))
    engine = ConversionEngine()

    engine.set_source(first_path)
    first = engine.source_identity
    engine.set_source(second_path)
    second = engine.source_identity

    assert first is not None and second is not None
    assert first != second
    assert (first.path, second.path) == (str(first_path.resolve()), str(second_path.resolve()))


def test_identity_is_frozen_and_hashable() -> None:
    identity = SourceIdentity(path="source.bmp", size=10, mtime_ns=5)

    assert {identity: "cache key"}[
        SourceIdentity(path="source.bmp", size=10, mtime_ns=5)
    ] == "cache key"
    assert identity != SourceIdentity(path="source.bmp", size=10, mtime_ns=6)
    with pytest.raises(FrozenInstanceError):
        identity.size = 11  # type: ignore[misc]


# --- change detection ---------------------------------------------------


def test_standard_mode_detects_a_new_mtime(tmp_path: Path) -> None:
    path = tmp_path / "source.bmp"
    _write_bmp(path, _image(0))
    engine = ConversionEngine()
    engine.set_source(path)

    assert engine.source_is_unchanged() is True

    os.utime(path, ns=(LATER, LATER))

    assert engine.source_is_unchanged() is False


def test_both_modes_detect_a_size_change(tmp_path: Path) -> None:
    path = tmp_path / "source.bmp"
    _write_bmp(path, _image(0))
    engine = ConversionEngine()
    engine.set_source(path, verify_content=True)
    size_before = path.stat().st_size

    _write_bmp(path, _image(0, size=24), mtime_ns=STAMP)

    assert path.stat().st_size != size_before
    assert engine.source_is_unchanged() is False
    assert engine.source_is_unchanged(verify_content=True) is False


def test_verified_mode_detects_a_rewrite_with_identical_size_and_mtime(tmp_path: Path) -> None:
    """The case the cheap triple cannot see: same byte count, same mtime, different pixels."""
    path = tmp_path / "source.bmp"
    _write_bmp(path, _image(0))
    engine = ConversionEngine()
    engine.set_source(path, verify_content=True)
    identity = engine.source_identity
    assert identity is not None and identity.sha256 is not None
    size_before = path.stat().st_size

    _write_bmp(path, _image(180), mtime_ns=STAMP)  # same dimensions, same mtime, other pixels

    assert path.stat().st_size == size_before
    assert engine.source_is_unchanged(verify_content=True) is False
    assert engine.source_is_unchanged() is True, "documented blindness of the cheap triple"


def test_verified_mode_treats_a_touch_as_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "source.bmp"
    _write_bmp(path, _image(0))
    engine = ConversionEngine()
    engine.set_source(path, verify_content=True)

    os.utime(path, ns=(LATER, LATER))  # a bare touch: new mtime, same bytes

    assert engine.source_is_unchanged() is False
    assert engine.source_is_unchanged(verify_content=True) is True


def test_standard_and_verified_modes_disagree_on_purpose(tmp_path: Path) -> None:
    """Both engines see the same file; only the recorded digest changes the two answers."""
    path = tmp_path / "source.bmp"
    _write_bmp(path, _image(0))
    cheap = ConversionEngine()
    verified = ConversionEngine()
    cheap.set_source(path)
    verified.set_source(path, verify_content=True)

    cheap_identity = cheap.source_identity
    verified_identity = verified.source_identity
    assert cheap_identity is not None and verified_identity is not None
    assert cheap_identity.sha256 is None
    assert verified_identity.sha256 is not None and len(verified_identity.sha256) == 64
    assert cheap_identity.path == verified_identity.path
    assert cheap_identity.size == verified_identity.size

    os.utime(path, ns=(LATER, LATER))  # touch: the content did not change

    assert cheap.source_is_unchanged() is False
    assert verified.source_is_unchanged(verify_content=True) is True

    _write_bmp(path, _image(180), mtime_ns=LATER)  # and now a real rewrite

    assert cheap.source_is_unchanged() is False
    assert verified.source_is_unchanged(verify_content=True) is False


def test_without_a_recorded_digest_only_the_cheap_fields_are_compared(tmp_path: Path) -> None:
    """Documented limitation: a same-size, same-mtime rewrite is invisible without a digest."""
    path = tmp_path / "source.bmp"
    _write_bmp(path, _image(0))
    engine = ConversionEngine()
    engine.set_source(path)

    _write_bmp(path, _image(180), mtime_ns=STAMP)

    assert engine.source_is_unchanged() is True
    assert engine.source_is_unchanged(verify_content=True) is True, "nothing to compare against"


def test_unchanged_answers_for_a_missing_or_absent_source(tmp_path: Path) -> None:
    engine = ConversionEngine()
    assert engine.source_is_unchanged() is False, "no source has been set"

    path = tmp_path / "source.bmp"
    _write_bmp(path, _image(0))
    engine.set_source(path, verify_content=True)
    assert engine.source_is_unchanged() is True
    assert engine.source_is_unchanged(verify_content=True) is True

    path.unlink()

    assert engine.source_is_unchanged() is False
    assert engine.source_is_unchanged(verify_content=True) is False


def test_reference_image_is_unchanged_right_after_loading() -> None:
    engine = ConversionEngine()
    engine.set_source(REFERENCE_IMAGE, verify_content=True)

    assert engine.source_is_unchanged() is True
    assert engine.source_is_unchanged(verify_content=True) is True


# --- hashing is opt-in --------------------------------------------------


def test_hashing_only_happens_when_it_is_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§5.2: the interactive path must never pay for SHA-256."""
    path = tmp_path / "source.bmp"
    _write_bmp(path, _image(0))
    hashed: list[Path] = []

    def fake_digest(target: Path) -> str:
        hashed.append(target)
        return "0" * 64

    monkeypatch.setattr(engine_module, "_file_sha256", fake_digest)
    engine = ConversionEngine()

    engine.set_source(path)
    assert hashed == [], "set_source() must not hash by default"
    identity = engine.source_identity
    assert identity is not None and identity.sha256 is None

    engine.set_source(path, verify_content=True)
    assert len(hashed) == 1
    identity = engine.source_identity
    assert identity is not None and identity.sha256 == "0" * 64


# --- array sources ------------------------------------------------------


def test_array_source_identity_and_unchanged() -> None:
    engine = ConversionEngine()
    assert engine.source_is_unchanged() is False

    image = _image(40)
    engine.set_source_array(image)

    first = engine.source_identity
    assert first is not None
    assert first.array_token == 1
    assert (first.path, first.size, first.mtime_ns, first.sha256) == (None, None, None, None)
    assert engine.source_is_unchanged() is True
    assert engine.source_is_unchanged(verify_content=True) is True

    engine.set_source_array(image)  # the same input again counts as a new source

    second = engine.source_identity
    assert second is not None
    assert second.array_token == 2
    assert second != first


# --- cache invalidation -------------------------------------------------


def test_changing_the_source_invalidates_the_derived_caches(tmp_path: Path) -> None:
    first_path = tmp_path / "first.bmp"
    second_path = tmp_path / "second.bmp"
    _write_bmp(first_path, _image(0))
    _write_bmp(second_path, _image(200))
    engine = ConversionEngine()
    config = _config()

    engine.set_source(first_path)
    first = engine.convert(config)
    engine.set_source(second_path)
    second = engine.convert(config)

    assert not np.array_equal(first.indexed, second.indexed)

    fresh = ConversionEngine()
    fresh.set_source(second_path)
    np.testing.assert_array_equal(second.indexed, fresh.convert(config).indexed)


def test_reloading_after_an_external_rewrite_picks_up_the_new_pixels(tmp_path: Path) -> None:
    path = tmp_path / "source.bmp"
    _write_bmp(path, _image(0))
    engine = ConversionEngine()
    config = _config()
    engine.set_source(path)
    before = engine.convert(config)

    _write_bmp(path, _image(200), mtime_ns=LATER)
    engine.set_source(path, verify_content=True)
    after = engine.convert(config)

    assert not np.array_equal(before.indexed, after.indexed)
