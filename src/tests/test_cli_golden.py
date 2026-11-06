"""CLI golden-output regression tests.

Freeze the byte-exact behaviour of the command line interface *before* the pipeline refactoring
starts (plan task T0.1), so that any unintended behavioural change shows up as a hash mismatch.

Every case runs ``converter.main.main()`` in-process against the committed reference image and
hashes every file it produces. ``{input}`` and ``{out}`` in the argument templates are replaced
with the reference image path and a per-case temporary directory.

Regenerating the golden file is deliberate and must be reviewed in the commit:

    $env:ATARI_UPDATE_GOLDEN=1; python -m pytest src/tests/test_cli_golden.py
    python -m pytest src/tests/test_cli_golden.py
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from converter.main import main

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GOLDEN_FILE = FIXTURES_DIR / "golden" / "cli_golden.json"
REFERENCE_IMAGE = FIXTURES_DIR / "reference.png"

UPDATE_ENV_VAR = "ATARI_UPDATE_GOLDEN"
GOLDEN_VERSION = 1


@dataclass(frozen=True)
class GoldenCase:
    """One frozen CLI scenario: a human-readable note plus one or more invocations."""

    note: str
    steps: tuple[tuple[str, ...], ...]
    produces_output: bool = True


CASES: dict[str, GoldenCase] = {
    "default_floyd_bin": GoldenCase(
        note="Default invocation (floyd, popularity, BIN inferred from the extension)",
        steps=(("{input}", "{out}/screen.bin"),),
    ),
    "none_median_cut_bin": GoldenCase(
        note="dither=none with a non-default palette (B1: quantize gets range(N), not the palette)",
        steps=(("{input}", "{out}/screen.bin", "--dither", "none", "--palette", "median_cut"),),
    ),
    "none_metrics": GoldenCase(
        note="dither=none with printed metrics (B2: metrics use range(N))",
        steps=(
            (
                "{input}",
                "{out}/screen.bin",
                "--dither",
                "none",
                "--palette",
                "median_cut",
                "--metrics",
            ),
        ),
    ),
    "none_preview": GoldenCase(
        note="dither=none with preview images (B3: previews use range(N) instead of the palette)",
        steps=(
            (
                "{input}",
                "{out}/screen.bin",
                "--dither",
                "none",
                "--palette",
                "median_cut",
                "--preview",
            ),
        ),
    ),
    "none_quantizer_adaptive": GoldenCase(
        note="dither=none with the adaptive quantizer (B4: no variance map is passed)",
        steps=(
            (
                "{input}",
                "{out}/screen.bin",
                "--dither",
                "none",
                "--quantizer",
                "adaptive",
                "--palette",
                "median_cut",
            ),
        ),
    ),
    "none_quantizer_perceptual": GoldenCase(
        note="dither=none with the perceptual quantizer",
        steps=(
            (
                "{input}",
                "{out}/screen.bin",
                "--dither",
                "none",
                "--quantizer",
                "perceptual",
                "--palette",
                "median_cut",
            ),
        ),
    ),
    "png_bayer4_octree": GoldenCase(
        note="PNG export with an ordered dither and the octree palette",
        steps=(
            (
                "{input}",
                "{out}/screen.png",
                "--export",
                "png",
                "--dither",
                "bayer4",
                "--palette",
                "octree",
            ),
        ),
    ),
    "bayer8_kmeans": GoldenCase(
        note="BIN export with bayer8 and the k-means palette (fixed seed)",
        steps=(("{input}", "{out}/screen.bin", "--dither", "bayer8", "--palette", "kmeans"),),
    ),
    "all_formats_atkinson": GoldenCase(
        note="All five exporters at once, with --atari-colors and an explicit ASM label",
        steps=(
            (
                "{input}",
                "{out}/screen.bin",
                "--export",
                "bin",
                "--export",
                "png",
                "--export",
                "asm",
                "--export",
                "dta",
                "--export",
                "xex",
                "--dither",
                "atkinson",
                "--palette",
                "median_cut",
                "--atari-colors",
                "0x00,0x04,0x0a,0x0e",
                "--label-name",
                "TESTLABEL",
            ),
        ),
    ),
    "asm_default_label": GoldenCase(
        note="ASM export without --label-name (default SCREEN_<INPUT_STEM> name)",
        steps=(
            (
                "{input}",
                "{out}/screen.s",
                "--export",
                "asm",
                "--dither",
                "bayer2",
                "--palette",
                "median_cut",
            ),
        ),
    ),
    "xex_extra_mode_d": GoldenCase(
        note="Separate -x XEX output in ANTIC mode D (160x96, PNG main output)",
        steps=(
            (
                "{input}",
                "{out}/screen.png",
                "--export",
                "png",
                "--width",
                "160",
                "--height",
                "96",
                "--dither",
                "bayer2",
                "--palette",
                "median_cut",
                "-x",
                "{out}/preview_d.xex",
                "-m",
                "D",
            ),
        ),
    ),
    "xex_extra_mode_e_190": GoldenCase(
        note="Separate -x XEX output in ANTIC mode E with a 190-line screen",
        steps=(
            (
                "{input}",
                "{out}/screen.bin",
                "--height",
                "190",
                "--dither",
                "bayer2",
                "--palette",
                "median_cut",
                "-x",
                "{out}/preview_e.xex",
                "-m",
                "E",
            ),
        ),
    ),
    "xex_extra_mode_f": GoldenCase(
        note="Separate -x XEX output in ANTIC mode F (320x192, 2 colours, 1bpp)",
        steps=(
            (
                "{input}",
                "{out}/screen.png",
                "--export",
                "png",
                "--width",
                "320",
                "--height",
                "192",
                "--colors",
                "2",
                "--dither",
                "bayer2",
                "--palette",
                "median_cut",
                "-x",
                "{out}/preview_f.xex",
                "-m",
                "F",
            ),
        ),
    ),
    "sharpen_resize_perceptual": GoldenCase(
        note="Sharpen + bicubic resize + perceptual palette",
        steps=(
            (
                "{input}",
                "{out}/screen.bin",
                "--sharpen",
                "--sharpen-strength",
                "1.5",
                "--resize",
                "bicubic",
                "--palette",
                "perceptual",
            ),
        ),
    ),
    "background_index": GoldenCase(
        note="Explicit background colour index",
        steps=(
            (
                "{input}",
                "{out}/screen.bin",
                "--dither",
                "none",
                "--palette",
                "median_cut",
                "--background-index",
                "6",
            ),
        ),
    ),
    "config_roundtrip": GoldenCase(
        note="--save-config output, then a second run that loads that file",
        steps=(
            (
                "{input}",
                "{out}/first.bin",
                "--palette",
                "median_cut",
                "--dither",
                "bayer4",
                "--save-config",
                "{out}/settings.json",
            ),
            ("{input}", "{out}/second.bin", "--config", "{out}/settings.json"),
        ),
    ),
    "error_atari_colors": GoldenCase(
        note="Invalid --atari-colors exits with code 1 and a readable message",
        steps=(("{input}", "{out}/screen.bin", "--atari-colors", "0x00,,0x04"),),
        produces_output=False,
    ),
}


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _update_requested() -> bool:
    """True when the golden file should be regenerated instead of compared."""
    return os.environ.get(UPDATE_ENV_VAR) == "1"


def _load_golden() -> dict:
    """Load the golden file."""
    return json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))


def _collect_files(directory: Path) -> dict[str, dict[str, object]]:
    """Hash every file below a directory, keyed by POSIX-style relative path."""
    paths = sorted(directory.rglob("*"), key=lambda item: item.as_posix())
    return {
        path.relative_to(directory).as_posix(): {
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in paths
        if path.is_file()
    }


def _run_case(
    case: GoldenCase,
    case_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> dict:
    """Run every step of a case and collect exit codes, console output and output hashes."""
    case_dir.mkdir(parents=True, exist_ok=True)
    substitutions = {"input": REFERENCE_IMAGE.as_posix(), "out": case_dir.as_posix()}
    returncodes: list[int] = []
    stdout: list[str] = []
    stderr: list[str] = []
    for template in case.steps:
        argv = [part.format(**substitutions) for part in template]
        returncodes.append(main(argv))
        captured = capsys.readouterr()
        stdout.append(captured.out)
        stderr.append(captured.err)

    files = _collect_files(case_dir)
    if case.produces_output:
        assert files, "the case produced no output files"
    else:
        assert not files, f"the case unexpectedly produced files: {sorted(files)}"

    return {"returncodes": returncodes, "stdout": stdout, "stderr": stderr, "files": files}


def _store_case(case_id: str, observed: dict) -> None:
    """Write or refresh the golden entry of one case (stale cases are dropped)."""
    GOLDEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = _load_golden() if GOLDEN_FILE.exists() else {"cases": {}}
    data["version"] = GOLDEN_VERSION
    data["input"] = {"file": REFERENCE_IMAGE.name, "sha256": _sha256_file(REFERENCE_IMAGE)}
    data["cases"] = {key: value for key, value in data["cases"].items() if key in CASES}
    data["cases"][case_id] = {
        "note": CASES[case_id].note,
        "commands": [" ".join(step) for step in CASES[case_id].steps],
        **observed,
    }
    GOLDEN_FILE.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_cli_golden(
    case_id: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every case must reproduce the frozen exit codes, console output and file hashes."""
    observed = _run_case(CASES[case_id], tmp_path / case_id, capsys)
    if _update_requested():
        _store_case(case_id, observed)
        pytest.skip(f"golden regenerated for '{case_id}'")

    expected = _load_golden()["cases"][case_id]
    assert observed["returncodes"] == expected["returncodes"]
    assert observed["stdout"] == expected["stdout"]
    assert observed["stderr"] == expected["stderr"]
    assert observed["files"] == expected["files"]


def test_golden_file_matches_case_list() -> None:
    """The golden file must describe exactly the cases defined in this module."""
    golden = _load_golden()
    assert golden["version"] == GOLDEN_VERSION
    assert sorted(golden["cases"]) == sorted(CASES)


def test_golden_input_fixture_is_unchanged() -> None:
    """The stored hashes are only valid for the exact reference image that produced them."""
    golden = _load_golden()
    assert golden["input"]["file"] == REFERENCE_IMAGE.name
    assert golden["input"]["sha256"] == _sha256_file(REFERENCE_IMAGE)


def test_golden_detects_a_changed_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Self-check: a single altered byte in an output file must break the comparison."""
    case_id = "none_median_cut_bin"
    case_dir = tmp_path / case_id
    _run_case(CASES[case_id], case_dir, capsys)
    expected = _load_golden()["cases"][case_id]["files"]
    assert _collect_files(case_dir) == expected

    output = case_dir / "screen.bin"
    altered = bytearray(output.read_bytes())
    altered[0] ^= 0xFF
    output.write_bytes(bytes(altered))

    assert _collect_files(case_dir) != expected
