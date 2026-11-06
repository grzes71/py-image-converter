"""Export of the displayed conversion to files (T4.4).

The GUI never serialises anything itself: this module decides *what* to write (the format table and the
compatibility verdict of ``converter.exporters.check_export_compatibility``), *where* (the canonical
suffix of each format) and hands the work to ``converter.pipeline.export_result``.

Two configurations travel together and are never merged (plan §7.5): ``payload.displayed.result`` is the
image on screen, ``payload.displayed.config`` the configuration that produced it, and
``payload.export_config`` the export-only settings that are current right now. Only the format is
replaced by the one the user picked; the label name, the hardware colours and the preview flag are taken
from the current settings, which is why a change to them never requires a reconversion.

Nothing here imports Qt: the window passes paths and a payload, and the pure helpers stay testable on
their own.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from converter.exporters import check_export_compatibility
from converter.pipeline import export_result
from converter.types import ConversionConfig, ConversionResult, ExportFormat

if TYPE_CHECKING:  # pragma: no cover - annotations only, so this module stays Qt-free
    from converter_gui.controllers.conversion_controller import DisplayedConversion

EXPORT_FORMATS: tuple[ExportFormat, ...] = (
    ExportFormat.PNG,
    ExportFormat.BIN,
    ExportFormat.ASM,
    ExportFormat.DTA,
    ExportFormat.XEX,
)
"""Every format the engine can write, in the order the dialog offers them."""

EXPORT_SUFFIX: dict[ExportFormat, str] = {
    ExportFormat.PNG: ".png",
    ExportFormat.BIN: ".bin",
    ExportFormat.ASM: ".s",
    ExportFormat.DTA: ".dta",
    ExportFormat.XEX: ".xex",
}
"""The suffix ``export_result`` writes for each format (ASM is ``.s``, like the CLI)."""

EXPORT_DESCRIPTION: dict[ExportFormat, str] = {
    ExportFormat.PNG: "PNG image",
    ExportFormat.BIN: "Atari screen (BIN)",
    ExportFormat.ASM: "Assembler source (ASM)",
    ExportFormat.DTA: "Atari data (DTA)",
    ExportFormat.XEX: "Atari executable (XEX)",
}

ASM_SUFFIX_ALIASES: tuple[str, ...] = (".s", ".asm")
"""``.asm`` is accepted because the CLI accepts it; the writer always emits ``.s``."""


@dataclass(frozen=True)
class ExportPayload:
    """Everything an export needs: the image on screen and the current export settings.

    The two configurations answer different questions, which is why they are kept apart:

    * ``displayed`` is the conversion that is on screen. Its ``config`` produced the image, so it
      contains image-relevant settings only - export-only changes never trigger a conversion and never
      reach it.
    * ``export_config`` is the user's configuration right now, including the export-only fields
      (``export_formats``, ``preview``, ``label_name``, ``atari_colors``). It may be newer than the
      displayed result: the image is still valid, only the way it will be written changed.
    """

    displayed: DisplayedConversion
    export_config: ConversionConfig


# --- formats and paths --------------------------------------------------


def format_from_suffix(path: str | Path) -> ExportFormat | None:
    """The format a path's suffix asks for (``None`` for an unknown or missing suffix)."""
    suffix = Path(path).suffix.lower()
    if suffix in ASM_SUFFIX_ALIASES:
        return ExportFormat.ASM
    for fmt, canonical in EXPORT_SUFFIX.items():
        if suffix == canonical:
            return fmt
    return None


def path_with_suffix(path: str | Path, fmt: ExportFormat) -> Path:
    """``path`` with the suffix ``export_result`` will really use for ``fmt``."""
    path = Path(path)
    if fmt is ExportFormat.ASM and path.suffix.lower() in ASM_SUFFIX_ALIASES:
        return path
    if path.suffix.lower() == EXPORT_SUFFIX[fmt]:
        return path
    return path.with_suffix(EXPORT_SUFFIX[fmt])


def suggested_path(directory: str | Path | None, stem: str, fmt: ExportFormat) -> Path:
    """Where the save dialog starts: the remembered folder and the source stem."""
    folder = Path(directory) if directory is not None else Path.cwd()
    return folder / f"{stem or 'screen'}{EXPORT_SUFFIX[fmt]}"


# --- compatibility ------------------------------------------------------


def availability(
    result: ConversionResult, *, antic_mode: str | None = None
) -> dict[ExportFormat, str | None]:
    """Map every format to ``None`` when it can be written, or to the reason it cannot."""
    return {
        fmt: check_export_compatibility(result, fmt, antic_mode=antic_mode)
        for fmt in EXPORT_FORMATS
    }


def refusal_reason(
    result: ConversionResult, fmt: ExportFormat, *, antic_mode: str | None = None
) -> str | None:
    """Why ``fmt`` cannot be written for ``result`` (``None`` when it can)."""
    return check_export_compatibility(result, fmt, antic_mode=antic_mode)


# --- the save dialog ----------------------------------------------------


def dialog_filters(formats: Sequence[ExportFormat]) -> list[tuple[str, ExportFormat]]:
    """``(filter text, format)`` pairs for a ``QFileDialog``, one per available format."""
    filters: list[tuple[str, ExportFormat]] = []
    for fmt in formats:
        if fmt is ExportFormat.ASM:
            patterns = " ".join(f"*{suffix}" for suffix in ASM_SUFFIX_ALIASES)
        else:
            patterns = f"*{EXPORT_SUFFIX[fmt]}"
        filters.append((f"{EXPORT_DESCRIPTION[fmt]} ({patterns})", fmt))
    return filters


def chosen_format(
    path: str | Path, selected_filter: str, formats: Sequence[ExportFormat]
) -> ExportFormat | None:
    """Resolve the format from what the user typed, falling back to the selected filter.

    A typed suffix always wins - including one whose format cannot be written, so the caller can explain
    the incompatibility instead of silently writing something else.
    """
    typed = format_from_suffix(path)
    if typed is not None:
        return typed
    for text, fmt in dialog_filters(formats):
        if text == selected_filter:
            return fmt
    return formats[0] if formats else None


# --- writing ------------------------------------------------------------


def export_displayed(payload: ExportPayload, fmt: ExportFormat, path: str | Path) -> list[Path]:
    """Write the displayed result as ``fmt``; the export-only settings come from the payload.

    Raises :class:`ValueError` when the combination is incompatible (the same verdict the dialog uses, so
    the GUI can never write something the engine would reject) and lets exporter errors through, so the
    window can report them.
    """
    reason = refusal_reason(payload.displayed.result, fmt)
    if reason is not None:
        raise ValueError(f"cannot export {EXPORT_DESCRIPTION[fmt]}: {reason}")
    target = path_with_suffix(path, fmt)
    config = dataclasses.replace(payload.export_config, export_formats=[fmt])
    return export_result(payload.displayed.result, target, config)
