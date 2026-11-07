"""Conversion parameter panel (T2.4, plan §6.2).

The panel edits the engine-supported part of :class:`~converter.types.ConversionConfig` and nothing
else: ``metrics``, ``preview``, ``export_formats`` and ``label_name`` belong to the main window and the
export dialog (T4.4), so they are carried through untouched. The panel never converts anything itself -
it only reports the configuration it currently represents.

Two pieces of GUI-only state sit on top of the config: the selected target
(:mod:`converter_gui.targets`) and the PNG-only export flag, which widens the allowed colour range from
the hardware limit of 4 to :data:`converter_gui.limits.MAX_PNG_COLORS`. Both are explicit API rather
than hidden config fields, so a configuration built here still round-trips through the CLI.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Iterator, Sequence
from typing import TypeVar

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from converter.aspect import pal_screen, pixel_aspect_for
from converter.config import validate_config
from converter.hardware_colors import GTIA_REGISTERS, derived_registers, mode_for_size
from converter.types import (
    MAX_ATARI_COLORS,
    AspectMode,
    ConversionConfig,
    DitherMethod,
    ExportFormat,
    PaletteMethod,
    QuantizerMethod,
    ResizeMethod,
)

from converter_gui.limits import MAX_PNG_COLORS
from converter_gui.targets import (
    ALL_TARGETS,
    CUSTOM,
    TARGETS_BY_KEY,
    TargetSpec,
    apply_target,
    match_target,
)

_EnumT = TypeVar("_EnumT", ResizeMethod, PaletteMethod, QuantizerMethod, DitherMethod, AspectMode)

MIN_WIDTH, MAX_WIDTH = 8, 640
MIN_HEIGHT, MAX_HEIGHT = 8, 480
MIN_STRENGTH, MAX_STRENGTH = 0.0, 5.0
MIN_BACKGROUND, MAX_BACKGROUND = 0, 255
MAX_PIXEL_ASPECT = 4.00
"""Upper bound of the explicit pixel aspect field (0.00 means "not set")."""

ASPECT_GROUP_TITLE = "Aspect Ratio Correction"

#: The three correction modes with the hint shown in the combo's tooltip. The wording says what the
#: implementation does; nothing here describes a mode the engine does not have.
ASPECT_CHOICES: tuple[tuple[AspectMode, str, str], ...] = (
    (
        AspectMode.DISABLED,
        "Disabled",
        "Stretch the whole image onto the target grid - the previous behaviour, no correction",
    ),
    (
        AspectMode.FIT,
        "Preserve Aspect Ratio (Fit)",
        "Keep the whole image and fill the margins with the background colour",
    ),
    (
        AspectMode.CROP,
        "Preserve Aspect Ratio (Crop)",
        "Fill the target grid completely and trim the image where it overflows",
    ),
)

ATARI_REGISTER_COUNT = 5
"""The exporters read ``atari_colors[:5]`` (COLOR0..COLOR4), so the panel edits exactly five."""
ATARI_REGISTER_MAX = 255

ATARI_GROUP_TITLE = "Hardware colour registers (export only)"

#: One line per mode explaining where the derived values end up; wording only, the rules live in
#: :mod:`converter.hardware_colors`.
_ATARI_LAYOUT_HINTS: dict[str | None, str] = {
    "D": "ANTIC D: colour 0 → COLBK (COLOR4), colours 1-3 → COLPF0-COLPF2 (COLOR0-COLOR2).",
    "E": "ANTIC E: colour 0 → COLBK (COLOR4), colours 1-3 → COLPF0-COLPF2 (COLOR0-COLOR2).",
    "F": (
        "ANTIC F: COLOR2 (COLPF2) carries the hue and the background luminance, "
        "COLOR1 (COLPF1) the foreground luminance."
    ),
}

#: Dithering methods grouped by family. The group names and hints describe what the implementations in
#: :mod:`converter.dithering` do: ordered dithering compares against a repeating threshold matrix,
#: error diffusion propagates the quantisation error to neighbouring pixels and ``adaptive`` classifies
#: 16x16 tiles and delegates to none/bayer4/floyd/atkinson.
DITHER_SECTIONS: tuple[tuple[str, tuple[DitherMethod, ...]], ...] = (
    ("No dithering", (DitherMethod.NONE,)),
    (
        "Error diffusion",
        (
            DitherMethod.FLOYD,
            DitherMethod.JARVIS,
            DitherMethod.STUCKI,
            DitherMethod.SIERRA,
            DitherMethod.SIERRA_LITE,
            DitherMethod.BURKES,
            DitherMethod.ATKINSON,
        ),
    ),
    ("Ordered (threshold matrix)", (DitherMethod.BAYER2, DitherMethod.BAYER4, DitherMethod.BAYER8)),
    ("Adaptive", (DitherMethod.ADAPTIVE,)),
    (
        "Atari-authentic patterns",
        (
            DitherMethod.ATARI_FALSE_FLOYD,
            DitherMethod.ATARI_TWO_ROW_SIERRA,
            DitherMethod.ATARI_DIAGONAL,
            DitherMethod.ATARI_BAYER2V2,
            DitherMethod.ATARI_BAYER2X4,
            DitherMethod.ATARI_NON_RECTANGULAR8,
        ),
    ),
)

_DITHER_HINTS: dict[DitherMethod, str] = {
    DitherMethod.NONE: "flat colours, no pattern - fastest",
    DitherMethod.FLOYD: "error diffusion - smooth gradients, slower",
    DitherMethod.JARVIS: "error diffusion, wider kernel - slower",
    DitherMethod.STUCKI: "error diffusion, large kernel - slower",
    DitherMethod.SIERRA: "error diffusion, three rows - slower",
    DitherMethod.SIERRA_LITE: "error diffusion, two rows - slower",
    DitherMethod.BURKES: "error diffusion, two rows - slower",
    DitherMethod.ATKINSON: "error diffusion, lighter - slower",
    DitherMethod.BAYER2: "ordered 2x2 matrix - fast",
    DitherMethod.BAYER4: "ordered 4x4 matrix - fast",
    DitherMethod.BAYER8: "ordered 8x8 matrix - fast",
    DitherMethod.ADAPTIVE: "per-tile choice of none/bayer4/floyd/atkinson - slowest",
    DitherMethod.ATARI_FALSE_FLOYD: "two-row diffusion in the Atari style",
    DitherMethod.ATARI_TWO_ROW_SIERRA: "two-row Sierra in the Atari style",
    DitherMethod.ATARI_DIAGONAL: "diagonal pattern in the Atari style",
    DitherMethod.ATARI_BAYER2V2: "2x2 ordered matrix, Atari variant",
    DitherMethod.ATARI_BAYER2X4: "2x4 ordered matrix, Atari variant",
    DitherMethod.ATARI_NON_RECTANGULAR8: "8x8 matrix with Atari line offsets",
}


def _pretty(method: ResizeMethod | PaletteMethod | QuantizerMethod | DitherMethod) -> str:
    """Display name of an enum member, derived from its value so no name is invented here."""
    return str(method.value).replace("_", " ").title()


def _atari_register(colors: list[int] | None, index: int) -> int:
    """Value of register ``index`` for a stored ``atari_colors`` list (padded with the last colour)."""
    if not colors:
        return 0
    return colors[index] if index < len(colors) else colors[-1]


class ParameterPanel(QWidget):
    """Edits the engine-supported parameters of a :class:`ConversionConfig`."""

    config_changed = Signal(ConversionConfig)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = ConversionConfig()
        self._png_only: bool | None = None  # None = derive from the carried export formats
        self._palette_colors: list[tuple[int, int, int]] | None = None
        self._palette_size: tuple[int, int] | None = None
        self._manual_registers: list[int] = [0] * ATARI_REGISTER_COUNT
        self._build_ui()
        self._populate(self._config)

    # --- public API -----------------------------------------------------

    def set_config(self, cfg: ConversionConfig) -> None:
        """Show ``cfg`` in every widget without emitting ``config_changed``.

        The argument is never modified: the panel keeps a copy with its own list fields. Values the
        widgets cannot represent (a width below 8, more colours than the current export mode allows)
        are clamped to the nearest representable value, which is what ``config()`` then reports.
        """
        problems = validate_config(cfg)
        if problems:
            raise ValueError("; ".join(problems))
        self._config = dataclasses.replace(
            cfg,
            export_formats=list(cfg.export_formats),
            atari_colors=list(cfg.atari_colors) if cfg.atari_colors is not None else None,
        )
        self._populate(self._config)

    def config(self) -> ConversionConfig:
        """The configuration the widgets currently represent (fields outside the panel are kept)."""
        cfg = dataclasses.replace(
            self._config,
            width=self.width_spin.value(),
            height=self.height_spin.value(),
            resize=self._selected(self.resize_combo, ResizeMethod),
            aspect=self._selected(self.aspect_combo, AspectMode),
            pixel_aspect=self._pixel_aspect_override(),
            sharpen=self.sharpen_check.isChecked(),
            sharpen_strength=self.sharpen_strength_spin.value(),
            palette=self._selected(self.palette_combo, PaletteMethod),
            colors=self.colors_spin.value(),
            background_index=self.background_spin.value(),
            dither=self._selected(self.dither_combo, DitherMethod),
            quantizer=self._selected(self.quantizer_combo, QuantizerMethod),
            atari_colors=self._atari_colors_from_widgets(),
        )
        problems = validate_config(cfg)
        if problems:  # a widget range was bypassed somewhere: never hand out an invalid config
            raise ValueError("; ".join(problems))
        return cfg

    @property
    def selected_target(self) -> TargetSpec:
        """The target currently shown in the combo (GUI state, not part of the config)."""
        return TARGETS_BY_KEY.get(str(self.target_combo.currentData()), CUSTOM)

    @property
    def png_only(self) -> bool:
        """Whether the panel currently allows more than the hardware four colours."""
        return self._effective_png_only()

    def set_png_only(self, png_only: bool) -> None:
        """Declare whether the current export selection is PNG-only (the export dialog owns this)."""
        self._png_only = bool(png_only)
        self._apply_colors_limit()

    def set_palette_colors(
        self,
        colors: Sequence[Sequence[int]] | None,
        *,
        size: tuple[int, int] | None = None,
    ) -> None:
        """Provide the palette of the current result so the background swatch can show its colour.

        Without a palette the swatch stays neutral and explains why; the panel never guesses. The same
        call refreshes the derived hardware registers, so they always describe the conversion that is
        on screen. ``size`` is the ``(width, height)`` of that result: it decides the ANTIC mode of the
        registers and therefore has to come from the result, not from the spin boxes (they may already
        show newer settings).
        """
        self._palette_colors = (
            [tuple(int(channel) for channel in color[:3]) for color in colors] if colors else None
        )
        self._palette_size = (int(size[0]), int(size[1])) if size is not None else None
        self._update_background_swatch()
        self._apply_derived_registers()
        self._update_atari_status()

    @property
    def background_color(self) -> QColor | None:
        """Colour of the selected background index, or ``None`` while the palette is unknown."""
        if not self._palette_colors:
            return None
        index = self.background_spin.value()
        if not 0 <= index < len(self._palette_colors):
            return None
        return QColor(*self._palette_colors[index])

    # --- construction ---------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_output_group())
        layout.addWidget(self._build_aspect_group())
        layout.addWidget(self._build_resize_group())
        layout.addWidget(self._build_sharpen_group())
        layout.addWidget(self._build_palette_group())
        layout.addWidget(self._build_dither_group())
        layout.addWidget(self._build_quantizer_group())
        layout.addWidget(self._build_atari_group())
        layout.addStretch(1)

    def _build_output_group(self) -> QGroupBox:
        group = QGroupBox("Output", self)
        form = QFormLayout(group)

        self.target_combo = QComboBox(group)
        for target in ALL_TARGETS:
            self.target_combo.addItem(target.label, target.key)
        self.target_combo.currentIndexChanged.connect(self._on_target_changed)
        form.addRow("Target", self.target_combo)

        self.width_spin = QSpinBox(group)
        self.width_spin.setRange(MIN_WIDTH, MAX_WIDTH)
        self.width_spin.setToolTip("Output width in pixels (editable for the Custom target)")
        self.width_spin.valueChanged.connect(self._on_size_changed)
        self.height_spin = QSpinBox(group)
        self.height_spin.setRange(MIN_HEIGHT, MAX_HEIGHT)
        self.height_spin.setToolTip("Output height in pixels (editable for the Custom target)")
        self.height_spin.valueChanged.connect(self._on_size_changed)

        size_row = QWidget(group)
        size_layout = QHBoxLayout(size_row)
        size_layout.setContentsMargins(0, 0, 0, 0)
        size_layout.addWidget(self.width_spin)
        size_layout.addWidget(QLabel("×", size_row))
        size_layout.addWidget(self.height_spin)
        size_layout.addStretch(1)
        form.addRow("Size", size_row)
        return group

    def _build_aspect_group(self) -> QGroupBox:
        group = QGroupBox(ASPECT_GROUP_TITLE, self)
        form = QFormLayout(group)

        self.aspect_combo = QComboBox(group)
        for mode, label, hint in ASPECT_CHOICES:
            self.aspect_combo.addItem(label, mode.value)
            self.aspect_combo.setItemData(
                self.aspect_combo.count() - 1, hint, Qt.ItemDataRole.ToolTipRole
            )
        self.aspect_combo.currentIndexChanged.connect(self._on_aspect_changed)
        form.addRow("Correction", self.aspect_combo)

        self.pixel_aspect_spin = QDoubleSpinBox(group)
        self.pixel_aspect_spin.setRange(0.0, MAX_PIXEL_ASPECT)
        self.pixel_aspect_spin.setSingleStep(0.01)
        self.pixel_aspect_spin.setDecimals(2)
        self.pixel_aspect_spin.setSpecialValueText("not set")
        self.pixel_aspect_spin.setToolTip(
            "Width/height of one pixel; a preset uses its PAL value, a Custom size needs it here"
        )
        self.pixel_aspect_spin.valueChanged.connect(self._on_pixel_aspect_changed)
        form.addRow("Pixel aspect", self.pixel_aspect_spin)

        self.aspect_hint = QLabel(group)
        self.aspect_hint.setWordWrap(True)
        form.addRow(self.aspect_hint)
        return group

    def _build_resize_group(self) -> QGroupBox:
        group = QGroupBox("Resize", self)
        form = QFormLayout(group)
        self.resize_combo = QComboBox(group)
        for method in ResizeMethod:
            self.resize_combo.addItem(_pretty(method), method.value)
        self.resize_combo.currentIndexChanged.connect(self._on_value_changed)
        form.addRow("Filter", self.resize_combo)
        return group

    def _build_sharpen_group(self) -> QGroupBox:
        group = QGroupBox("Sharpen", self)
        form = QFormLayout(group)
        self.sharpen_check = QCheckBox("Sharpen before quantizing", group)
        self.sharpen_check.toggled.connect(self._on_sharpen_toggled)
        form.addRow(self.sharpen_check)

        self.sharpen_strength_spin = QDoubleSpinBox(group)
        self.sharpen_strength_spin.setRange(MIN_STRENGTH, MAX_STRENGTH)
        self.sharpen_strength_spin.setSingleStep(0.1)
        self.sharpen_strength_spin.setDecimals(2)
        self.sharpen_strength_spin.valueChanged.connect(self._on_value_changed)
        form.addRow("Strength", self.sharpen_strength_spin)
        return group

    def _build_palette_group(self) -> QGroupBox:
        group = QGroupBox("Palette", self)
        form = QFormLayout(group)

        self.palette_combo = QComboBox(group)
        for method in PaletteMethod:
            self.palette_combo.addItem(_pretty(method), method.value)
        self.palette_combo.currentIndexChanged.connect(self._on_value_changed)
        form.addRow("Selector", self.palette_combo)

        self.colors_spin = QSpinBox(group)
        self.colors_spin.setRange(2, MAX_ATARI_COLORS)
        self.colors_spin.valueChanged.connect(self._on_value_changed)
        form.addRow("Colours", self.colors_spin)

        self.background_spin = QSpinBox(group)
        self.background_spin.setRange(MIN_BACKGROUND, MAX_BACKGROUND)
        self.background_spin.valueChanged.connect(self._on_background_changed)
        self.background_swatch = QFrame(group)
        self.background_swatch.setFixedSize(18, 18)
        self.background_swatch.setFrameShape(QFrame.Shape.Box)
        self.background_index_label = QLabel(group)

        background_row = QWidget(group)
        background_layout = QHBoxLayout(background_row)
        background_layout.setContentsMargins(0, 0, 0, 0)
        background_layout.addWidget(self.background_spin)
        background_layout.addWidget(self.background_swatch)
        background_layout.addWidget(self.background_index_label)
        background_layout.addStretch(1)
        form.addRow("Background index", background_row)
        return group

    def _build_dither_group(self) -> QGroupBox:
        group = QGroupBox("Dithering", self)
        form = QFormLayout(group)
        self.dither_combo = QComboBox(group)
        for section, methods in DITHER_SECTIONS:
            self.dither_combo.addItem(f"—— {section} ——", None)
            header = self.dither_combo.model().item(self.dither_combo.count() - 1)
            if header is not None:
                header.setEnabled(False)
            for method in methods:
                text = f"{_pretty(method)} — {_DITHER_HINTS[method]}"
                self.dither_combo.addItem(text, method.value)
                self.dither_combo.setItemData(
                    self.dither_combo.count() - 1,
                    _DITHER_HINTS[method],
                    Qt.ItemDataRole.ToolTipRole,
                )
        self.dither_combo.currentIndexChanged.connect(self._on_dither_changed)
        form.addRow("Algorithm", self.dither_combo)
        return group

    def _build_quantizer_group(self) -> QGroupBox:
        group = QGroupBox("Quantization", self)
        form = QFormLayout(group)
        self.quantizer_combo = QComboBox(group)
        for method in QuantizerMethod:
            self.quantizer_combo.addItem(_pretty(method), method.value)
        self.quantizer_combo.currentIndexChanged.connect(self._on_value_changed)
        self.quantizer_combo.setToolTip("Only used by the engine when dithering is off")
        form.addRow("Quantizer", self.quantizer_combo)
        return group

    def _build_atari_group(self) -> QGroupBox:
        group = QGroupBox(ATARI_GROUP_TITLE, self)
        layout = QVBoxLayout(group)

        hint = QLabel(
            "Export only: these values are written to the GTIA colour registers (COLOR0-COLOR4) of "
            "the generated .xex/.s files. They do not change the converted image or the preview.",
            group,
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.atari_palette_check = QCheckBox("Use converter palette", group)
        self.atari_palette_check.setToolTip(
            "Checked: the registers follow the converted palette, mapped to the nearest Atari colours"
        )
        self.atari_palette_check.toggled.connect(self._on_atari_mode_toggled)
        layout.addWidget(self.atari_palette_check)

        registers = QGridLayout()
        self.atari_color_spins: list[QSpinBox] = []
        for index in range(ATARI_REGISTER_COUNT):
            spin = QSpinBox(group)
            spin.setRange(0, ATARI_REGISTER_MAX)
            spin.setDisplayIntegerBase(16)
            spin.setPrefix("$")
            spin.setToolTip(
                f"COLOR{index} shadow (hex), written to {GTIA_REGISTERS[index]}"
            )
            spin.valueChanged.connect(self._on_value_changed)
            self.atari_color_spins.append(spin)
            target = QLabel(f"COLOR{index} → {GTIA_REGISTERS[index]}", group)
            target.setToolTip(f"Shadows $02C{4 + index} and feeds {GTIA_REGISTERS[index]}")
            registers.addWidget(target, index // 3, ((index % 3) * 3))
            registers.addWidget(spin, index // 3, ((index % 3) * 3) + 1)
            registers.setColumnStretch(((index % 3) * 3) + 2, 1)
        layout.addLayout(registers)

        self.atari_status = QLabel(group)
        self.atari_status.setWordWrap(True)
        layout.addWidget(self.atari_status)
        return group

    # --- state ----------------------------------------------------------

    def _populate(self, cfg: ConversionConfig) -> None:
        """Fill every widget from ``cfg`` with all signals blocked."""
        with self._signals_blocked():
            self._select_target(match_target(cfg.width, cfg.height))
            self.width_spin.setValue(cfg.width)
            self.height_spin.setValue(cfg.height)
            self._select_data(self.resize_combo, cfg.resize)
            self._select_data(self.aspect_combo, cfg.aspect)
            self.pixel_aspect_spin.setValue(self._displayed_pixel_aspect(cfg))

            self.sharpen_check.setChecked(cfg.sharpen)
            self.sharpen_strength_spin.setValue(cfg.sharpen_strength)
            self._select_data(self.palette_combo, cfg.palette)
            self.colors_spin.setMaximum(self._colors_limit())
            self.colors_spin.setValue(cfg.colors)
            self.background_spin.setValue(cfg.background_index)
            self._select_data(self.dither_combo, cfg.dither)
            self._select_data(self.quantizer_combo, cfg.quantizer)
            self.atari_palette_check.setChecked(cfg.atari_colors is None)
            for index, spin in enumerate(self.atari_color_spins):
                spin.setValue(_atari_register(cfg.atari_colors, index))
            if cfg.atari_colors is not None:
                self._manual_registers = [
                    spin.value() for spin in self.atari_color_spins
                ]
        self._refresh_enabled_states()
        self._update_background_swatch()

    def _refresh_enabled_states(self) -> None:
        """Enable exactly what the current selection makes meaningful."""
        custom = self.selected_target.is_custom
        self.width_spin.setEnabled(custom)
        self.height_spin.setEnabled(custom)
        self.pixel_aspect_spin.setEnabled(
            custom and self._selected(self.aspect_combo, AspectMode) is not AspectMode.DISABLED
        )
        self.sharpen_strength_spin.setEnabled(self.sharpen_check.isChecked())
        self.quantizer_combo.setEnabled(
            self._selected(self.dither_combo, DitherMethod) is DitherMethod.NONE
        )
        use_converter_palette = self.atari_palette_check.isChecked()
        for spin in self.atari_color_spins:
            spin.setEnabled(not use_converter_palette)
        self._apply_derived_registers()
        self._update_atari_status()
        self._update_aspect_hint()

    def _effective_png_only(self) -> bool:
        """Explicit flag if set, otherwise derived from the carried export formats."""
        if self._png_only is not None:
            return self._png_only
        formats = self._config.export_formats
        return bool(formats) and all(fmt is ExportFormat.PNG for fmt in formats)

    def _colors_limit(self) -> int:
        return MAX_PNG_COLORS if self._effective_png_only() else MAX_ATARI_COLORS

    def _apply_colors_limit(self) -> None:
        """Re-apply the colour cap after the flag changed, emitting only if the value moved."""
        before = self.colors_spin.value()
        with self._signals_blocked():
            self.colors_spin.setMaximum(self._colors_limit())
        if self.colors_spin.value() != before:
            self._emit()

    def _atari_colors_from_widgets(self) -> list[int] | None:
        if self.atari_palette_check.isChecked():
            return None
        return [spin.value() for spin in self.atari_color_spins]

    def _pixel_aspect_override(self) -> float | None:
        """Explicit PAL pixel aspect, which only a Custom size can carry.

        A preset always resolves its factor from :data:`converter.aspect.PAL_SCREENS`, so switching
        between ANTIC D/E/F really does change the value instead of leaving a stale copy behind.
        """
        if not self.selected_target.is_custom:
            return None
        value = round(self.pixel_aspect_spin.value(), 2)
        return value if value > 0.0 else None

    @staticmethod
    def _displayed_pixel_aspect(cfg: ConversionConfig) -> float:
        """The factor to show in the spin box (``0.0`` when nothing is known or set)."""
        factor = pixel_aspect_for(cfg.width, cfg.height, cfg.pixel_aspect)
        return 0.0 if factor is None else factor

    def _atari_automatic(self) -> bool:
        """True while the registers follow the converted palette (the checkbox is on)."""
        return self.atari_palette_check.isChecked()

    def _atari_mode(self) -> str | None:
        """ANTIC mode the export of the conversion on screen would use (``None`` when unknown)."""
        width, height = (
            self._palette_size
            if self._palette_size is not None
            else (self.width_spin.value(), self.height_spin.value())
        )
        try:
            return mode_for_size(width, height)
        except ValueError:
            return None

    def _apply_derived_registers(self) -> None:
        """Show the values the export will really write while the converter palette is used.

        The panel only mirrors what :func:`converter.hardware_colors.resolved_registers` computes for
        the result on screen - the mapping itself lives in the core, so the two cannot disagree.
        """
        if not self._atari_automatic() or self._palette_colors is None:
            return
        mode = self._atari_mode()
        if mode is None:
            return
        derived = derived_registers(self._palette_colors, mode)
        with self._signals_blocked():
            for spin, value in zip(self.atari_color_spins, derived, strict=True):
                spin.setValue(value)

    def _update_atari_status(self) -> None:
        """Say where the values come from, without adding another control."""
        mode = self._atari_mode()
        if not self._atari_automatic():
            self.atari_status.setText(
                "Manual: these values are written to COLOR0-COLOR4 as they are."
            )
            return
        if self._palette_colors is None:
            self.atari_status.setText(
                "Automatic: nearest Atari colours of the converted palette - shown after the first "
                "conversion."
            )
            return
        colours = len(self._palette_colors)
        text = (
            f"Automatic: {colours} converted palette colour(s) mapped to the nearest Atari "
            "colours (CIE94 match on the measured PAL palette)."
        )
        self.atari_status.setText(text + " " + _ATARI_LAYOUT_HINTS.get(mode, ""))

    def _update_aspect_hint(self) -> None:
        """Say what the selected mode does with the current size, without guessing a factor."""
        mode = self._selected(self.aspect_combo, AspectMode)
        if mode is AspectMode.DISABLED:
            self.aspect_hint.setText(
                "The image is stretched onto the whole target grid - no proportions are preserved."
            )
            return
        width = self.width_spin.value()
        height = self.height_spin.value()
        pixel_aspect = pixel_aspect_for(width, height, self._pixel_aspect_override())
        if pixel_aspect is None:
            self.aspect_hint.setText(
                f"No PAL pixel aspect is known for {width}×{height}: the correction stays off. "
                "Pick a preset or set the pixel aspect above."
            )
            return
        screen = pal_screen(width, height)
        source = f"{screen.name}, " if screen is not None else ""
        self.aspect_hint.setText(
            f"PAL pixel aspect {pixel_aspect:g}:1 ({source}{width}×{height}). "
            "The result preview shows it at this shape."
        )

    # --- slots ----------------------------------------------------------

    def _on_target_changed(self, *_: object) -> None:
        target = self.selected_target
        if not target.is_custom:
            updated = apply_target(self.config(), target)
            with self._signals_blocked():
                self.width_spin.setValue(updated.width)
                self.height_spin.setValue(updated.height)
                self.colors_spin.setValue(updated.colors)
                # The factor follows the target, it is never carried over from another geometry.
                self.pixel_aspect_spin.setValue(target.pixel_aspect or 0.0)
        self._refresh_enabled_states()
        self._emit()

    def _on_size_changed(self, *_: object) -> None:
        """A custom size starts with its table factor, or with none at all - never a stale one."""
        with self._signals_blocked():
            self._select_target(CUSTOM)
            self.pixel_aspect_spin.setValue(
                pixel_aspect_for(self.width_spin.value(), self.height_spin.value()) or 0.0
            )
        self._refresh_enabled_states()
        self._emit()

    def _on_sharpen_toggled(self, *_: object) -> None:
        self._refresh_enabled_states()
        self._emit()

    def _on_dither_changed(self, *_: object) -> None:
        self._refresh_enabled_states()
        self._emit()

    def _on_aspect_changed(self, *_: object) -> None:
        self._refresh_enabled_states()
        self._emit()

    def _on_pixel_aspect_changed(self, *_: object) -> None:
        self._update_aspect_hint()
        self._emit()

    def _on_atari_mode_toggled(self, *_: object) -> None:
        """Switch between automatic and manual registers without losing the manual values."""
        if self._atari_automatic():
            self._manual_registers = [spin.value() for spin in self.atari_color_spins]
        else:
            with self._signals_blocked():
                for spin, value in zip(
                    self.atari_color_spins, self._manual_registers, strict=True
                ):
                    spin.setValue(value)
        self._refresh_enabled_states()
        self._emit()

    def _on_background_changed(self, *_: object) -> None:
        self._update_background_swatch()
        self._emit()

    def _on_value_changed(self, *_: object) -> None:
        self._emit()

    def _emit(self) -> None:
        self.config_changed.emit(self.config())

    # --- helpers --------------------------------------------------------

    def _select_target(self, target: TargetSpec) -> None:
        index = self.target_combo.findData(target.key)
        if index < 0:
            raise ValueError(f"unknown target {target.key!r}")
        self.target_combo.setCurrentIndex(index)

    @staticmethod
    def _select_data(
        combo: QComboBox,
        value: ResizeMethod | PaletteMethod | QuantizerMethod | DitherMethod,
    ) -> None:
        """Select the entry whose user data is the value of ``value``."""
        index = combo.findData(value.value)
        if index < 0:
            raise ValueError(f"{value!r} is not offered by this combo box")
        combo.setCurrentIndex(index)

    @staticmethod
    def _selected(
        combo: QComboBox,
        enum_type: type[_EnumT],
    ) -> _EnumT:
        """Rebuild the enum member selected in ``combo`` (its data is stored as a plain string)."""
        data = combo.currentData()
        if data is None:
            raise ValueError("no value selected")
        return enum_type(data)

    def _update_background_swatch(self) -> None:
        index = self.background_spin.value()
        self.background_index_label.setText(f"index {index}")
        color = self.background_color
        if color is None:
            self.background_swatch.setStyleSheet("background-color: palette(mid);")
            self.background_swatch.setToolTip(
                "The background colour is only known once a conversion has produced its palette"
            )
            return
        self.background_swatch.setStyleSheet(
            f"background-color: rgb({color.red()}, {color.green()}, {color.blue()});"
        )
        self.background_swatch.setToolTip(f"palette[{index}] = {color.name()}")

    def _signal_widgets(self) -> tuple[QWidget, ...]:
        return (
            self.target_combo,
            self.width_spin,
            self.height_spin,
            self.aspect_combo,
            self.pixel_aspect_spin,
            self.resize_combo,
            self.sharpen_check,
            self.sharpen_strength_spin,
            self.palette_combo,
            self.colors_spin,
            self.background_spin,
            self.dither_combo,
            self.quantizer_combo,
            self.atari_palette_check,
            *self.atari_color_spins,
        )

    @contextlib.contextmanager
    def _signals_blocked(self) -> Iterator[None]:
        """Silence every widget while the panel updates itself programmatically."""
        with contextlib.ExitStack() as stack:
            for widget in self._signal_widgets():
                stack.enter_context(QSignalBlocker(widget))
            yield
