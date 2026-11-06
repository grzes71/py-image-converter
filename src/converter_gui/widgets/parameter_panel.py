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

from converter.config import validate_config
from converter.types import (
    MAX_ATARI_COLORS,
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

_EnumT = TypeVar("_EnumT", ResizeMethod, PaletteMethod, QuantizerMethod, DitherMethod)

MIN_WIDTH, MAX_WIDTH = 8, 640
MIN_HEIGHT, MAX_HEIGHT = 8, 480
MIN_STRENGTH, MAX_STRENGTH = 0.0, 5.0
MIN_BACKGROUND, MAX_BACKGROUND = 0, 255

ATARI_REGISTER_COUNT = 5
"""The exporters read ``atari_colors[:5]`` (COLOR0..COLOR4), so the panel edits exactly five."""
ATARI_REGISTER_MAX = 255

ATARI_GROUP_TITLE = "Hardware colour registers (export only)"

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

    def set_palette_colors(self, colors: Sequence[Sequence[int]] | None) -> None:
        """Provide the palette of the current result so the background swatch can show its colour.

        Without a palette the swatch stays neutral and explains why; the panel never guesses.
        """
        self._palette_colors = (
            [tuple(int(channel) for channel in color[:3]) for color in colors] if colors else None
        )
        self._update_background_swatch()

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
            "Checked: the exporters use the converted palette colours instead of these registers"
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
            spin.setToolTip(f"GTIA colour register COLOR{index} (hex)")
            spin.valueChanged.connect(self._on_value_changed)
            self.atari_color_spins.append(spin)
            registers.addWidget(QLabel(f"COLOR{index}", group), index // 3, (index % 3) * 2)
            registers.addWidget(spin, index // 3, (index % 3) * 2 + 1)
        layout.addLayout(registers)
        return group

    # --- state ----------------------------------------------------------

    def _populate(self, cfg: ConversionConfig) -> None:
        """Fill every widget from ``cfg`` with all signals blocked."""
        with self._signals_blocked():
            self._select_target(match_target(cfg.width, cfg.height))
            self.width_spin.setValue(cfg.width)
            self.height_spin.setValue(cfg.height)
            self._select_data(self.resize_combo, cfg.resize)
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
        self._refresh_enabled_states()
        self._update_background_swatch()

    def _refresh_enabled_states(self) -> None:
        """Enable exactly what the current selection makes meaningful."""
        custom = self.selected_target.is_custom
        self.width_spin.setEnabled(custom)
        self.height_spin.setEnabled(custom)
        self.sharpen_strength_spin.setEnabled(self.sharpen_check.isChecked())
        self.quantizer_combo.setEnabled(
            self._selected(self.dither_combo, DitherMethod) is DitherMethod.NONE
        )
        use_converter_palette = self.atari_palette_check.isChecked()
        for spin in self.atari_color_spins:
            spin.setEnabled(not use_converter_palette)

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

    # --- slots ----------------------------------------------------------

    def _on_target_changed(self, *_: object) -> None:
        target = self.selected_target
        if not target.is_custom:
            updated = apply_target(self.config(), target)
            with self._signals_blocked():
                self.width_spin.setValue(updated.width)
                self.height_spin.setValue(updated.height)
                self.colors_spin.setValue(updated.colors)
        self._refresh_enabled_states()
        self._emit()

    def _on_size_changed(self, *_: object) -> None:
        with self._signals_blocked():
            self._select_target(CUSTOM)
        self._refresh_enabled_states()
        self._emit()

    def _on_sharpen_toggled(self, *_: object) -> None:
        self._refresh_enabled_states()
        self._emit()

    def _on_dither_changed(self, *_: object) -> None:
        self._refresh_enabled_states()
        self._emit()

    def _on_atari_mode_toggled(self, *_: object) -> None:
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
