"""Tests for the conversion parameter panel (T2.4).

The assertions go through the widgets the user would touch and then read ``config()`` back, so they
describe behaviour (a control emits, a limit changes, a value survives being disabled) rather than the
mere existence of widgets.
"""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("PySide6", reason="PySide6 is not installed (install the 'gui' extra)")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QComboBox, QGroupBox  # noqa: E402

from converter.types import (  # noqa: E402
    MAX_ATARI_COLORS,
    ConversionConfig,
    DitherMethod,
    ExportFormat,
    PaletteMethod,
    QuantizerMethod,
    ResizeMethod,
)
from converter_gui.limits import MAX_PNG_COLORS  # noqa: E402
from converter_gui.widgets.parameter_panel import (  # noqa: E402
    ATARI_GROUP_TITLE,
    ATARI_REGISTER_COUNT,
    DITHER_SECTIONS,
    MAX_HEIGHT,
    MIN_WIDTH,
    ParameterPanel,
)


def _panel(qtbot) -> ParameterPanel:
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    return panel


def _collect(panel: ParameterPanel) -> list[ConversionConfig]:
    received: list[ConversionConfig] = []
    panel.config_changed.connect(received.append)
    return received


def _select(combo: QComboBox, value: str) -> None:
    index = combo.findData(value)
    assert index >= 0, f"{value!r} is not in the combo"
    combo.setCurrentIndex(index)


def test_construction(qtbot) -> None:
    panel = _panel(qtbot)

    cfg = panel.config()

    assert isinstance(cfg, ConversionConfig)
    assert (cfg.width, cfg.height) == (160, 192)
    assert cfg.colors == 4
    assert cfg.dither is DitherMethod.FLOYD
    assert cfg.quantizer is QuantizerMethod.NEAREST
    assert panel.selected_target.key == "antic_e_192"
    assert panel.width_spin.isEnabled() is False, "a fixed target owns the size"
    assert panel.quantizer_combo.isEnabled() is False, "floyd does not use a quantizer"


def test_the_documented_groups_exist(qtbot) -> None:
    panel = _panel(qtbot)

    titles = {group.title() for group in panel.findChildren(QGroupBox)}

    assert {"Output", "Resize", "Sharpen", "Palette", "Dithering", "Quantization"} <= titles
    assert ATARI_GROUP_TITLE in titles


def test_round_trip_of_the_whole_configuration(qtbot) -> None:
    panel = _panel(qtbot)
    original = ConversionConfig(
        width=160,
        height=190,
        palette=PaletteMethod.OCTREE,
        colors=3,
        dither=DitherMethod.STUCKI,
        quantizer=QuantizerMethod.WEIGHTED,
        resize=ResizeMethod.BICUBIC,
        sharpen=True,
        sharpen_strength=2.5,
        background_index=42,
        metrics=True,
        preview=True,
        export_formats=[ExportFormat.PNG, ExportFormat.XEX],
        atari_colors=[16, 32, 48, 64, 80],
        label_name="SCREEN",
    )

    panel.set_config(original)

    assert panel.config() == original
    assert panel.selected_target.key == "antic_e_190"


def test_set_config_emits_nothing(qtbot) -> None:
    panel = _panel(qtbot)
    received = _collect(panel)

    panel.set_config(ConversionConfig(width=100, height=50, colors=2, dither=DitherMethod.NONE))

    assert received == []


def test_a_user_change_emits_the_new_config(qtbot) -> None:
    panel = _panel(qtbot)
    received = _collect(panel)

    panel.colors_spin.setValue(2)

    assert len(received) == 1
    assert received[0].colors == 2

    panel.sharpen_check.setChecked(True)

    assert len(received) == 2
    assert received[1].sharpen is True


@pytest.mark.parametrize(
    ("enum_type", "combo_name", "field"),
    [
        (ResizeMethod, "resize_combo", "resize"),
        (PaletteMethod, "palette_combo", "palette"),
        (DitherMethod, "dither_combo", "dither"),
        (QuantizerMethod, "quantizer_combo", "quantizer"),
    ],
)
def test_every_enum_member_is_offered_and_maps_back(
    qtbot, enum_type, combo_name: str, field: str
) -> None:
    panel = _panel(qtbot)
    combo = getattr(panel, combo_name)
    offered = {combo.itemData(i) for i in range(combo.count()) if combo.itemData(i) is not None}

    assert offered == {member.value for member in enum_type}
    for member in enum_type:
        _select(combo, member.value)
        assert getattr(panel.config(), field) is member


def test_dither_methods_are_grouped_with_hints(qtbot) -> None:
    panel = _panel(qtbot)
    combo = panel.dither_combo

    headers = [i for i in range(combo.count()) if combo.itemData(i) is None]
    assert len(headers) == len(DITHER_SECTIONS)
    for index in headers:
        item = combo.model().item(index)
        assert item is not None and item.isEnabled() is False, "a header cannot be selected"
    for index in range(combo.count()):
        if index in headers:
            continue
        assert "—" in combo.itemText(index), "every method carries a short hint"
        assert combo.itemData(index, Qt.ItemDataRole.ToolTipRole)


# --- targets and size ---------------------------------------------------


def test_selecting_a_target_writes_its_size_and_clamps_colours(qtbot) -> None:
    panel = _panel(qtbot)
    received = _collect(panel)
    panel.set_config(ConversionConfig(width=64, height=64, colors=4))

    _select(panel.target_combo, "antic_f_192")

    cfg = panel.config()
    assert (cfg.width, cfg.height) == (320, 192)
    assert cfg.colors == 2, "ANTIC F is a two-colour mode"
    assert panel.width_spin.isEnabled() is False
    assert len(received) == 1
    assert received[0].colors == 2


def test_custom_target_enables_the_size_widgets(qtbot) -> None:
    panel = _panel(qtbot)
    assert panel.width_spin.isEnabled() is False

    _select(panel.target_combo, "custom")

    assert panel.width_spin.isEnabled() is True
    assert panel.height_spin.isEnabled() is True


def test_editing_the_size_switches_to_custom(qtbot) -> None:
    panel = _panel(qtbot)
    assert panel.selected_target.key == "antic_e_192"

    panel.width_spin.setValue(200)

    assert panel.selected_target.key == "custom"
    cfg = panel.config()
    assert (cfg.width, cfg.height) == (200, 192)

    panel.height_spin.setValue(100)

    assert panel.selected_target.key == "custom"
    assert panel.config().height == 100


def test_set_config_picks_the_matching_target(qtbot) -> None:
    panel = _panel(qtbot)

    panel.set_config(ConversionConfig(width=320, height=192))

    assert panel.selected_target.key == "antic_f_192"
    panel.set_config(ConversionConfig(width=123, height=45))
    assert panel.selected_target.key == "custom"


# --- sharpen ------------------------------------------------------------


def test_sharpen_controls_the_strength_widget(qtbot) -> None:
    panel = _panel(qtbot)
    panel.set_config(ConversionConfig(sharpen=False, sharpen_strength=1.5))
    assert panel.sharpen_strength_spin.isEnabled() is False

    panel.sharpen_check.setChecked(True)
    assert panel.sharpen_strength_spin.isEnabled() is True

    panel.sharpen_strength_spin.setValue(2.5)
    assert panel.config().sharpen is True
    assert panel.config().sharpen_strength == pytest.approx(2.5)


def test_disabling_sharpen_keeps_the_strength_value(qtbot) -> None:
    panel = _panel(qtbot)
    panel.set_config(ConversionConfig(sharpen=True, sharpen_strength=3.0))

    panel.sharpen_check.setChecked(False)

    assert panel.sharpen_strength_spin.isEnabled() is False
    cfg = panel.config()
    assert cfg.sharpen is False
    assert cfg.sharpen_strength == pytest.approx(3.0), "the value is not destroyed"


# --- dithering and quantization -----------------------------------------


def test_quantizer_is_enabled_only_without_dithering(qtbot) -> None:
    panel = _panel(qtbot)
    panel.set_config(ConversionConfig(dither=DitherMethod.FLOYD))
    assert panel.quantizer_combo.isEnabled() is False

    _select(panel.dither_combo, DitherMethod.NONE.value)
    assert panel.quantizer_combo.isEnabled() is True

    _select(panel.dither_combo, DitherMethod.BAYER8.value)
    assert panel.quantizer_combo.isEnabled() is False


def test_the_quantizer_value_survives_being_disabled(qtbot) -> None:
    panel = _panel(qtbot)
    _select(panel.dither_combo, DitherMethod.NONE.value)
    _select(panel.quantizer_combo, QuantizerMethod.PERCEPTUAL.value)

    _select(panel.dither_combo, DitherMethod.BAYER4.value)

    assert panel.quantizer_combo.isEnabled() is False
    assert panel.config().quantizer is QuantizerMethod.PERCEPTUAL

    _select(panel.dither_combo, DitherMethod.NONE.value)

    assert panel.quantizer_combo.isEnabled() is True
    assert panel.config().quantizer is QuantizerMethod.PERCEPTUAL


# --- colour limits ------------------------------------------------------


def test_the_default_limit_is_the_hardware_four(qtbot) -> None:
    panel = _panel(qtbot)

    assert panel.png_only is False
    assert panel.colors_spin.minimum() == 2
    assert panel.colors_spin.maximum() == MAX_ATARI_COLORS
    assert panel.config().colors == 4


def test_png_only_raises_the_limit_to_sixteen(qtbot) -> None:
    panel = _panel(qtbot)
    received = _collect(panel)

    panel.set_png_only(True)

    assert panel.png_only is True
    assert panel.colors_spin.maximum() == MAX_PNG_COLORS

    panel.colors_spin.setValue(12)

    assert panel.config().colors == 12
    assert received[-1].colors == 12


def test_leaving_png_only_restores_the_limit_and_clamps(qtbot) -> None:
    panel = _panel(qtbot)
    panel.set_png_only(True)
    panel.colors_spin.setValue(16)
    received = _collect(panel)

    panel.set_png_only(False)

    assert panel.colors_spin.maximum() == MAX_ATARI_COLORS
    assert panel.config().colors == MAX_ATARI_COLORS
    assert len(received) == 1, "the clamped value is reported once"
    assert received[0].colors == MAX_ATARI_COLORS


def test_a_png_only_configuration_derives_the_wider_limit(qtbot) -> None:
    panel = _panel(qtbot)

    panel.set_config(ConversionConfig(export_formats=[ExportFormat.PNG], colors=8))

    assert panel.png_only is True
    assert panel.colors_spin.maximum() == MAX_PNG_COLORS
    assert panel.config().colors == 8

    panel.set_config(
        ConversionConfig(export_formats=[ExportFormat.PNG, ExportFormat.BIN], colors=4)
    )

    assert panel.png_only is False
    assert panel.colors_spin.maximum() == MAX_ATARI_COLORS


# --- background ---------------------------------------------------------


def test_background_index_is_editable_and_reported(qtbot) -> None:
    panel = _panel(qtbot)

    panel.background_spin.setValue(200)

    assert panel.background_spin.maximum() == 255
    assert panel.config().background_index == 200
    assert panel.background_index_label.text() == "index 200"


def test_background_swatch_needs_a_palette(qtbot) -> None:
    panel = _panel(qtbot)

    assert panel.background_color is None
    assert "only known once a conversion" in panel.background_swatch.toolTip()

    panel.set_palette_colors([(10, 20, 30), (40, 50, 60)])
    panel.background_spin.setValue(1)
    colour = panel.background_color

    assert colour is not None
    assert (colour.red(), colour.green(), colour.blue()) == (40, 50, 60)
    assert colour.name() == "#28323c"
    assert panel.background_swatch.toolTip() == "palette[1] = #28323c"

    panel.background_spin.setValue(7)  # beyond the palette: unknown again

    assert panel.background_color is None


# --- Atari colour registers ---------------------------------------------


def test_atari_colours_are_none_while_the_converter_palette_is_used(qtbot) -> None:
    panel = _panel(qtbot)

    assert panel.atari_palette_check.isChecked() is True
    assert panel.config().atari_colors is None
    assert len(panel.atari_color_spins) == ATARI_REGISTER_COUNT
    assert all(spin.isEnabled() is False for spin in panel.atari_color_spins)
    assert all(spin.displayIntegerBase() == 16 for spin in panel.atari_color_spins)


def test_atari_colour_registers_round_trip(qtbot) -> None:
    panel = _panel(qtbot)
    panel.atari_palette_check.setChecked(False)

    for spin, value in zip(panel.atari_color_spins, (1, 2, 3, 4, 255)):
        spin.setValue(value)

    assert all(spin.isEnabled() is True for spin in panel.atari_color_spins)
    assert panel.config().atari_colors == [1, 2, 3, 4, 255]

    panel.atari_palette_check.setChecked(True)

    assert panel.config().atari_colors is None
    assert [spin.value() for spin in panel.atari_color_spins] == [1, 2, 3, 4, 255], "kept for reuse"


def test_a_short_atari_colour_list_is_normalised_to_five(qtbot) -> None:
    panel = _panel(qtbot)

    panel.set_config(ConversionConfig(atari_colors=[16, 32]))

    assert panel.atari_palette_check.isChecked() is False
    assert [spin.value() for spin in panel.atari_color_spins] == [16, 32, 32, 32, 32]
    assert panel.config().atari_colors == [16, 32, 32, 32, 32]


# --- signal blocking, mutation and validation ---------------------------


def test_programmatic_updates_are_silent(qtbot) -> None:
    panel = _panel(qtbot)
    received = _collect(panel)

    panel.set_config(
        ConversionConfig(
            width=200,
            height=100,
            colors=2,
            dither=DitherMethod.NONE,
            sharpen=True,
            atari_colors=[1, 2, 3, 4, 5],
        )
    )
    panel.set_png_only(True)
    panel.set_png_only(True)  # no value change: nothing to report
    panel.set_palette_colors([(1, 2, 3)])

    assert received == []


def test_the_source_configuration_is_never_mutated(qtbot) -> None:
    panel = _panel(qtbot)
    original = ConversionConfig(atari_colors=[1, 2, 3, 4, 5], export_formats=[ExportFormat.PNG])
    snapshot = dataclasses.replace(
        original,
        atari_colors=list(original.atari_colors),
        export_formats=list(original.export_formats),
    )

    panel.set_config(original)
    panel.colors_spin.setValue(8)
    panel.background_spin.setValue(9)
    panel.atari_color_spins[0].setValue(200)

    assert original == snapshot
    assert original.atari_colors == [1, 2, 3, 4, 5]
    assert panel.config().atari_colors == [200, 2, 3, 4, 5]
    assert panel.config().background_index == 9


@pytest.mark.parametrize(
    ("cfg", "needle"),
    [
        (ConversionConfig(colors=1), "colors"),
        (ConversionConfig(background_index=300), "background_index"),
        (ConversionConfig(width=0), "width"),
    ],
)
def test_an_invalid_configuration_is_rejected(qtbot, cfg: ConversionConfig, needle: str) -> None:
    panel = _panel(qtbot)

    with pytest.raises(ValueError, match=needle):
        panel.set_config(cfg)


def test_values_outside_the_widget_ranges_are_clamped(qtbot) -> None:
    panel = _panel(qtbot)

    panel.set_config(ConversionConfig(width=4, height=1000))

    cfg = panel.config()
    assert cfg.width == MIN_WIDTH
    assert cfg.height == MAX_HEIGHT


def test_the_panel_keeps_the_fields_it_does_not_edit(qtbot) -> None:
    panel = _panel(qtbot)
    panel.set_config(
        ConversionConfig(
            mode=panel.config().mode,
            metrics=True,
            preview=True,
            export_formats=[ExportFormat.XEX],
            label_name="SCREEN_X",
        )
    )

    panel.colors_spin.setValue(2)
    cfg = panel.config()

    assert cfg.metrics is True
    assert cfg.preview is True
    assert cfg.export_formats == [ExportFormat.XEX]
    assert cfg.label_name == "SCREEN_X"
