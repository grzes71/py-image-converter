"""GUI tests for the PAL pixel aspect ratio correction.

Two things are checked here that the core suite cannot see: the panel really offers the three modes
and reports them in the configuration, and the result preview is drawn with the pixel shape of the
selected mode while the source view, the zoom level and the panning stay as they were.
"""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("PySide6", reason="PySide6 is not installed (install the 'gui' extra)")

from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from converter.aspect import preview_pixel_aspect  # noqa: E402
from converter.types import AspectMode, ConversionConfig  # noqa: E402
from converter_gui.controllers.conversion_controller import (  # noqa: E402
    DisplayedConversion,
    image_key,
)
from converter_gui.targets import get_target  # noqa: E402
from converter_gui.widgets.image_preview import ZOOM_LEVELS, ImagePreview  # noqa: E402
from converter_gui.widgets.parameter_panel import (  # noqa: E402
    ASPECT_CHOICES,
    ParameterPanel,
)

from gui_fakes import build_window, make_result  # noqa: E402


def _combo_labels(panel: ParameterPanel) -> list[str]:
    combo = panel.aspect_combo
    return [combo.itemText(index) for index in range(combo.count())]


def _select(panel: ParameterPanel, mode: AspectMode) -> None:
    panel.aspect_combo.setCurrentIndex(panel.aspect_combo.findData(mode.value))
    QApplication.processEvents()


def _qimage(width: int, height: int, colour: tuple[int, int, int] = (10, 20, 30)) -> QImage:
    image = QImage(width, height, QImage.Format.Format_RGB888)
    image.fill(QColor(*colour))
    return image


def _preview(qtbot, width: int = 400, height: int = 300) -> ImagePreview:
    preview = ImagePreview()
    qtbot.addWidget(preview)
    preview.resize(width, height)
    preview.show()
    preview.layout().activate()
    QApplication.processEvents()
    return preview


# --- the panel ----------------------------------------------------------


def test_the_panel_offers_the_three_documented_modes(qtbot) -> None:
    panel = ParameterPanel()
    qtbot.addWidget(panel)

    assert _combo_labels(panel) == [label for _mode, label, _hint in ASPECT_CHOICES]
    assert _combo_labels(panel) == [
        "Disabled",
        "Preserve Aspect Ratio (Fit)",
        "Preserve Aspect Ratio (Crop)",
    ]
    assert panel.aspect_combo.currentData() == AspectMode.DISABLED.value, "off by default"


def test_every_mode_maps_back_to_the_configuration(qtbot) -> None:
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    seen: list[ConversionConfig] = []
    panel.config_changed.connect(seen.append)

    for mode in (AspectMode.FIT, AspectMode.CROP, AspectMode.DISABLED):
        _select(panel, mode)

    assert [config.aspect for config in seen] == [
        AspectMode.FIT,
        AspectMode.CROP,
        AspectMode.DISABLED,
    ]
    assert panel.config().aspect is AspectMode.DISABLED


def test_the_default_configuration_is_unchanged(qtbot) -> None:
    panel = ParameterPanel()
    qtbot.addWidget(panel)

    config = panel.config()

    assert config.aspect is AspectMode.DISABLED
    assert config.pixel_aspect is None


@pytest.mark.parametrize(
    ("key", "factor"),
    [("antic_e_192", 2.08), ("antic_e_190", 2.08), ("antic_d_96", 1.04), ("antic_f_192", 1.04)],
)
def test_a_preset_shows_its_pal_factor(key: str, factor: float, qtbot) -> None:
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    panel.set_config(dataclasses.replace(panel.config(), aspect=AspectMode.FIT))

    panel.target_combo.setCurrentIndex(panel.target_combo.findData(key))
    QApplication.processEvents()

    assert panel.pixel_aspect_spin.value() == pytest.approx(factor)
    assert panel.pixel_aspect_spin.isEnabled() is False, "a preset owns its factor"
    assert panel.config().pixel_aspect is None, "the value is derived, not stored"
    assert f"{factor:g}:1" in panel.aspect_hint.text()
    assert get_target(key).pixel_aspect == pytest.approx(factor)


def test_switching_the_target_updates_its_factor(qtbot) -> None:
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    panel.set_config(dataclasses.replace(panel.config(), aspect=AspectMode.FIT))

    panel.target_combo.setCurrentIndex(panel.target_combo.findData("antic_d_96"))
    QApplication.processEvents()
    assert panel.pixel_aspect_spin.value() == pytest.approx(1.04)

    panel.target_combo.setCurrentIndex(panel.target_combo.findData("antic_e_192"))
    QApplication.processEvents()
    assert panel.pixel_aspect_spin.value() == pytest.approx(2.08)

    panel.target_combo.setCurrentIndex(panel.target_combo.findData("antic_f_192"))
    QApplication.processEvents()
    assert panel.pixel_aspect_spin.value() == pytest.approx(1.04)


def test_a_custom_size_can_set_an_explicit_factor(qtbot) -> None:
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    panel.set_config(ConversionConfig(width=200, height=100, aspect=AspectMode.FIT))
    QApplication.processEvents()

    assert panel.pixel_aspect_spin.isEnabled() is True
    assert panel.pixel_aspect_spin.value() == 0.0, "no factor is invented for a Custom size"
    assert panel.config().pixel_aspect is None
    assert "No PAL pixel aspect is known" in panel.aspect_hint.text()

    panel.pixel_aspect_spin.setValue(1.5)
    QApplication.processEvents()

    assert panel.config().pixel_aspect == pytest.approx(1.5)
    assert "1.5:1" in panel.aspect_hint.text()


def test_the_explicit_factor_is_dropped_again_at_zero(qtbot) -> None:
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    panel.set_config(
        ConversionConfig(width=200, height=100, aspect=AspectMode.FIT, pixel_aspect=1.5)
    )

    panel.pixel_aspect_spin.setValue(0.0)
    QApplication.processEvents()

    assert panel.config().pixel_aspect is None


def test_a_preset_never_keeps_an_explicit_factor(qtbot) -> None:
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    panel.set_config(
        ConversionConfig(width=200, height=100, aspect=AspectMode.FIT, pixel_aspect=1.5)
    )

    panel.target_combo.setCurrentIndex(panel.target_combo.findData("antic_e_192"))
    QApplication.processEvents()

    config = panel.config()
    assert (config.width, config.height) == (160, 192)
    assert config.pixel_aspect is None, "the preset factor comes from the table"


def test_the_hint_describes_the_selected_mode(qtbot) -> None:
    panel = ParameterPanel()
    qtbot.addWidget(panel)

    assert "stretched onto the whole target grid" in panel.aspect_hint.text()

    _select(panel, AspectMode.FIT)
    assert "2.08:1" in panel.aspect_hint.text()
    assert "ANTIC E" in panel.aspect_hint.text()

    _select(panel, AspectMode.DISABLED)
    assert "stretched onto the whole target grid" in panel.aspect_hint.text()


def test_set_config_reports_the_mode_it_was_given(qtbot) -> None:
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    seen: list[ConversionConfig] = []
    panel.config_changed.connect(seen.append)

    panel.set_config(ConversionConfig(aspect=AspectMode.CROP))
    QApplication.processEvents()

    assert panel.config().aspect is AspectMode.CROP
    assert seen == [], "programmatic updates stay silent"


def test_the_image_key_covers_the_correction() -> None:
    """A changed mode must schedule a conversion instead of being treated as the same image."""
    base = ConversionConfig()

    assert image_key(base) != image_key(dataclasses.replace(base, aspect=AspectMode.FIT))
    assert image_key(base) != image_key(dataclasses.replace(base, pixel_aspect=1.5))
    assert image_key(dataclasses.replace(base, aspect=AspectMode.FIT)) != image_key(
        dataclasses.replace(base, aspect=AspectMode.CROP)
    )


# --- the preview ----------------------------------------------------------


def test_the_result_view_is_drawn_with_the_pixel_aspect(qtbot) -> None:
    preview = _preview(qtbot)

    preview.set_image_pair(_qimage(160, 192), _qimage(160, 192), secondary_aspect=2.08)

    primary, secondary = preview.views
    assert primary.transform().m11() == pytest.approx(1.0)
    assert primary.transform().m22() == pytest.approx(1.0)
    assert secondary.transform().m11() == pytest.approx(2.08), "wider pixels, wider on screen"
    assert secondary.transform().m22() == pytest.approx(1.0), "the height is not touched"


@pytest.mark.parametrize("level", ZOOM_LEVELS)
def test_the_pixel_aspect_scales_with_the_zoom(level: float, qtbot) -> None:
    preview = _preview(qtbot)
    preview.set_image_pair(_qimage(160, 192), _qimage(160, 192), secondary_aspect=1.04)

    preview.set_zoom(level)

    primary, secondary = preview.views
    assert preview.zoom == pytest.approx(level), "the zoom level is unchanged"
    assert primary.transform().m11() == pytest.approx(level)
    assert secondary.transform().m11() == pytest.approx(level * 1.04)
    assert secondary.transform().m22() == pytest.approx(level)


def test_the_aspect_does_not_reset_the_zoom(qtbot) -> None:
    preview = _preview(qtbot)
    preview.set_image_pair(_qimage(160, 192), _qimage(160, 192))
    preview.set_zoom(4.0)

    preview.set_image_pair(_qimage(160, 192), _qimage(160, 192), secondary_aspect=2.08)

    assert preview.zoom == pytest.approx(4.0)
    assert preview.is_fit is False
    assert preview.views[1].transform().m11() == pytest.approx(8.32)


def test_disabled_keeps_the_square_pixel_preview(qtbot) -> None:
    preview = _preview(qtbot)

    preview.set_image_pair(_qimage(160, 192), _qimage(160, 192), secondary_aspect=1.0)

    for view in preview.views:
        assert view.transform().m11() == pytest.approx(view.transform().m22())


def test_fit_uses_the_displayed_width(qtbot) -> None:
    preview = _preview(qtbot, 200, 400)
    preview.set_image_pair(_qimage(160, 192), _qimage(160, 192))

    preview.fit()
    plain = preview.zoom

    preview.set_image_pair(_qimage(160, 192), _qimage(160, 192), secondary_aspect=2.08)

    stretched = preview.zoom
    viewport = preview.views[0].viewport().size()
    assert stretched < plain, "the wider result needs a smaller factor to fit"
    assert 160 * 2.08 * stretched <= viewport.width() + 1
    assert 192 * stretched <= viewport.height() + 1


def test_panning_keeps_both_views_on_the_same_content(qtbot) -> None:
    preview = _preview(qtbot, 200, 200)
    preview.set_image_pair(_qimage(160, 192), _qimage(160, 192), secondary_aspect=2.08)
    preview.set_zoom(8.0)
    primary, secondary = preview.views
    # Scrollbars are integral, so the alignment is exact up to one device pixel.
    tolerance = 1.0 / 8.0

    primary.horizontalScrollBar().setValue(150)
    primary.verticalScrollBar().setValue(90)
    QApplication.processEvents()

    assert abs(secondary.mapToScene(0, 0).x() - primary.mapToScene(0, 0).x()) <= tolerance
    assert abs(secondary.mapToScene(0, 0).y() - primary.mapToScene(0, 0).y()) <= tolerance

    secondary.horizontalScrollBar().setValue(40)
    QApplication.processEvents()

    assert abs(primary.mapToScene(0, 0).x() - secondary.mapToScene(0, 0).x()) <= tolerance


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan"), float("inf")])
def test_an_invalid_pixel_aspect_is_rejected(invalid: float, qtbot) -> None:
    preview = _preview(qtbot)

    with pytest.raises(ValueError):
        preview.set_image_pair(_qimage(10, 10), _qimage(10, 10), secondary_aspect=invalid)


# --- the window ----------------------------------------------------------


def _displayed(config: ConversionConfig) -> DisplayedConversion:
    return DisplayedConversion(
        generation=1,
        config=config,
        result=make_result(),
        timings={},
        produced_at=0.0,
    )


def test_the_window_shows_the_result_with_the_mode_aspect(qtbot) -> None:
    window, _backend, _boxes = build_window(qtbot)
    config = ConversionConfig(width=160, height=192, aspect=AspectMode.FIT)

    window._controller.result_ready.emit(_displayed(config))
    QApplication.processEvents()

    primary, secondary = window.preview.views
    assert secondary.transform().m11() == pytest.approx(2.08 * primary.transform().m11())
    assert preview_pixel_aspect(config) == pytest.approx(2.08)


def test_the_window_keeps_square_pixels_while_disabled(qtbot) -> None:
    window, _backend, _boxes = build_window(qtbot)

    window._controller.result_ready.emit(_displayed(ConversionConfig()))
    QApplication.processEvents()

    primary, secondary = window.preview.views
    assert secondary.transform().m11() == pytest.approx(primary.transform().m11())


def test_the_window_follows_the_displayed_configuration_not_the_requested_one(qtbot) -> None:
    """The pixels on screen belong to the finished conversion, so its mode decides the shape."""
    window, _backend, _boxes = build_window(qtbot)
    window.preview.set_image_pair(_qimage(160, 192), _qimage(160, 192))
    window.preview.set_zoom(2.0)

    done = ConversionConfig(width=320, height=192, aspect=AspectMode.CROP)
    window._controller.result_ready.emit(_displayed(done))
    QApplication.processEvents()

    secondary = window.preview.views[1]
    assert secondary.transform().m11() == pytest.approx(2.0 * 1.04)
    assert window.preview.zoom == pytest.approx(2.0), "the zoom the user chose survives"
