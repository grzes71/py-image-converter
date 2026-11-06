"""Tests for the read-only metrics panel (T2.5).

Assertions check the text the panel really shows and the text it puts on the Qt clipboard, so a wrong
format, a dropped field or a missing timing fails here instead of in a screenshot.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6", reason="PySide6 is not installed (install the 'gui' extra)")

from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel  # noqa: E402

from converter.metrics import format_metrics  # noqa: E402
from converter.types import ConversionResult, MetricResult  # noqa: E402
from converter_gui.widgets.metrics_panel import (  # noqa: E402
    METRICS_TITLE,
    OUTDATED_TEMPLATE,
    PLACEHOLDER_EMPTY,
    PLACEHOLDER_NO_METRICS,
    SUM_LABEL,
    TIMINGS_TITLE,
    MetricsPanel,
)

FULL_TIMINGS = {
    "prepare": 0.0123,
    "palette": 0.0,
    "edges": 0.004,
    "render": 0.25,
    "metrics": 0.0009,
}


def _metrics(**overrides: float) -> MetricResult:
    values = {
        "mse": 12.345,
        "psnr": 18.25,
        "ssim": 0.91234,
        "mean_rgb_error": 5.06,
        "mean_luminance_error": 4.0,
        "colors_used": 4,
        "histogram_coverage": 0.6288,
    }
    values.update(overrides)
    return MetricResult(**values)


def _result(
    metrics: MetricResult | None = None,
    timings: dict[str, float] | None = None,
) -> ConversionResult:
    return ConversionResult(
        indexed=np.zeros((2, 2), dtype=np.uint8),
        palette_indices=[0, 1],
        palette_rgb=np.zeros((2, 3), dtype=np.uint8),
        source_resized=np.zeros((2, 2, 3), dtype=np.uint8),
        metrics=metrics,
        timings=dict(timings) if timings else {},
    )


def _panel(qtbot) -> MetricsPanel:
    panel = MetricsPanel()
    qtbot.addWidget(panel)
    return panel


def _visible_texts(panel: MetricsPanel) -> list[str]:
    return [label.text() for label in panel.findChildren(QLabel) if label.text()]


def _metric_tooltip(panel: MetricsPanel, label: str) -> str:
    """Tooltip of the value label of a metric row (rows follow ``metric_values`` order)."""
    row = list(panel.metric_values).index(label)
    item = panel.metrics_form.itemAt(row, QFormLayout.ItemRole.FieldRole)
    widget = item.widget() if item is not None else None
    return widget.toolTip() if widget is not None else ""


# --- structure and initial state ----------------------------------------


def test_construction(qtbot) -> None:
    panel = _panel(qtbot)

    titles = {group.title() for group in panel.findChildren(QGroupBox)}
    assert titles == {METRICS_TITLE, TIMINGS_TITLE}
    assert panel.copy_button.text() == "Copy"
    assert panel.timings_hint.text().startswith("0.00 ms means the stage did no work")


def test_the_initial_state_is_empty(qtbot) -> None:
    panel = _panel(qtbot)

    assert panel.metric_values == {}
    assert panel.timing_values == {}
    assert panel.report_text() == ""
    assert panel.copy_button.isEnabled() is False
    assert PLACEHOLDER_EMPTY in _visible_texts(panel)
    assert panel.metrics_form.rowCount() == 1


def test_a_result_without_metrics_and_timings_offers_nothing_to_copy(qtbot) -> None:
    panel = _panel(qtbot)

    panel.set_result(_result())

    assert panel.report_text() == ""
    assert panel.copy_button.isEnabled() is False
    assert PLACEHOLDER_NO_METRICS in _visible_texts(panel)


# --- metrics ------------------------------------------------------------


def test_all_seven_metrics_are_shown(qtbot) -> None:
    panel = _panel(qtbot)

    panel.set_result(_result(metrics=_metrics()))

    assert panel.metric_values == {
        "MSE": "12.35",
        "PSNR": "18.25 dB",
        "SSIM": "0.9123",
        "Mean RGB error": "5.06",
        "Mean luminance error": "4.00",
        "Colors used": "4",
        "Histogram coverage": "62.88%",
    }
    assert panel.metrics_form.rowCount() == 7
    for text in ("12.35", "18.25 dB", "0.9123", "62.88%"):
        assert text in _visible_texts(panel)


def test_the_metrics_match_format_metrics_exactly(qtbot) -> None:
    panel = _panel(qtbot)

    panel.set_result(_result(metrics=_metrics()))

    displayed = [f"{label}: {value}" for label, value in panel.metric_values.items()]
    assert displayed == format_metrics(_metrics()).splitlines()


def test_no_metrics_is_reported_clearly(qtbot) -> None:
    panel = _panel(qtbot)

    panel.set_result(_result(metrics=None))

    assert panel.metric_values == {}
    assert PLACEHOLDER_NO_METRICS in _visible_texts(panel)
    assert panel.metrics_form.rowCount() == 1


# --- timings ------------------------------------------------------------


def test_all_five_stages_are_shown(qtbot) -> None:
    panel = _panel(qtbot)

    panel.set_result(_result(metrics=_metrics(), timings=FULL_TIMINGS))

    assert list(panel.timing_values) == ["prepare", "palette", "edges", "render", "metrics"]
    assert panel.timing_values["prepare"] == "12.30 ms"
    assert panel.timing_values["edges"] == "4.00 ms"
    assert panel.timing_values["render"] == "250.00 ms"
    assert panel.timing_values["metrics"] == "0.90 ms"
    assert panel.timings_form.rowCount() == len(FULL_TIMINGS) + 1, "five stages plus the sum"


def test_seconds_are_converted_to_milliseconds(qtbot) -> None:
    panel = _panel(qtbot)

    panel.set_result(_result(metrics=_metrics(), timings={"render": 1.5}))

    assert panel.timing_values["render"] == "1500.00 ms"
    assert "Render: 1500.00 ms" in panel.report_text()


def test_a_zero_timing_means_no_work(qtbot) -> None:
    panel = _panel(qtbot)

    panel.set_result(_result(metrics=_metrics(), timings={"palette": 0.0}))

    assert panel.timing_values["palette"] == "0.00 ms (no work: cache hit or skipped)"
    assert "no work" in panel.report_text()


def test_a_timing_below_the_display_resolution_is_not_shown_as_zero(qtbot) -> None:
    panel = _panel(qtbot)

    panel.set_result(_result(metrics=_metrics(), timings={"edges": 0.000001}))

    assert panel.timing_values["edges"] == "< 0.01 ms"


def test_missing_timing_keys_are_not_invented(qtbot) -> None:
    panel = _panel(qtbot)

    panel.set_result(_result(metrics=_metrics(), timings={"render": 0.25}))

    assert panel.timing_values == {"render": "250.00 ms"}
    assert panel.timings_form.rowCount() == 2, "one stage plus the sum"
    assert SUM_LABEL in _visible_texts(panel)


def test_an_unknown_stage_is_still_displayed(qtbot) -> None:
    panel = _panel(qtbot)

    panel.set_result(_result(metrics=_metrics(), timings={"render": 0.1, "export": 0.02}))

    assert list(panel.timing_values) == ["render", "export"]
    assert panel.timing_values["export"] == "20.00 ms"
    assert "Export: 20.00 ms" in panel.report_text()


def test_timings_are_shown_even_without_metrics(qtbot) -> None:
    panel = _panel(qtbot)

    panel.set_result(_result(metrics=None, timings={"render": 0.25}))

    assert panel.metric_values == {}
    assert panel.timing_values == {"render": "250.00 ms"}
    report = panel.report_text()
    assert PLACEHOLDER_NO_METRICS in report
    assert "Render: 250.00 ms" in report
    assert panel.copy_button.isEnabled() is True


def test_the_sum_is_labelled_as_the_sum_of_stages(qtbot) -> None:
    panel = _panel(qtbot)

    panel.set_result(_result(metrics=_metrics(), timings=FULL_TIMINGS))

    report = panel.report_text()
    assert f"{SUM_LABEL} (not the total conversion time): 267.20 ms" in report
    assert "not the total conversion time" in panel.timings_hint.text()


# --- clearing, copying and edge values ----------------------------------


def test_set_result_none_clears_the_panel(qtbot) -> None:
    panel = _panel(qtbot)
    panel.set_result(_result(metrics=_metrics(), timings=FULL_TIMINGS))

    panel.set_result(None)

    assert panel.metric_values == {}
    assert panel.timing_values == {}
    assert panel.report_text() == ""
    assert panel.copy_button.isEnabled() is False
    assert PLACEHOLDER_EMPTY in _visible_texts(panel)


def test_clear_is_idempotent(qtbot) -> None:
    panel = _panel(qtbot)
    panel.set_result(_result(metrics=_metrics(), timings=FULL_TIMINGS))

    panel.clear()
    panel.clear()

    assert panel.metric_values == {}
    assert panel.metrics_form.rowCount() == 1
    assert panel.timings_form.rowCount() == 1


def test_a_new_result_replaces_the_previous_rows(qtbot) -> None:
    panel = _panel(qtbot)
    panel.set_result(_result(metrics=_metrics(), timings=FULL_TIMINGS))

    panel.set_result(_result(metrics=_metrics(mse=1.0), timings={"render": 0.5}))

    assert panel.metric_values["MSE"] == "1.00"
    assert list(panel.timing_values) == ["render"]
    assert panel.metrics_form.rowCount() == 7, "rows are rebuilt, not appended"
    assert panel.timings_form.rowCount() == 2


def test_copy_puts_the_report_on_the_clipboard(qtbot) -> None:
    panel = _panel(qtbot)
    panel.set_result(_result(metrics=_metrics(), timings=FULL_TIMINGS))

    panel.copy_metrics()

    assert QGuiApplication.clipboard().text() == panel.report_text()


def test_the_report_contains_metrics_and_timings(qtbot) -> None:
    panel = _panel(qtbot)
    panel.set_result(_result(metrics=_metrics(), timings=FULL_TIMINGS))

    report = panel.report_text()

    assert report.startswith(format_metrics(_metrics()))
    assert "Stage timings" in report
    assert "Prepare: 12.30 ms" in report
    assert "Render: 250.00 ms" in report
    assert "\n" in report, "plain text, readable in any editor"


def test_copy_without_a_result_does_nothing(qtbot) -> None:
    panel = _panel(qtbot)
    QGuiApplication.clipboard().setText("untouched")

    panel.copy_metrics()

    assert QGuiApplication.clipboard().text() == "untouched"


def test_infinite_and_not_a_number_metrics_are_shown_and_explained(qtbot) -> None:
    panel = _panel(qtbot)

    panel.set_result(_result(metrics=_metrics(psnr=float("inf"), ssim=float("nan"))))

    assert panel.metric_values["PSNR"] == "inf dB"
    assert panel.metric_values["SSIM"] == "nan"
    assert "Infinite" in _metric_tooltip(panel, "PSNR")
    assert "Not a number" in _metric_tooltip(panel, "SSIM")
    assert "inf dB" in panel.report_text()


def test_non_finite_and_negative_timings_keep_their_meaning(qtbot) -> None:
    panel = _panel(qtbot)

    panel.set_result(_result(metrics=_metrics(), timings={"render": float("inf"), "edges": -0.001}))

    assert panel.timing_values["render"] == "inf ms"
    assert panel.timing_values["edges"] == "-1.00 ms"


def test_a_result_with_only_zero_timings_is_still_a_report(qtbot) -> None:
    panel = _panel(qtbot)

    panel.set_result(_result(metrics=None, timings=dict.fromkeys(FULL_TIMINGS, 0.0)))

    assert set(panel.timing_values) == set(FULL_TIMINGS)
    assert "no work" in panel.report_text()
    assert panel.copy_button.isEnabled() is True


def test_the_result_is_never_modified(qtbot) -> None:
    panel = _panel(qtbot)
    metrics = _metrics()
    timings = dict(FULL_TIMINGS)
    result = _result(metrics=metrics, timings=timings)
    before_metrics = dict(vars(metrics))
    before_timings = dict(timings)

    panel.set_result(result)
    panel.copy_metrics()
    panel.report_text()
    panel.clear()

    assert vars(metrics) == before_metrics
    assert timings == before_timings
    assert result.metrics is metrics
    assert result.timings == before_timings


# --- outdated marker (T3.3) ---------------------------------------------


def test_the_outdated_mark_is_hidden_by_default(qtbot) -> None:
    panel = _panel(qtbot)

    assert panel.outdated_generation is None
    assert panel.outdated_label.isHidden() is True
    assert panel.outdated_label.objectName() == "outdatedLabel"


def test_set_outdated_marks_the_shown_numbers(qtbot) -> None:
    """Old metrics stay readable, but they are labelled with the generation they belong to."""
    panel = _panel(qtbot)
    panel.set_result(_result(metrics=_metrics(), timings=FULL_TIMINGS))
    report_before = panel.report_text()

    panel.set_outdated(3)

    assert panel.outdated_generation == 3
    assert panel.outdated_label.isHidden() is False
    assert panel.outdated_label.text() == OUTDATED_TEMPLATE.format(generation=3)
    assert "3" in panel.outdated_label.toolTip()
    assert panel.metrics_form.rowCount() == 7, "the numbers are still shown"
    assert panel.report_text() == report_before, "the mark is not copied into the report"


def test_set_outdated_none_clears_the_mark(qtbot) -> None:
    panel = _panel(qtbot)
    panel.set_result(_result(metrics=_metrics(), timings=FULL_TIMINGS))
    panel.set_outdated(7)

    panel.set_outdated(None)

    assert panel.outdated_generation is None
    assert panel.outdated_label.isHidden() is True


def test_a_new_result_clears_the_outdated_mark(qtbot) -> None:
    panel = _panel(qtbot)
    panel.set_result(_result(metrics=_metrics(), timings=FULL_TIMINGS))
    panel.set_outdated(2)

    panel.set_result(_result(metrics=_metrics(mse=1.0), timings=FULL_TIMINGS))

    assert panel.outdated_generation == 2, "the panel does not decide when a result is current"
    panel.clear()
    assert panel.outdated_generation is None, "clearing drops the numbers and their mark"


def test_clear_removes_the_outdated_mark(qtbot) -> None:
    panel = _panel(qtbot)
    panel.set_result(_result(metrics=_metrics(), timings=FULL_TIMINGS))
    panel.set_outdated(5)

    panel.clear()

    assert panel.outdated_label.isHidden() is True
    assert panel.outdated_generation is None
    assert panel.metric_values == {}
