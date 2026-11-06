"""Read-only metrics panel (T2.5, plan §6.4).

The panel presents one :class:`~converter.types.ConversionResult`: the seven ``MetricResult`` fields and
the per-stage timings. It computes nothing and interprets nothing - it keeps the result it was given and
renders what is in it, so a zero timing keeps its "no work performed" meaning and an infinite PSNR stays
infinite.

Both parts live in a :class:`CollapsibleSection` that starts collapsed, so the diagnostics are available
on request instead of competing with the preview for space; the values are rendered either way, the
sections only decide whether they are on screen.

Values are formatted by :func:`converter.metrics.format_metrics` itself: the displayed rows are the
lines of that report, which keeps the panel and the CLI telling exactly the same story. The copy action
reuses the same text and appends the stage timings, which ``format_metrics`` does not know about.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from converter.metrics import format_metrics
from converter.types import ConversionResult, MetricResult
from converter_gui.widgets.collapsible_section import CollapsibleSection

METRICS_TITLE = "Metrics"
TIMINGS_TITLE = "Stage timings"

PLACEHOLDER_EMPTY = "No conversion result yet."
PLACEHOLDER_NO_METRICS = "Metrics were not computed for this result."
PLACEHOLDER_NO_TIMINGS = "No stage timings recorded for this result."

OUTDATED_TEMPLATE = "outdated (gen {generation})"
"""Shown when the numbers belong to a generation the controller has already superseded."""

#: The five stages of the engine contract, in display order. Unknown keys are appended after these, so a
#: future stage shows up instead of being dropped.
TIMING_KEYS: tuple[str, ...] = ("prepare", "palette", "edges", "render", "metrics")

SUM_LABEL = "Sum of stages"
SUM_TOOLTIP = (
    "Sum of the stage times above - not the total conversion time: loading the file and the GUI work "
    "are not part of ConversionResult.timings"
)
TIMINGS_HINT = (
    "0.00 ms means the stage did no work (cache hit or skipped). The sum of stages is not the total "
    "conversion time."
)


def _stage_label(key: str) -> str:
    """Display label of a timing key (``"prepare"`` -> ``"Prepare"``)."""
    return key.replace("_", " ").title()


def _unusual_note(value: str) -> str:
    """Explain a non-finite metric without inventing a quality verdict."""
    lowered = value.lower()
    if "nan" in lowered:
        return "Not a number: the metric is undefined for this input."
    if "inf" in lowered:
        return "Infinite: the metric reached its limit (PSNR is infinite when MSE is 0)."
    return ""


def format_stage_time(seconds: float) -> str:
    """Render a stage duration in milliseconds, keeping ``0.0`` meaningful.

    The engine contract says ``0.0`` means "no work performed" (a cache hit or a skipped stage), so it is
    spelled out instead of being shown as an absurdly fast stage. Values below the display resolution,
    negative ones and non-finite ones keep their exact meaning.
    """
    milliseconds = float(seconds) * 1000.0
    if milliseconds == 0.0:
        return "0.00 ms (no work: cache hit or skipped)"
    if not math.isfinite(milliseconds):
        return f"{milliseconds} ms"
    if 0.0 < milliseconds < 0.01:
        return "< 0.01 ms"
    return f"{milliseconds:.2f} ms"


class MetricsPanel(QWidget):
    """Read-only view of the metrics and stage timings of a conversion result."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: ConversionResult | None = None
        self._metric_values: dict[str, str] = {}
        self._timing_values: dict[str, str] = {}
        self._outdated_generation: int | None = None
        self._build_ui()
        self.clear()

    # --- public API -----------------------------------------------------

    def set_result(self, result: ConversionResult | None) -> None:
        """Show ``result``; ``None`` clears the panel back to its empty state."""
        if result is None:
            self.clear()
            return
        self._result = result
        self._rebuild_metrics(result.metrics)
        self._rebuild_timings(result.timings)
        self.copy_button.setEnabled(bool(self.report_text()))

    def clear(self) -> None:
        """Return to the empty state: no result, no values, nothing to copy."""
        self._result = None
        self.set_outdated(None)
        self._rebuild_metrics(None, placeholder=PLACEHOLDER_EMPTY)
        self._rebuild_timings(None)
        self.copy_button.setEnabled(False)

    def copy_metrics(self) -> None:
        """Put the text of :meth:`report_text` on the standard Qt clipboard."""
        text = self.report_text()
        if text:
            QGuiApplication.clipboard().setText(text)

    def report_text(self) -> str:
        """The report Copy produces: ``format_metrics`` plus the stage timings."""
        if self._result is None:
            return ""
        metrics = self._result.metrics
        timings = self._result.timings
        if metrics is None and not timings:
            return ""  # only placeholders: nothing worth copying
        sections: list[str] = []
        sections.append(format_metrics(metrics) if metrics is not None else PLACEHOLDER_NO_METRICS)
        timings_text = self._timings_section(timings)
        if timings_text:
            sections.append(timings_text)
        return "\n\n".join(sections)

    @property
    def metric_values(self) -> dict[str, str]:
        """The displayed metric rows (label -> formatted value)."""
        return dict(self._metric_values)

    @property
    def timing_values(self) -> dict[str, str]:
        """The displayed stage timings (stage -> formatted value)."""
        return dict(self._timing_values)

    @property
    def outdated_generation(self) -> int | None:
        """Generation the shown numbers belong to when they are outdated (``None`` if current)."""
        return self._outdated_generation

    def set_outdated(self, generation: int | None) -> None:
        """Mark the shown numbers as belonging to an outdated generation (or clear the mark)."""
        self._outdated_generation = generation
        if generation is None:
            self.outdated_label.hide()
            return
        self.outdated_label.setText(OUTDATED_TEMPLATE.format(generation=generation))
        self.outdated_label.setToolTip(
            f"These numbers come from generation {generation}, which no longer matches the settings"
        )
        self.outdated_label.show()

    # --- construction ---------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.outdated_label = QLabel(self)
        self.outdated_label.setObjectName("outdatedLabel")
        self.outdated_label.setStyleSheet(
            "background:#8a5a00; color:#ffffff; padding:2px 6px; border-radius:3px;"
        )
        self.outdated_label.hide()
        layout.addWidget(self.outdated_label)

        # Both sections start collapsed: the diagnostics are there on request and do not compete with the
        # preview for space. Their content is a body of their own, so hiding it frees the space it took.
        self.metrics_section = CollapsibleSection(METRICS_TITLE, self)
        self.metrics_form = QFormLayout()
        self.metrics_section.body_layout.addLayout(self.metrics_form)
        layout.addWidget(self.metrics_section)

        self.timings_section = CollapsibleSection(TIMINGS_TITLE, self)
        self.timings_form = QFormLayout()
        self.timings_section.body_layout.addLayout(self.timings_form)
        self.timings_hint = QLabel(TIMINGS_HINT, self.timings_section.body)
        self.timings_hint.setWordWrap(True)
        self.timings_section.body_layout.addWidget(self.timings_hint)
        layout.addWidget(self.timings_section)

        self.copy_button = QPushButton("Copy", self)
        self.copy_button.setToolTip("Copy the metrics and the stage timings to the clipboard")
        self.copy_button.clicked.connect(self.copy_metrics)
        layout.addWidget(self.copy_button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addStretch(1)

    # --- rendering ------------------------------------------------------

    def _rebuild_metrics(
        self, metrics: MetricResult | None, placeholder: str | None = None
    ) -> None:
        """Render the metric rows straight from ``format_metrics`` (no duplicated formatting)."""
        self._clear_form(self.metrics_form)
        self._metric_values = {}
        if metrics is None:
            self._add_placeholder(self.metrics_form, placeholder or PLACEHOLDER_NO_METRICS)
            return
        for line in format_metrics(metrics).splitlines():
            label, _, value = line.partition(": ")
            row = self._value_label(value)
            note = _unusual_note(value)
            if note:
                row.setToolTip(note)
            self.metrics_form.addRow(QLabel(label, self), row)
            self._metric_values[label] = value

    def _rebuild_timings(self, timings: dict[str, float] | None) -> None:
        """Render one row per stage plus the sum of the stages."""
        self._clear_form(self.timings_form)
        self._timing_values = {}
        if not timings:
            placeholder = PLACEHOLDER_NO_TIMINGS if timings is not None else PLACEHOLDER_EMPTY
            self._add_placeholder(self.timings_form, placeholder)
            return
        ordered = self._ordered_timings(timings)
        for stage, seconds in ordered:
            text = format_stage_time(seconds)
            self.timings_form.addRow(QLabel(_stage_label(stage), self), self._value_label(text))
            self._timing_values[stage] = text
        total = self._value_label(format_stage_time(sum(seconds for _, seconds in ordered)))
        total.setToolTip(SUM_TOOLTIP)
        self.timings_form.addRow(QLabel(SUM_LABEL, self), total)

    def _timings_section(self, timings: dict[str, float]) -> str:
        """The timings part of the copied report, in plain text."""
        if not timings:
            return f"{TIMINGS_TITLE}\n{PLACEHOLDER_NO_TIMINGS}"
        ordered = self._ordered_timings(timings)
        lines = [TIMINGS_TITLE]
        lines.extend(
            f"{_stage_label(stage)}: {format_stage_time(seconds)}" for stage, seconds in ordered
        )
        total = sum(seconds for _, seconds in ordered)
        lines.append(f"{SUM_LABEL} (not the total conversion time): {format_stage_time(total)}")
        return "\n".join(lines)

    @staticmethod
    def _ordered_timings(timings: dict[str, float]) -> list[tuple[str, float]]:
        """Known stages in contract order, then any unknown stage, each coerced to ``float``."""
        keys = [key for key in TIMING_KEYS if key in timings]
        keys.extend(sorted(key for key in timings if key not in TIMING_KEYS))
        ordered: list[tuple[str, float]] = []
        for key in keys:
            try:
                ordered.append((key, float(timings[key])))
            except (TypeError, ValueError):
                ordered.append((key, float("nan")))
        return ordered

    def _value_label(self, text: str) -> QLabel:
        label = QLabel(text, self)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    @staticmethod
    def _add_placeholder(form: QFormLayout, text: str) -> None:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setEnabled(False)
        form.addRow(label)

    @staticmethod
    def _clear_form(form: QFormLayout) -> None:
        while form.rowCount():
            form.removeRow(0)
