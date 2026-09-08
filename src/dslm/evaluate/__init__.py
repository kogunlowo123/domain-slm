"""Scoring models, comparing them honestly, and gating on the result."""

from __future__ import annotations

from dslm.evaluate.baseline import Baseline, Verdict, compare
from dslm.evaluate.metrics import Comparison, Metrics, evaluate, mcnemar
from dslm.evaluate.report import Evaluation, render_json, render_junit, render_markdown

__all__ = [
    "Baseline",
    "Comparison",
    "Evaluation",
    "Metrics",
    "Verdict",
    "compare",
    "evaluate",
    "mcnemar",
    "render_json",
    "render_junit",
    "render_markdown",
]
