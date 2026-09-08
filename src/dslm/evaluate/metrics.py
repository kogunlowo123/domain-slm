"""Metrics, and the arithmetic that says whether a difference is real.

Accuracy on its own is the number everybody reports and the number that hides
the most. Three things are reported alongside it here, and each answers a
question accuracy cannot:

**Macro F1** — accuracy is dominated by the largest classes. A model that gives
up entirely on the two rarest chapters can still look fine. Macro F1 gives every
class the same weight, so giving up is visible.

**Per-class recall** — which classes it gives up on. A single macro number says
that something is wrong; the table says what.

**A significance test** — the difference between two models is a sample
statistic over a few hundred records, and on this corpus the two models differ
by two tenths of a point. Reporting that as "the neural model is better" would be
reporting noise. :func:`mcnemar` says whether the difference survives.

## Why McNemar and not a two-proportion test

The two models are evaluated on **the same records**, so the samples are paired
and a test that assumes independence throws away that pairing — and with it most
of its power. McNemar uses only the records the models disagree about, which is
exactly the evidence that distinguishes them.

The exact binomial form is used rather than the chi-squared approximation,
because the approximation needs roughly 25 discordant pairs to be trustworthy and
two similar models on a few hundred records routinely produce fewer. The same
choice, for the same reason, as the evaluation harness elsewhere in this series.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from dslm.errors import EvaluationError
from dslm.model.features import LabelSpace


@dataclass(frozen=True, slots=True)
class ClassScore:
    """Precision, recall and F1 for one class."""

    label: int
    support: int
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, Any]:
        """Serialise for a report."""
        return {
            "label": self.label,
            "support": self.support,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
        }


@dataclass(frozen=True, slots=True)
class Metrics:
    """What a model scored on one split."""

    name: str
    records: int
    accuracy: float
    macro_f1: float
    per_class: tuple[ClassScore, ...]
    #: Mean probability assigned to the predicted class. A model that is right
    #: 92% of the time and 99% confident is differently wrong from one that is
    #: right 92% of the time and knows it.
    mean_confidence: float
    #: Accuracy on the records the model was most confident about — the top
    #: decile. If this is not far above the overall accuracy, the confidence
    #: carries no information and should not be used to route or to abstain.
    top_decile_accuracy: float
    parameters: int

    @property
    def worst_class(self) -> ClassScore | None:
        """The class with the lowest F1, which is where to look first."""
        return min(self.per_class, key=lambda score: score.f1) if self.per_class else None

    def summary(self) -> str:
        """One line, for a terminal."""
        worst = self.worst_class
        tail = f", worst class {worst.label} at F1 {worst.f1:.3f}" if worst else ""
        return (
            f"{self.name}: accuracy {self.accuracy:.4f}, macro F1 {self.macro_f1:.4f} "
            f"over {self.records} record(s){tail}"
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialise for a report."""
        return {
            "name": self.name,
            "records": self.records,
            "accuracy": round(self.accuracy, 6),
            "macro_f1": round(self.macro_f1, 6),
            "mean_confidence": round(self.mean_confidence, 6),
            "top_decile_accuracy": round(self.top_decile_accuracy, 6),
            "parameters": self.parameters,
            "per_class": [score.as_dict() for score in self.per_class],
        }


def evaluate(  # noqa: PLR0913 - predictions, probabilities, targets and the space
    name: str,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    targets: np.ndarray,
    space: LabelSpace,
    *,
    parameters: int = 0,
) -> Metrics:
    """Score one model's predictions."""
    if targets.size == 0:
        raise EvaluationError(
            "there are no records to score.",
            remedy=(
                "An empty evaluation set has no accuracy. Reporting 1.0 for it is the "
                "single most misleading thing this code could do."
            ),
        )
    if predictions.shape != targets.shape:
        raise EvaluationError(
            f"{predictions.shape[0]} prediction(s) for {targets.shape[0]} target(s).",
            remedy="The model and the split disagree about how many records there are.",
        )

    correct = predictions == targets
    per_class: list[ClassScore] = []
    for index in range(space.size):
        predicted = predictions == index
        actual = targets == index
        true_positive = int((predicted & actual).sum())
        precision = true_positive / int(predicted.sum()) if predicted.any() else 0.0
        recall = true_positive / int(actual.sum()) if actual.any() else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class.append(
            ClassScore(
                label=space.label_at(index),
                support=int(actual.sum()),
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    confidence = probabilities[np.arange(targets.size), predictions]
    decile = max(1, targets.size // 10)
    most_confident = np.argsort(-confidence)[:decile]

    return Metrics(
        name=name,
        records=int(targets.size),
        accuracy=float(correct.mean()),
        macro_f1=float(np.mean([score.f1 for score in per_class])) if per_class else 0.0,
        per_class=tuple(per_class),
        mean_confidence=float(confidence.mean()),
        top_decile_accuracy=float(correct[most_confident].mean()),
        parameters=parameters,
    )


@dataclass(frozen=True, slots=True)
class Comparison:
    """Whether the difference between two models is real."""

    left: str
    right: str
    #: Records the left model got right and the right model got wrong, and the
    #: other way round. Only these carry information about the difference.
    left_only: int
    right_only: int
    p_value: float
    alpha: float

    @property
    def discordant(self) -> int:
        """How many records the two models disagreed about."""
        return self.left_only + self.right_only

    @property
    def significant(self) -> bool:
        """Is the difference distinguishable from chance at *alpha*?"""
        return self.p_value < self.alpha

    def summary(self) -> str:
        """One line, for a terminal. Says 'indistinguishable', never 'the same'."""
        if not self.significant:
            return (
                f"{self.left} and {self.right} are statistically indistinguishable "
                f"(p = {self.p_value:.3f} over {self.discordant} disagreement(s); "
                f"absence of evidence, not evidence of absence)"
            )
        better, worse = (
            (self.left, self.right) if self.left_only > self.right_only else (self.right, self.left)
        )
        return (
            f"{better} beats {worse} (p = {self.p_value:.4f} over "
            f"{self.discordant} disagreement(s))"
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialise for a report."""
        return {
            "left": self.left,
            "right": self.right,
            "left_only_correct": self.left_only,
            "right_only_correct": self.right_only,
            "discordant": self.discordant,
            "p_value": round(self.p_value, 8),
            "alpha": self.alpha,
            "significant": self.significant,
        }


def mcnemar(  # noqa: PLR0913 - two named models, their predictions and the targets
    left_name: str,
    left: np.ndarray,
    right_name: str,
    right: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float = 0.05,
) -> Comparison:
    """Exact McNemar test over the records the two models disagree about.

    The null hypothesis is that a discordant record is equally likely to fall
    either way, so the count is binomial with p = 0.5 and the two-sided p-value
    is computed exactly rather than approximated.

    With no disagreements the p-value is 1.0: two models that made identical
    predictions provide no evidence that either is better, which is a different
    statement from "they are equally good" and is worded that way in
    :meth:`Comparison.summary`.
    """
    if not (left.shape == right.shape == targets.shape):
        raise EvaluationError(
            "the two models' predictions and the targets are different lengths.",
            remedy="A paired test needs both models scored on the same records, in order.",
        )

    left_correct = left == targets
    right_correct = right == targets
    left_only = int((left_correct & ~right_correct).sum())
    right_only = int((right_correct & ~left_correct).sum())
    total = left_only + right_only

    if total == 0:
        p_value = 1.0
    else:
        smaller = min(left_only, right_only)
        tail = sum(math.comb(total, k) for k in range(smaller + 1)) / (2**total)
        p_value = min(1.0, 2.0 * tail)

    return Comparison(
        left=left_name,
        right=right_name,
        left_only=left_only,
        right_only=right_only,
        p_value=p_value,
        alpha=alpha,
    )


def confusion(predictions: np.ndarray, targets: np.ndarray, space: LabelSpace) -> list[list[int]]:
    """Return the confusion matrix as rows of true label by predicted label."""
    matrix = np.zeros((space.size, space.size), dtype=np.int64)
    np.add.at(matrix, (targets, predictions), 1)
    return [[int(value) for value in row] for row in matrix]


def most_confused(
    predictions: np.ndarray, targets: np.ndarray, space: LabelSpace, *, limit: int = 5
) -> list[tuple[int, int, int]]:
    """The (true, predicted, count) triples the model gets wrong most often.

    More useful than the whole matrix in a summary: twelve classes is a 144-cell
    table nobody reads, and the five cells that matter are the off-diagonal ones
    with the largest counts.
    """
    matrix = confusion(predictions, targets, space)
    pairs: list[tuple[int, int, int]] = []
    for row, values in enumerate(matrix):
        for column, count in enumerate(values):
            if row != column and count:
                pairs.append((space.label_at(row), space.label_at(column), count))
    pairs.sort(key=lambda item: (-item[2], item[0], item[1]))
    return pairs[:limit]


def label_noise_ceiling(records: Sequence[Any]) -> float | None:
    """The best accuracy achievable on a corpus whose labels carry known noise.

    Only meaningful for the synthesised corpus, where the generator recorded the
    true chapter of every record it mislabelled. Returns ``None`` when nothing in
    the corpus claims to know — because a ceiling asserted without evidence is
    worse than no ceiling, and every real corpus is in that position.
    """
    known = [record for record in records if getattr(record, "meta", {}).get("true_label")]
    if not known:
        return None
    return 1.0 - len(known) / len(records)
