"""The committed baseline, and the regression gate over it.

A baseline is a file in the repository recording what the model scored when
somebody last looked. Its whole purpose is to turn "the model got worse" from a
thing somebody notices into a thing the build says.

Three decisions worth stating, because each has an obvious alternative that is
worse.

**The gate is one-sided with a tolerance, not an equality.** Requiring the exact
number would fail on every platform whose BLAS accumulates differently — this
project measured that difference and it is real (see ADR-003). Requiring only
"not worse" would let a model drift downwards a tenth of a point at a time
forever. So: a drop of more than the tolerance fails, and an *improvement* larger
than the tolerance is reported and does not fail, but is flagged as needing the
baseline updated. A silent improvement is how a baseline stops meaning anything.

**The baseline records what it was measured on.** The corpus digest, the split
digest, the tokenizer digest. A baseline compared against a different test set is
not a comparison, and without the digests nothing would notice. Comparing across
a changed split is refused rather than warned about.

**The contamination verdict is part of the baseline.** A number measured on a
contaminated split should never become the standard that future runs are held to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dslm.errors import BaselineError
from dslm.evaluate.metrics import Metrics

#: The baseline format version.
BASELINE_SCHEMA_VERSION = 1

#: How far a metric may fall before the gate fails.
#:
#: 0.005 — half a point. Chosen against the measured cross-platform spread,
#: which for this model is a 2.4e-14 relative difference in the weights and no
#: difference at all in the metrics; the tolerance is therefore not absorbing
#: floating-point noise, it is absorbing the genuine run-to-run variation that
#: comes from a different NumPy version resolving a different BLAS.
DEFAULT_TOLERANCE = 0.005


@dataclass(frozen=True, slots=True)
class Baseline:
    """What the models scored last time, and what they were measured on."""

    metrics: dict[str, dict[str, float]]
    corpus_digest: str
    split_digest: str
    tokenizer_digest: str
    contamination_rate: float
    contamination_calibrated: bool
    recorded_at: str
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Serialise."""
        return {
            "dslm_baseline": BASELINE_SCHEMA_VERSION,
            "recorded_at": self.recorded_at,
            "note": self.note,
            "measured_on": {
                "corpus_digest": self.corpus_digest,
                "split_digest": self.split_digest,
                "tokenizer_digest": self.tokenizer_digest,
                "contamination_rate": round(self.contamination_rate, 6),
                "contamination_calibrated": self.contamination_calibrated,
            },
            "metrics": {
                name: {key: round(value, 6) for key, value in scores.items()}
                for name, scores in sorted(self.metrics.items())
            },
        }

    def save(self, path: str | Path) -> Path:
        """Write the baseline as JSON and return the path."""
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return file


def from_metrics(  # noqa: PLR0913 - a baseline records everything it was measured on
    metrics: dict[str, Metrics],
    *,
    corpus_digest: str,
    split_digest: str,
    tokenizer_digest: str,
    contamination_rate: float,
    contamination_calibrated: bool,
    recorded_at: str,
    note: str = "",
) -> Baseline:
    """Build a baseline from a set of scored models."""
    return Baseline(
        metrics={
            name: {"accuracy": score.accuracy, "macro_f1": score.macro_f1}
            for name, score in metrics.items()
        },
        corpus_digest=corpus_digest,
        split_digest=split_digest,
        tokenizer_digest=tokenizer_digest,
        contamination_rate=contamination_rate,
        contamination_calibrated=contamination_calibrated,
        recorded_at=recorded_at,
        note=note,
    )


def load(path: str | Path) -> Baseline:
    """Read a baseline written by :meth:`Baseline.save`."""
    file = Path(path)
    if not file.is_file():
        raise BaselineError(
            f"the baseline {str(file)!r} could not be opened.",
            remedy=(
                "Record one with 'dslm evaluate --update-baseline'. Until there is a "
                "baseline there is nothing to regress against, and the gate says so "
                "rather than passing."
            ),
        )
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BaselineError(
            f"{file.name} is not valid JSON: {exc}.",
            remedy="Re-record it rather than repairing it by hand.",
        ) from exc

    version = payload.get("dslm_baseline")
    if not isinstance(version, int) or version > BASELINE_SCHEMA_VERSION:
        raise BaselineError(
            f"{file.name} was written by a newer version (schema {version!r}).",
            remedy=f"This build understands schema {BASELINE_SCHEMA_VERSION}. Upgrade dslm.",
        )

    measured = payload.get("measured_on") or {}
    metrics = payload.get("metrics") or {}
    if not metrics:
        raise BaselineError(
            f"{file.name} records no metrics.",
            remedy="A baseline with nothing in it passes every gate. Re-record it.",
        )
    return Baseline(
        metrics={
            str(name): {str(key): float(value) for key, value in scores.items()}
            for name, scores in metrics.items()
        },
        corpus_digest=str(measured.get("corpus_digest", "")),
        split_digest=str(measured.get("split_digest", "")),
        tokenizer_digest=str(measured.get("tokenizer_digest", "")),
        contamination_rate=float(measured.get("contamination_rate", 0.0)),
        contamination_calibrated=bool(measured.get("contamination_calibrated", False)),
        recorded_at=str(payload.get("recorded_at", "")),
        note=str(payload.get("note", "")),
    )


@dataclass(frozen=True, slots=True)
class Change:
    """One metric's movement against the baseline."""

    model: str
    metric: str
    was: float
    now: float

    @property
    def delta(self) -> float:
        """How far it moved. Negative is worse."""
        return self.now - self.was

    def as_dict(self) -> dict[str, Any]:
        """Serialise for a report."""
        return {
            "model": self.model,
            "metric": self.metric,
            "was": round(self.was, 6),
            "now": round(self.now, 6),
            "delta": round(self.delta, 6),
        }


@dataclass(frozen=True, slots=True)
class Verdict:
    """What comparing against the baseline decided."""

    regressions: tuple[Change, ...]
    improvements: tuple[Change, ...]
    unchanged: int
    #: Models present now and absent from the baseline, or the reverse. Reported
    #: rather than ignored: a model silently dropped from a comparison is the
    #: easiest way for a regression to go unnoticed.
    added: tuple[str, ...]
    removed: tuple[str, ...]
    tolerance: float

    @property
    def passed(self) -> bool:
        """Did every metric hold?"""
        return not self.regressions and not self.removed

    def summary(self) -> str:
        """One line, for a terminal."""
        if self.passed:
            note = f", {len(self.improvements)} improvement(s)" if self.improvements else ""
            return f"held: no metric fell by more than {self.tolerance:g}{note}"
        parts = []
        if self.regressions:
            worst = min(self.regressions, key=lambda change: change.delta)
            parts.append(
                f"{len(self.regressions)} regression(s), worst {worst.model}.{worst.metric} "
                f"{worst.was:.4f} -> {worst.now:.4f} ({worst.delta:+.4f})"
            )
        if self.removed:
            parts.append(
                f"{len(self.removed)} model(s) no longer reported: {', '.join(self.removed)}"
            )
        return "; ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        """Serialise for a report."""
        return {
            "passed": self.passed,
            "tolerance": self.tolerance,
            "regressions": [change.as_dict() for change in self.regressions],
            "improvements": [change.as_dict() for change in self.improvements],
            "unchanged": self.unchanged,
            "added_models": list(self.added),
            "removed_models": list(self.removed),
        }


def compare(
    baseline: Baseline,
    metrics: dict[str, Metrics],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    split_digest: str = "",
) -> Verdict:
    """Compare *metrics* against *baseline*.

    Refuses outright when the split has changed, rather than reporting a
    comparison between numbers measured on different data. That is not a
    conservative choice — a regression gate whose two sides were measured on
    different test sets produces confident nonsense in both directions.
    """
    if split_digest and baseline.split_digest and split_digest != baseline.split_digest:
        raise BaselineError(
            "the baseline was recorded on a different split.",
            remedy=(
                f"The baseline records {baseline.split_digest[:23]}... and this run used "
                f"{split_digest[:23]}.... Comparing accuracies measured on different test "
                "sets is not a comparison. Re-record the baseline, in its own commit, so "
                "that the change of split is visible in review."
            ),
        )

    regressions: list[Change] = []
    improvements: list[Change] = []
    unchanged = 0

    for model, scores in sorted(baseline.metrics.items()):
        current = metrics.get(model)
        if current is None:
            continue
        now = {"accuracy": current.accuracy, "macro_f1": current.macro_f1}
        for metric, was in sorted(scores.items()):
            value = now.get(metric)
            if value is None:
                continue
            change = Change(model=model, metric=metric, was=was, now=value)
            if change.delta < -tolerance:
                regressions.append(change)
            elif change.delta > tolerance:
                improvements.append(change)
            else:
                unchanged += 1

    return Verdict(
        regressions=tuple(regressions),
        improvements=tuple(improvements),
        unchanged=unchanged,
        added=tuple(sorted(set(metrics) - set(baseline.metrics))),
        removed=tuple(sorted(set(baseline.metrics) - set(metrics))),
        tolerance=tolerance,
    )
