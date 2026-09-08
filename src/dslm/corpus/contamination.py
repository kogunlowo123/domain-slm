"""The contamination gate: is this evaluation a measurement at all?

This is the module the repository exists for.

A model's reported accuracy is a claim about data it has not seen. When part of
the evaluation set also appears in training, the number is partly a memorisation
score — and there is no way to tell from the number itself. A contaminated 96%
and a clean 96% look identical. The only defence is to measure the overlap
directly, before believing either.

So the check runs *before* the metric is reported, and above the limit it
**refuses to report the metric at all**. Not a warning printed above the
accuracy: a non-zero exit and no number. A warning above a number is read as a
number.

## How overlap is measured

By **n-gram containment**, not by Jaccard similarity. The two answer different
questions and only one of them is the right one.

Jaccard asks "how similar are these two records?" — symmetric, and diluted by
everything the evaluation record contains that the training record does not.
Containment asks "what fraction of this evaluation record's n-grams appear
*anywhere* in training?", which is the actual question: a record whose every
phrase the model has already seen is not a test, however long it is.

The training side is one set of every n-gram it contains. That is the right
shape — the model saw all of training, not one record at a time — and it is what
makes this affordable: one pass to build the set, one probe per record.

## Why the published threshold does not transfer, and what replaces it

The usual rule, from the technical reports of several published models, is
"13-gram containment at or above 0.5 is contamination". This project inherited
it, measured it, and found that **on short formulaic text the number it produces
is background rather than signal.**

The measurement is reproducible from the shipped corpus and is asserted in the
test suite. The ``holdout`` corpus is generated from the same grammar with a
different seed, so it shares no record with training *by construction*. Scored
against training by that rule:

======================================  ==========
corpus                                  hit rate
======================================  ==========
``holdout`` — **cannot** be contaminated  1.80%
the genuine test split                    1.25%
======================================  ==========

Data that cannot be contaminated scores *higher* than data that might be. The
rule cannot even order the two correctly, so a threshold applied to its output
decides on noise.

Nothing is wrong with the arithmetic. Maintenance write-ups are formulaic —
whole clauses are standardised — so two entirely different work orders routinely
share a thirteen-word run. The rule measures how formulaic the domain is. On
web-scale prose, where it was calibrated, that background is near zero and the
rule works. Here it is the whole reading. Restricting the index to *distinctive*
n-grams — those appearing in exactly one training record — halves both numbers
and leaves the ordering just as wrong (0.54% for test against a 1.13%
background): the background is a property of the domain, not of the index.

**So the null calibrates the threshold, not the verdict.** Take a corpus known
to be disjoint from training, score every record, and set the threshold at a
high quantile of that distribution — the containment level disjoint data
essentially never reaches. On the shipped corpus that is **0.818**, not 0.5.
Above it, a record is a hit.

The first design calibrated differently: it compared *rates*, the evaluation
set's against the null's, and failed on the excess. That was measured too, and
it was too blunt — a clean split sits below the background, so contamination had
to climb past it before the gate noticed, and a **10% verbatim injection
passed**. Calibrating the threshold instead makes the reported rate mean what it
says. Measured over the shipped corpus, with records copied from training into
test at a known rate:

=========  ==========
injected    measured
=========  ==========
0%          0.00%
2%          1.97%
5%          4.84%
10%         9.86%
25%         24.91%
=========  ==========

Within 0.16 points at every level. That is no longer a flag; it is an estimate
of how much of the evaluation set the model has already seen.

And it catches the mistake it exists for. A split taken *before* deduplication —
the ordinary way to get this wrong — measures **4.67%** against the 2% limit and
fails, while the correctly built split measures 0.00% and passes.

## When there is no null

The tool reports the raw rate and **refuses to gate on it**. A number nobody can
interpret is not a gate, and on this corpus the uncalibrated rule would fail a
clean split. ``allow_uncalibrated`` opts back in for text where the published
rule genuinely applies — a deliberate act, visible in a command line and a diff.

## The other way this check fails

A 13-gram is longer than many records here, and a check whose n-gram never fits
is a check that always passes. :class:`Report` carries the coverage, and the
gate refuses a split where most of the evaluation set was too short to examine.
"The check did not apply" and "the check passed" are different facts.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from dslm.corpus.curate import normalise
from dslm.corpus.record import Corpus, Record
from dslm.errors import ContaminationError

#: N-gram width, in words. Thirteen is the width used in several published model
#: reports, inherited rather than invented for the same reason the burn-rate
#: table was inherited in the telemetry tool in this series: a number chosen here
#: would have had to be defended here, and would have been worse.
NGRAM_WORDS = 13

#: A shorter width, always also reported and never gated on. A record too short
#: for a 13-gram is invisible to the primary check; 8 catches overlap the primary
#: width misses in short text, at a false-positive rate on formulaic domain text
#: that is genuinely too high to gate on.
SHORT_NGRAM_WORDS = 8

#: The containment at or above which a record is a hit **when no calibration is
#: available**. This is the published figure. It is a poor threshold for
#: formulaic text — the module docstring says why — and calibration replaces it.
DEFAULT_THRESHOLD = 0.5

#: The quantile of the null distribution used as the calibrated threshold.
#:
#: 0.999 rather than 0.99. A quantile is a promise about how often disjoint data
#: will be called contaminated, and at p99 that is one record in a hundred — a
#: floor under every measurement the gate makes. At p99.9 the shipped null's own
#: hit rate at the chosen threshold falls to 0.13%, a clean split measures
#: exactly zero, and a contaminated one measures its true rate to within 0.16
#: points.
DEFAULT_QUANTILE = 0.999

#: The hit rate allowed before the gate fails. Not zero: one duplicated record in
#: a thousand does not invalidate a measurement, and a gate that fails on it is a
#: gate someone will disable. With a calibrated threshold this is a limit on real
#: contamination rather than a fudge factor.
DEFAULT_MAX_CONTAMINATED = 0.02

#: The fraction of the evaluation set that must be long enough to check. A check
#: that could not be applied to most of the evaluation set has not passed; it has
#: abstained.
DEFAULT_MIN_COVERAGE = 0.5


def ngrams(text: str, *, width: int = NGRAM_WORDS) -> frozenset[str]:
    """Return the set of overlapping *width*-word n-grams of *text*.

    Empty when the text is shorter than the width — deliberately empty rather
    than falling back to the whole string, because a caller has to be able to
    tell "no overlap" from "not checkable" and a fallback erases the difference.
    """
    words = normalise(text).split()
    if len(words) < width:
        return frozenset()
    return frozenset(
        " ".join(words[index : index + width]) for index in range(len(words) - width + 1)
    )


def build_index(records: Iterable[Record], *, width: int = NGRAM_WORDS) -> frozenset[str]:
    """Every n-gram on the training side, as one set.

    One set for the whole side, not one per record: the model saw all of
    training, and an evaluation record whose phrases are spread over three
    training records is exactly as contaminated as one matching a single record.
    """
    index: set[str] = set()
    for record in records:
        index |= ngrams(record.text, width=width)
    return frozenset(index)


def containment(text: str, index: frozenset[str], *, width: int = NGRAM_WORDS) -> float | None:
    """What fraction of *text*'s n-grams appear in *index*?

    ``None`` when the text is too short to have any — which is not zero, and a
    caller that treats it as zero reports a clean result for a record that was
    never checked.
    """
    grams = ngrams(text, width=width)
    if not grams:
        return None
    return len(grams & index) / len(grams)


def _quantile(values: Sequence[float], q: float) -> float:
    """The *q*-quantile of *values*, by nearest rank.

    Nearest rank rather than interpolation: the threshold has to be a value the
    null actually produced, so that "disjoint data does not exceed this" is a
    statement about observations rather than about an average of two of them.
    """
    if not values:
        raise ContaminationError(
            "the null corpus produced no scores to calibrate against.",
            remedy="Every null record was too short for the n-gram width. Lower --ngram-width.",
        )
    ordered = sorted(values)
    rank = min(int(q * len(ordered)), len(ordered) - 1)
    return ordered[rank]


@dataclass(frozen=True, slots=True)
class Calibration:
    """A contamination threshold derived from data known to be disjoint."""

    threshold: float
    quantile: float
    width: int
    #: How many null records were scored.
    scored: int
    #: How many were too short for the n-gram width.
    not_checkable: int
    #: Records removed from the null because they also appeared in training.
    #: Reported, because a null corpus that silently shrinks is one nobody would
    #: think to question.
    dropped: int
    #: The null's own hit rate at the calibrated threshold, approximately
    #: ``1 - quantile`` by construction. Printed so a reader can see that the
    #: calibration did what it claims.
    null_rate: float
    #: The published threshold's hit rate on this null: the number that makes the
    #: case for calibrating at all.
    uncalibrated_null_rate: float

    def summary(self) -> str:
        """One line, for a terminal."""
        return (
            f"threshold {self.threshold:.3f} at the {self.quantile:.1%} quantile of "
            f"{self.scored} disjoint record(s); the published {DEFAULT_THRESHOLD:g} threshold "
            f"would call {self.uncalibrated_null_rate:.1%} of them contaminated"
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialise for a report."""
        return {
            "threshold": round(self.threshold, 6),
            "quantile": self.quantile,
            "width": self.width,
            "null_records_scored": self.scored,
            "null_records_not_checkable": self.not_checkable,
            "null_records_dropped": self.dropped,
            "null_rate_at_threshold": round(self.null_rate, 6),
            "null_rate_at_published_threshold": round(self.uncalibrated_null_rate, 6),
        }


@dataclass(frozen=True, slots=True)
class Hit:
    """One evaluation record that overlaps training."""

    record_id: str
    containment: float
    label: int
    excerpt: str

    def as_dict(self) -> dict[str, Any]:
        """Serialise for a report."""
        return {
            "record_id": self.record_id,
            "containment": round(self.containment, 4),
            "label": self.label,
            "excerpt": self.excerpt,
        }


@dataclass(frozen=True, slots=True)
class Report:
    """What the contamination check found."""

    evaluated: int
    #: Records too short to have an n-gram of the primary width. Not "clean".
    not_checkable: int
    hits: tuple[Hit, ...]
    threshold: float
    width: int
    #: Hits at the shorter width, reported alongside and never gated on.
    short_width: int
    short_hits: int
    #: The calibration that produced ``threshold``, when there was one. Without
    #: it this is a measurement with no scale, and :func:`enforce` says so.
    calibration: Calibration | None = None

    @property
    def calibrated(self) -> bool:
        """Was the threshold derived from data known to be disjoint?"""
        return self.calibration is not None

    @property
    def total(self) -> int:
        """How many evaluation records were considered."""
        return self.evaluated + self.not_checkable

    @property
    def coverage(self) -> float:
        """The fraction of the evaluation set the primary check could apply to."""
        return self.evaluated / self.total if self.total else 0.0

    @property
    def rate(self) -> float:
        """The contaminated fraction **of the records that were checkable**.

        Of the checkable ones, not of the total. Dividing by the total would let
        a split with poor coverage report a reassuringly small number for the
        arithmetic reason that most of it was never examined.
        """
        return len(self.hits) / self.evaluated if self.evaluated else 0.0

    def summary(self) -> str:
        """One line, for a terminal."""
        if not self.evaluated:
            return (
                f"no evaluation record reached {self.width} words, so nothing was checked "
                f"({self.not_checkable} record(s) too short)"
            )
        scale = "calibrated" if self.calibrated else "UNCALIBRATED"
        return (
            f"{len(self.hits)} of {self.evaluated} checkable record(s) contaminated "
            f"({self.rate:.2%}) at {self.width}-gram containment >= {self.threshold:.3f} "
            f"({scale}); coverage {self.coverage:.1%}"
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialise for a report."""
        payload: dict[str, Any] = {
            "total": self.total,
            "evaluated": self.evaluated,
            "not_checkable": self.not_checkable,
            "coverage": round(self.coverage, 4),
            "contaminated": len(self.hits),
            "rate": round(self.rate, 6),
            "threshold": round(self.threshold, 6),
            "calibrated": self.calibrated,
            "width": self.width,
            "short_width": self.short_width,
            "short_width_hits": self.short_hits,
            "examples": [hit.as_dict() for hit in self.hits[:10]],
        }
        if self.calibration is not None:
            payload["calibration"] = self.calibration.as_dict()
        return payload


def calibrate(
    train: Sequence[Record] | Corpus,
    null: Sequence[Record] | Corpus,
    *,
    quantile: float = DEFAULT_QUANTILE,
    width: int = NGRAM_WORDS,
    drop_shared: bool = False,
) -> Calibration:
    """Derive a contamination threshold from a corpus known to be disjoint.

    The null must share no record with *train*: a null that overlaps training
    calibrates a threshold that is partly contamination, making the gate lenient
    in exactly the situation it exists for. So disjointness is **checked** rather
    than trusted — one set operation against the alternative of a gate that is
    quietly wrong.

    ``drop_shared`` removes the offending records and reports how many. It is off
    by default and has to be asked for, because a null corpus that silently
    shrinks is one nobody would think to question. It is not a workaround: two
    corpora from one grammar with different seeds really can produce the same
    sentence, and the shipped ``holdout`` collides with training on exactly one
    record in 1,500 — which is what a finite vocabulary does, and is worth seeing
    rather than hiding.
    """
    if not 0.0 < quantile < 1.0:
        raise ContaminationError(
            f"the calibration quantile is {quantile:g}; it must be strictly between 0 and 1.",
            remedy="1.0 would set the threshold at the null's maximum, which no data exceeds.",
        )

    train_records = list(train)
    null_records = list(null)
    seen = {record.fingerprint() for record in train_records}
    shared = [record for record in null_records if record.fingerprint() in seen]
    if shared and not drop_shared:
        raise ContaminationError(
            f"the null corpus shares {len(shared)} record(s) with training, "
            f"beginning with {shared[0].record_id!r}.",
            remedy=(
                "A null corpus calibrates what 'no contamination' looks like, so it has to "
                "contain none. Generate it with a different seed, hold it out before any of "
                "this pipeline runs, or pass --drop-shared to remove the overlapping records "
                "and have the count reported."
            ),
        )
    if shared:
        null_records = [record for record in null_records if record.fingerprint() not in seen]
        if not null_records:
            raise ContaminationError(
                "every record in the null corpus also appears in training.",
                remedy="This is not a null corpus. Generate one from data training never saw.",
            )

    index = build_index(train_records, width=width)
    scores: list[float] = []
    not_checkable = 0
    for record in null_records:
        score = containment(record.text, index, width=width)
        if score is None:
            not_checkable += 1
        else:
            scores.append(score)

    threshold = _quantile(scores, quantile)
    at_threshold = sum(1 for score in scores if score >= threshold) / len(scores)
    at_published = sum(1 for score in scores if score >= DEFAULT_THRESHOLD) / len(scores)
    return Calibration(
        threshold=threshold,
        quantile=quantile,
        width=width,
        scored=len(scores),
        not_checkable=not_checkable,
        dropped=len(shared),
        null_rate=at_threshold,
        uncalibrated_null_rate=at_published,
    )


def contamination_report(  # noqa: PLR0913 - two corpora, a calibration and two widths
    train: Sequence[Record] | Corpus,
    evaluate: Sequence[Record] | Corpus,
    *,
    calibration: Calibration | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    width: int = NGRAM_WORDS,
    short_width: int = SHORT_NGRAM_WORDS,
) -> Report:
    """Measure how much of *evaluate* the model would already have seen in *train*.

    Supply a *calibration* and its threshold and width are used: a threshold and
    the null it came from belong together, and passing one without the other is
    how a report ends up labelled calibrated while using a number from somewhere
    else.
    """
    if calibration is not None:
        threshold = calibration.threshold
        width = calibration.width

    train_records = list(train)
    eval_records = list(evaluate)
    index = build_index(train_records, width=width)
    short_index = build_index(train_records, width=short_width)

    hits: list[Hit] = []
    evaluated = 0
    not_checkable = 0
    short_hits = 0

    for record in eval_records:
        score = containment(record.text, index, width=width)
        if score is None:
            not_checkable += 1
        else:
            evaluated += 1
            if score >= threshold:
                hits.append(
                    Hit(
                        record_id=record.record_id,
                        containment=score,
                        label=record.label,
                        excerpt=record.text[:120],
                    )
                )
        short_score = containment(record.text, short_index, width=short_width)
        if short_score is not None and short_score >= threshold:
            short_hits += 1

    hits.sort(key=lambda hit: (-hit.containment, hit.record_id))
    return Report(
        evaluated=evaluated,
        not_checkable=not_checkable,
        hits=tuple(hits),
        threshold=threshold,
        width=width,
        short_width=short_width,
        short_hits=short_hits,
        calibration=calibration,
    )


def enforce(
    report: Report,
    *,
    max_contaminated: float = DEFAULT_MAX_CONTAMINATED,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    allow_uncalibrated: bool = False,
) -> None:
    """Raise :class:`ContaminationError` unless this evaluation can be trusted.

    Three independent ways to fail, and the last two are the ones people forget.

    **Too much overlap.** The obvious one. The metric would be part memorisation
    and nothing in the number says how much.

    **Too little coverage.** The check could not be applied to enough of the
    evaluation set to mean anything, and a check that did not apply has not
    passed.

    **No calibration.** Without a null there is nothing to say whether this
    domain's background is 0% or 13%, so the rate cannot be read. Refusing is the
    whole design: this would rather return "cannot decide" than a verdict it
    cannot stand behind.
    """
    if report.coverage < min_coverage:
        raise ContaminationError(
            f"only {report.coverage:.1%} of the evaluation set was long enough for a "
            f"{report.width}-gram check; the minimum is {min_coverage:.0%}.",
            remedy=(
                f"{report.not_checkable} of {report.total} record(s) are shorter than "
                f"{report.width} words. Lower --ngram-width to suit the records, or accept "
                "that this split cannot be checked at this width — but do not read the result "
                "as clean. A check that did not apply has not passed."
            ),
        )

    if not report.calibrated and not allow_uncalibrated:
        raise ContaminationError(
            f"the hit rate is {report.rate:.2%} at an uncalibrated threshold, and there is "
            "nothing that says what that means for this data.",
            remedy=(
                "Supply a null corpus that shares no record with training (--null <corpus>); "
                "its containment distribution sets the threshold. On the corpus shipped with "
                "this project the published threshold scores provably disjoint data "
                "*higher* than the real test split, so a rule applied to its output "
                "decides on noise. Pass --allow-uncalibrated to apply it anyway."
            ),
        )

    if report.rate > max_contaminated:
        worst = report.hits[0]
        scale = (
            f"calibrated threshold {report.threshold:.3f}"
            if report.calibrated
            else f"uncalibrated threshold {report.threshold:g}"
        )
        raise ContaminationError(
            f"{len(report.hits)} of {report.evaluated} evaluation record(s) ({report.rate:.2%}) "
            f"also appear in training, at the {scale}; the limit is {max_contaminated:.0%}.",
            remedy=(
                "Any metric measured on this split is part memorisation, and nothing in the "
                "number says how much. Deduplicate before splitting, and group the split by "
                f"duplicate cluster. Worst offender: {worst.record_id} at "
                f"{worst.containment:.0%} containment — {worst.excerpt!r}"
            ),
        )
