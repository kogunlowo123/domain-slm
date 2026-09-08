"""Three renderings of one evaluation.

JSON for machines, JUnit XML so a regression appears beside the unit tests, and
Markdown for a job summary. One evaluation object, three views — the alternative
is three code paths that agree until they do not, and then a summary that says
something different from the exit code.

The rule every renderer obeys: **a number that could not be trusted is not
shown**. When the contamination gate failed there is no accuracy in the JSON, no
passing test case in the XML, and the Markdown says why instead of printing a
figure with a warning above it. A warning above a number is read as a number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from xml.etree import ElementTree  # noqa: ICN001  # nosec B405

from dslm.corpus.contamination import Report as ContaminationReport
from dslm.evaluate.baseline import Verdict
from dslm.evaluate.metrics import Comparison, Metrics

#: Rows before a table is truncated in the Markdown summary. A job summary with
#: two hundred rows is one nobody reads.
LIST_LIMIT = 20


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Everything one run of ``dslm evaluate`` decided."""

    metrics: dict[str, Metrics]
    contamination: ContaminationReport
    #: Present only when the contamination gate passed and a baseline existed.
    verdict: Verdict | None
    #: Pairwise significance tests between the models scored.
    comparisons: tuple[Comparison, ...]
    corpus_digest: str
    split_digest: str
    tokenizer_digest: str
    #: Why no metric is reported, when none is. Empty when the run succeeded.
    refused: str = ""
    #: The best accuracy the labels allow, when the corpus records its own noise.
    ceiling: float | None = None
    #: Most-confused (true, predicted, count) triples, for the leading model.
    confusions: tuple[tuple[int, int, int], ...] = ()

    @property
    def trustworthy(self) -> bool:
        """Can the numbers in this evaluation be believed?"""
        return not self.refused

    @property
    def passed(self) -> bool:
        """Did this run hold — contamination clean, and no regression?"""
        if not self.trustworthy:
            return False
        return self.verdict is None or self.verdict.passed

    def best(self) -> Metrics | None:
        """The highest-accuracy model, for a headline."""
        if not self.metrics:
            return None
        return max(self.metrics.values(), key=lambda score: score.accuracy)


def as_document(evaluation: Evaluation) -> dict[str, Any]:
    """Build the JSON document.

    ``passed`` first, so a script that greps rather than parses can answer the
    only question that matters without reading the rest.
    """
    document: dict[str, Any] = {
        "passed": evaluation.passed,
        "trustworthy": evaluation.trustworthy,
        "measured_on": {
            "corpus_digest": evaluation.corpus_digest,
            "split_digest": evaluation.split_digest,
            "tokenizer_digest": evaluation.tokenizer_digest,
        },
        "contamination": evaluation.contamination.as_dict(),
    }
    if evaluation.refused:
        # No metrics at all. Not "metrics, with a caveat": a number that cannot
        # be trusted must not be available to be quoted out of context.
        document["refused"] = evaluation.refused
        return document

    document["metrics"] = {
        name: score.as_dict() for name, score in sorted(evaluation.metrics.items())
    }
    if evaluation.ceiling is not None:
        document["label_noise_ceiling"] = round(evaluation.ceiling, 6)
    if evaluation.comparisons:
        document["comparisons"] = [item.as_dict() for item in evaluation.comparisons]
    if evaluation.confusions:
        document["most_confused"] = [
            {"true": true, "predicted": predicted, "count": count}
            for true, predicted, count in evaluation.confusions
        ]
    if evaluation.verdict is not None:
        document["baseline"] = evaluation.verdict.as_dict()
    return document


def render_json(evaluation: Evaluation) -> str:
    """Render the evaluation as JSON."""
    return json.dumps(as_document(evaluation), indent=2, ensure_ascii=False) + "\n"


def render_junit(evaluation: Evaluation) -> str:
    """Render the evaluation as JUnit XML.

    One test case per gate, so a contamination failure and a regression are
    distinguishable in a CI dashboard rather than both being "the evaluation
    job failed".
    """
    cases: list[tuple[str, str, str]] = []

    if evaluation.refused:
        cases.append(("contamination", "failure", evaluation.refused))
    else:
        cases.append(("contamination", "", ""))
        if evaluation.verdict is None:
            # No baseline is not a pass. Recorded as skipped so that a green
            # suite cannot be read as "no regression".
            cases.append(("regression", "skipped", "no baseline recorded, so nothing was compared"))
        elif evaluation.verdict.passed:
            cases.append(("regression", "", ""))
        else:
            cases.append(("regression", "failure", evaluation.verdict.summary()))

    failures = sum(1 for _, kind, _ in cases if kind == "failure")
    skipped = sum(1 for _, kind, _ in cases if kind == "skipped")
    suite = ElementTree.Element(
        "testsuite",
        {
            "name": "dslm",
            "tests": str(len(cases)),
            "failures": str(failures),
            "errors": "0",
            "skipped": str(skipped),
        },
    )
    properties = ElementTree.SubElement(suite, "properties")
    for name, value in (
        ("corpus_digest", evaluation.corpus_digest),
        ("split_digest", evaluation.split_digest),
        ("tokenizer_digest", evaluation.tokenizer_digest),
        ("contamination_rate", f"{evaluation.contamination.rate:.6f}"),
    ):
        ElementTree.SubElement(properties, "property", {"name": name, "value": value})

    for name, kind, message in cases:
        case = ElementTree.SubElement(suite, "testcase", {"classname": "dslm.gate", "name": name})
        if kind:
            ElementTree.SubElement(case, kind, {"message": message})

    return ElementTree.tostring(suite, encoding="unicode", xml_declaration=True) + "\n"


def render_markdown(evaluation: Evaluation) -> str:
    """Render the evaluation as a Markdown job summary."""
    if not evaluation.trustworthy:
        return "\n".join(
            [
                "# Evaluation: REFUSED",
                "",
                "No metric is reported, because none of them would mean anything.",
                "",
                f"> {evaluation.refused}",
                "",
                "## Contamination",
                "",
                evaluation.contamination.summary(),
                "",
                (
                    "A number measured on an evaluation set the model was trained on "
                    "is part memorisation, and nothing in the number says how much. "
                    "That is why this reports no accuracy rather than one with a "
                    "warning above it."
                ),
                "",
            ]
        )

    verdict = "held" if evaluation.passed else "REGRESSED"
    lines = [f"# Evaluation: {verdict}", ""]

    if evaluation.verdict is not None and not evaluation.verdict.passed:
        lines.extend(["## What changed", "", evaluation.verdict.summary(), ""])
        lines.extend(["| Model | Metric | Was | Now | Delta |", "| --- | --- | --- | --- | --- |"])
        lines.extend(
            f"| {change.model} | {change.metric} | {change.was:.4f} | "
            f"{change.now:.4f} | {change.delta:+.4f} |"
            for change in evaluation.verdict.regressions[:LIST_LIMIT]
        )
        lines.append("")

    lines.extend(
        [
            "## Models",
            "",
            "| Model | Accuracy | Macro F1 | Parameters | Mean confidence |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for name, score in sorted(evaluation.metrics.items(), key=lambda item: -item[1].accuracy):
        lines.append(
            f"| {name} | {score.accuracy:.4f} | {score.macro_f1:.4f} | "
            f"{score.parameters:,} | {score.mean_confidence:.3f} |"
        )
    if evaluation.ceiling is not None:
        lines.append(f"| _label-noise ceiling_ | {evaluation.ceiling:.4f} | — | — | — |")
    lines.append("")

    if evaluation.comparisons:
        lines.extend(["## Is the difference real?", ""])
        lines.extend(f"- {item.summary()}" for item in evaluation.comparisons)
        lines.append("")

    lines.extend(["## Contamination", "", evaluation.contamination.summary(), ""])
    if evaluation.contamination.calibration is not None:
        lines.append(f"Calibration: {evaluation.contamination.calibration.summary()}")
        lines.append("")

    if evaluation.confusions:
        lines.extend(
            [
                "## Most confused",
                "",
                "| True | Predicted | Records |",
                "| --- | --- | --- |",
            ]
        )
        lines.extend(
            f"| {true} | {predicted} | {count} |"
            for true, predicted, count in evaluation.confusions[:LIST_LIMIT]
        )
        lines.append("")

    if evaluation.verdict is None:
        lines.extend(
            [
                (
                    "> No baseline was recorded, so nothing was compared. That is "
                    "**not** the same as no regression — record one with "
                    "`dslm evaluate --update-baseline`."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            (
                f"corpus `{evaluation.corpus_digest}` · "
                f"split `{evaluation.split_digest}` · "
                f"tokenizer `{evaluation.tokenizer_digest}`"
            ),
            "",
        ]
    )
    return "\n".join(lines)
