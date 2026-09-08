"""The three report shapes, and especially the sections nobody sees until a gate fails."""

from __future__ import annotations

import json

# Parsing XML this package generated three lines earlier, in a test. There is no
# untrusted document here and nothing to defuse; bandit excludes tests/ for the
# same reason. The *writer* is the side that matters, and it is justified in
# src/dslm/evaluate/report.py and security/audit-exceptions.md.
from xml.etree import ElementTree

import pytest

from dslm.corpus.contamination import calibrate, contamination_report
from dslm.corpus.record import Corpus
from dslm.evaluate import baseline as baselines
from dslm.evaluate.metrics import Comparison, Metrics, mcnemar
from dslm.evaluate.report import (
    Evaluation,
    render_json,
    render_junit,
    render_markdown,
)

pytestmark = pytest.mark.integration


def metrics(name: str, accuracy: float, *, parameters: int = 100) -> Metrics:
    return Metrics(
        name=name,
        records=200,
        accuracy=accuracy,
        macro_f1=accuracy - 0.001,
        per_class=(),
        mean_confidence=0.88,
        top_decile_accuracy=min(1.0, accuracy + 0.05),
        parameters=parameters,
    )


@pytest.fixture(scope="module")
def clean(small: Corpus, small_null: Corpus):
    calibration = calibrate(small, small_null, drop_shared=True)
    return contamination_report(small, small_null, calibration=calibration)


@pytest.fixture
def held(clean) -> Evaluation:
    return Evaluation(
        metrics={"naive-bayes": metrics("naive-bayes", 0.92), "neural": metrics("neural", 0.918)},
        contamination=clean,
        verdict=None,
        comparisons=(),
        corpus_digest="sha256:corpus",
        split_digest="sha256:split",
        tokenizer_digest="sha256:tok",
        ceiling=0.96,
        confusions=((21, 32, 7), (34, 36, 4)),
    )


@pytest.fixture
def refused(clean) -> Evaluation:
    return Evaluation(
        metrics={},
        contamination=clean,
        verdict=None,
        comparisons=(),
        corpus_digest="sha256:corpus",
        split_digest="sha256:split",
        tokenizer_digest="sha256:tok",
        refused="18% of the evaluation set also appears in training.",
    )


class TestJson:
    def test_passed_is_the_first_key(self, held: Evaluation):
        # A script that greps rather than parses should be able to answer the
        # only question that matters without reading the whole document.
        assert next(iter(json.loads(render_json(held)))) == "passed"

    def test_a_held_run_carries_its_metrics_and_provenance(self, held: Evaluation):
        document = json.loads(render_json(held))
        assert set(document["metrics"]) == {"naive-bayes", "neural"}
        assert document["measured_on"]["split_digest"] == "sha256:split"
        assert document["label_noise_ceiling"] == pytest.approx(0.96)

    def test_a_refused_run_carries_no_metric_at_all(self, refused: Evaluation):
        document = json.loads(render_json(refused))
        assert document["passed"] is False
        assert "metrics" not in document
        assert document["refused"].startswith("18%")

    def test_confusions_are_reported_as_labels(self, held: Evaluation):
        document = json.loads(render_json(held))
        assert document["most_confused"][0] == {"true": 21, "predicted": 32, "count": 7}


class TestJUnit:
    def test_it_parses_as_xml(self, held: Evaluation):
        assert ElementTree.fromstring(render_junit(held)).tag == "testsuite"

    def test_a_refused_run_is_a_failing_case(self, refused: Evaluation):
        suite = ElementTree.fromstring(render_junit(refused))
        case = next(item for item in suite if item.tag == "testcase")
        assert case.get("name") == "contamination"
        assert [child.tag for child in case] == ["failure"]

    def test_no_baseline_is_skipped_not_passed(self, held: Evaluation):
        # A green suite must not be readable as "no regression" when nothing was
        # compared.
        suite = ElementTree.fromstring(render_junit(held))
        assert suite.get("skipped") == "1"
        regression = next(
            item for item in suite if item.tag == "testcase" and item.get("name") == "regression"
        )
        assert [child.tag for child in regression] == ["skipped"]

    def test_a_regression_is_its_own_failing_case(self, held: Evaluation, clean):
        baseline = baselines.from_metrics(
            {"neural": metrics("neural", 0.99)},
            corpus_digest="c",
            split_digest="sha256:split",
            tokenizer_digest="t",
            contamination_rate=0.0,
            contamination_calibrated=True,
            recorded_at="2026-09-08T00:00:00+00:00",
        )
        verdict = baselines.compare(baseline, {"neural": metrics("neural", 0.80)})
        evaluation = Evaluation(
            metrics=held.metrics,
            contamination=clean,
            verdict=verdict,
            comparisons=(),
            corpus_digest="c",
            split_digest="s",
            tokenizer_digest="t",
        )
        suite = ElementTree.fromstring(render_junit(evaluation))
        assert suite.get("failures") == "1"

    def test_the_properties_record_what_was_measured(self, held: Evaluation):
        suite = ElementTree.fromstring(render_junit(held))
        block = suite.find("properties")
        assert block is not None
        properties = {item.get("name"): item.get("value") for item in block}
        assert properties["split_digest"] == "sha256:split"


class TestMarkdownWhenTheNewsIsBad:
    """The sections a reader only ever sees when something failed."""

    def test_a_held_run_says_so_on_the_first_line(self, held: Evaluation):
        assert render_markdown(held).splitlines()[0] == "# Evaluation: held"

    def test_a_refused_run_says_so_and_prints_no_accuracy(self, refused: Evaluation):
        rendered = render_markdown(refused)
        assert rendered.splitlines()[0] == "# Evaluation: REFUSED"
        assert "Accuracy" not in rendered
        assert "18%" in rendered

    def test_the_reason_comes_before_anything_else(self, refused: Evaluation):
        rendered = render_markdown(refused)
        assert rendered.index("18%") < rendered.index("## Contamination")

    def test_a_regression_table_comes_before_the_model_table(self, held: Evaluation, clean):
        baseline = baselines.from_metrics(
            {"neural": metrics("neural", 0.99)},
            corpus_digest="c",
            split_digest="sha256:split",
            tokenizer_digest="t",
            contamination_rate=0.0,
            contamination_calibrated=True,
            recorded_at="2026-09-08T00:00:00+00:00",
        )
        evaluation = Evaluation(
            metrics=held.metrics,
            contamination=clean,
            verdict=baselines.compare(baseline, {"neural": metrics("neural", 0.80)}),
            comparisons=(),
            corpus_digest="c",
            split_digest="s",
            tokenizer_digest="t",
        )
        rendered = render_markdown(evaluation)
        assert rendered.splitlines()[0] == "# Evaluation: REGRESSED"
        assert rendered.index("## What changed") < rendered.index("## Models")

    def test_the_ceiling_is_shown_beside_the_models(self, held: Evaluation):
        assert "label-noise ceiling" in render_markdown(held)

    def test_the_significance_test_is_rendered(self, held: Evaluation, clean):
        import numpy as np

        targets = np.zeros(50, dtype=np.int32)
        comparison = mcnemar("a", targets, "b", targets, targets)
        evaluation = Evaluation(
            metrics=held.metrics,
            contamination=clean,
            verdict=None,
            comparisons=(comparison,),
            corpus_digest="c",
            split_digest="s",
            tokenizer_digest="t",
        )
        rendered = render_markdown(evaluation)
        assert "## Is the difference real?" in rendered
        assert "indistinguishable" in rendered

    def test_a_run_with_no_baseline_says_that_is_not_the_same_as_no_regression(
        self, held: Evaluation
    ):
        assert "not** the same as no regression" in render_markdown(held)

    def test_the_footer_records_what_was_measured(self, held: Evaluation):
        rendered = render_markdown(held)
        assert "corpus `sha256:corpus`" in rendered
        assert "tokenizer `sha256:tok`" in rendered

    def test_the_calibration_is_shown_when_there_is_one(self, held: Evaluation):
        assert "Calibration:" in render_markdown(held)


class TestComparisonWording:
    def test_it_never_claims_two_models_are_the_same(self):
        # "Indistinguishable" and "the same" are different claims, and only one
        # of them is supported by a p-value.
        comparison = Comparison(
            left="a", right="b", left_only=1, right_only=1, p_value=1.0, alpha=0.05
        )
        summary = comparison.summary()
        assert "indistinguishable" in summary
        assert "absence of evidence" in summary
