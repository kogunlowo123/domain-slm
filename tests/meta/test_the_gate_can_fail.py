"""The gate's own negative controls.

A test suite that has only ever observed the gate passing is indistinguishable
from one that asserts ``True``. Delete this file and the claim on the front of
the README — that a contaminated evaluation is refused rather than reported —
stops being supported by anything.

Each test breaks exactly one thing and asserts the specific failure it should
produce, with a control asserting the healthy pipeline stays green. The control
is the half that is easy to leave out and the half that carries the argument:
without it, "the gate fired" could be a fact about the tool rather than about
the data.

The centrepiece is :class:`TestContaminationIsMeasuredNotGuessed`, which injects
contamination at a **known rate** and asserts the gate both fires and reports
approximately that rate back. A gate that fires is worth something; a gate whose
number is an estimate of the thing it is measuring is worth considerably more,
and only a test with a known answer can tell the two apart.
"""

from __future__ import annotations

import pytest

from dslm.corpus import contamination as contam
from dslm.corpus.curate import apply_curation, find_duplicates
from dslm.corpus.record import Corpus, build_corpus
from dslm.corpus.split import Split, split_corpus
from dslm.errors import BaselineError, ContaminationError
from dslm.evaluate import baseline as baselines
from dslm.evaluate.metrics import Metrics
from tests.conftest import SMALL, make_record

pytestmark = pytest.mark.meta


def inject(split: Split, rate: float) -> Corpus:
    """Return the test split with *rate* of it replaced by verbatim training records."""
    count = int(len(split.test) * rate)
    kept = list(split.test.records)[count:]
    leaked = [
        record.model_copy(update={"record_id": f"leak-{index:04d}"})
        for index, record in enumerate(split.train.records[:count])
    ]
    return build_corpus([*kept, *leaked])


@pytest.fixture(scope="module")
def calibration(pipeline: Split, small_null: Corpus) -> contam.Calibration:
    return contam.calibrate(pipeline.train, small_null, drop_shared=True)


class TestContaminationIsMeasuredNotGuessed:
    def test_the_control_is_clean(self, pipeline: Split, calibration: contam.Calibration):
        # Without this every result below could be a fact about the tool.
        report = contam.contamination_report(pipeline.train, pipeline.test, calibration=calibration)
        contam.enforce(report)
        assert report.rate == 0.0

    @pytest.mark.parametrize("rate", [0.10, 0.25, 0.50])
    def test_injected_contamination_is_detected_and_measured(
        self, pipeline: Split, calibration: contam.Calibration, rate: float
    ):
        report = contam.contamination_report(
            pipeline.train, inject(pipeline, rate), calibration=calibration
        )
        with pytest.raises(ContaminationError, match="also appear in training"):
            contam.enforce(report)
        # Not merely "fired": the reported rate is an estimate of the injected
        # rate, which is the difference between a flag and a measurement.
        assert report.rate == pytest.approx(rate, abs=0.03)

    def test_a_small_injection_is_below_the_limit_by_design(
        self, pipeline: Split, calibration: contam.Calibration
    ):
        # The limit is 2%, and it is not zero on purpose: one duplicated record
        # in a thousand does not invalidate a measurement, and a gate that fails
        # on it is a gate someone will disable.
        report = contam.contamination_report(
            pipeline.train, inject(pipeline, 0.01), calibration=calibration
        )
        contam.enforce(report)


class TestTheOrdinaryMistake:
    """Splitting before deduplicating — the way this actually goes wrong."""

    def test_a_split_taken_before_deduplication_fails_the_gate(
        self, small: Corpus, small_null: Corpus
    ):
        naive = split_corpus(small, seed=1)
        calibration = contam.calibrate(naive.train, small_null, drop_shared=True)
        report = contam.contamination_report(naive.train, naive.test, calibration=calibration)
        with pytest.raises(ContaminationError):
            contam.enforce(report)

    def test_and_the_correct_pipeline_passes_the_same_gate(self, small: Corpus, small_null: Corpus):
        cleaned = apply_curation(small, find_duplicates(small.records))
        good = split_corpus(cleaned, seed=1, curation=find_duplicates(cleaned.records))
        calibration = contam.calibrate(good.train, small_null, drop_shared=True)
        contam.enforce(contam.contamination_report(good.train, good.test, calibration=calibration))


class TestTheGateRefusesRatherThanGuesses:
    def test_without_a_null_it_will_not_decide(self, pipeline: Split):
        # The design. On this corpus the published rule scores provably disjoint
        # data higher than the real test split, so applying it would decide on
        # noise — and deciding on noise looks exactly like deciding.
        report = contam.contamination_report(pipeline.train, pipeline.test)
        with pytest.raises(ContaminationError, match="nothing that says what that means"):
            contam.enforce(report)

    def test_a_null_that_overlaps_training_is_refused(self, pipeline: Split):
        with pytest.raises(ContaminationError, match="shares"):
            contam.calibrate(pipeline.train, pipeline.train)

    def test_a_check_that_could_not_apply_is_not_a_pass(self, pipeline: Split):
        short = build_corpus(
            [
                make_record(record_id=f"s-{index}", text="four words only here")
                for index in range(20)
            ]
        )
        with pytest.raises(ContaminationError, match="long enough"):
            contam.enforce(contam.contamination_report(pipeline.train, short))


class TestTheRegressionGateCanFail:
    @staticmethod
    def _metrics(accuracy: float) -> dict[str, Metrics]:
        return {
            "neural": Metrics(
                name="neural",
                records=100,
                accuracy=accuracy,
                macro_f1=accuracy,
                per_class=(),
                mean_confidence=0.9,
                top_decile_accuracy=accuracy,
                parameters=1,
            )
        }

    def _baseline(self, accuracy: float, *, split: str = "s") -> baselines.Baseline:
        return baselines.from_metrics(
            self._metrics(accuracy),
            corpus_digest="c",
            split_digest=split,
            tokenizer_digest="t",
            contamination_rate=0.0,
            contamination_calibrated=True,
            recorded_at="2026-09-08T00:00:00+00:00",
        )

    def test_an_unchanged_model_holds(self):
        verdict = baselines.compare(self._baseline(0.90), self._metrics(0.90))
        assert verdict.passed

    def test_a_real_drop_fails(self):
        verdict = baselines.compare(self._baseline(0.90), self._metrics(0.80))
        assert not verdict.passed
        assert verdict.regressions[0].delta == pytest.approx(-0.10)

    def test_a_drop_inside_the_tolerance_holds(self):
        # The tolerance absorbs the run-to-run variation a different BLAS
        # produces, not a real regression.
        assert baselines.compare(self._baseline(0.90), self._metrics(0.898)).passed

    def test_an_improvement_is_reported_and_does_not_fail(self):
        # A silent improvement is how a baseline stops meaning anything.
        verdict = baselines.compare(self._baseline(0.90), self._metrics(0.95))
        assert verdict.passed
        assert verdict.improvements

    def test_a_model_that_stops_being_reported_fails(self):
        # The easiest way for a regression to go unnoticed is to stop measuring.
        verdict = baselines.compare(self._baseline(0.90), {})
        assert not verdict.passed
        assert verdict.removed == ("neural",)

    def test_comparing_across_a_changed_split_is_refused_not_reported(self):
        # Two accuracies measured on different test sets are not a comparison,
        # and reporting one produces confident nonsense in both directions.
        with pytest.raises(BaselineError, match="different split"):
            baselines.compare(
                self._baseline(0.90, split="one"), self._metrics(0.90), split_digest="two"
            )

    def test_a_baseline_with_no_metrics_is_refused(self, tmp_path):
        # It would pass every gate.
        import json

        path = tmp_path / "baseline.json"
        path.write_text(json.dumps({"dslm_baseline": 1, "metrics": {}}), encoding="utf-8")
        with pytest.raises(BaselineError, match="no metrics"):
            baselines.load(path)


class TestTheCorpusCanDrift:
    def test_a_changed_generator_moves_the_digest(self, small: Corpus):
        # What `dslm check` compares. If this ever stopped being true the drift
        # check would pass forever.
        from dslm.corpus.synth import Plan, generate

        other = generate(Plan(name=SMALL.name, records=SMALL.records, seed=SMALL.seed + 1))
        assert other.digest() != small.digest()

    def test_an_edited_record_moves_the_digest(self, small: Corpus):
        edited = build_corpus(
            [
                record.model_copy(update={"text": record.text + " and one more thing"})
                if index == 0
                else record
                for index, record in enumerate(small.records)
            ]
        )
        assert edited.digest() != small.digest()
