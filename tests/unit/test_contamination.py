"""The contamination gate.

The module the repository exists for, so these are the tests that matter most.
Three groups: the arithmetic, the calibration, and the three independent ways
:func:`enforce` refuses.
"""

from __future__ import annotations

import pytest

from dslm.corpus.contamination import (
    DEFAULT_QUANTILE,
    DEFAULT_THRESHOLD,
    NGRAM_WORDS,
    build_index,
    calibrate,
    containment,
    contamination_report,
    enforce,
    ngrams,
)
from dslm.corpus.record import Corpus, build_corpus
from dslm.errors import ContaminationError
from tests.conftest import make_corpus, make_record

pytestmark = pytest.mark.unit

LONG = " ".join(f"word{index}" for index in range(30))


class TestNgrams:
    def test_a_record_shorter_than_the_width_yields_nothing(self):
        # Empty, not a fallback to the whole string: a caller has to be able to
        # tell "no overlap" from "not checkable".
        assert ngrams("only four words here", width=13) == frozenset()

    def test_ngrams_overlap(self):
        assert ngrams("a b c d", width=3) == frozenset({"a b c", "b c d"})

    def test_containment_is_none_when_the_record_is_too_short(self):
        # None, never zero. A caller treating this as zero reports a clean
        # result for a record that was never examined.
        assert containment("too short", frozenset(), width=13) is None

    def test_containment_is_the_fraction_present(self):
        index = ngrams("a b c d", width=3)
        assert containment("a b c d", index, width=3) == 1.0
        assert containment("a b c x", index, width=3) == pytest.approx(0.5)

    def test_the_index_is_one_set_for_the_whole_side(self):
        # The model saw all of training. A record whose phrases are spread over
        # three training records is exactly as contaminated as one matching one.
        records = make_corpus(["a b c d e f", "g h i j k l"])
        index = build_index(records, width=3)
        assert "a b c" in index
        assert "g h i" in index


class TestCalibration:
    def test_the_threshold_comes_from_the_null_distribution(
        self, small: Corpus, small_null: Corpus
    ):
        calibration = calibrate(small, small_null, width=5)
        assert 0.0 < calibration.threshold <= 1.0
        assert calibration.scored > 0

    def test_it_reports_what_the_published_rule_would_have_said(
        self, small: Corpus, small_null: Corpus
    ):
        # The number that makes the case for calibrating at all.
        calibration = calibrate(small, small_null, width=5)
        assert 0.0 <= calibration.uncalibrated_null_rate <= 1.0

    def test_a_null_that_overlaps_training_is_refused(self):
        shared = make_corpus([LONG])
        with pytest.raises(ContaminationError, match="shares 1 record"):
            calibrate(shared, shared)

    def test_it_can_be_repaired_explicitly_and_says_how_much(self):
        # Off by default and asked for, because a null corpus that silently
        # shrinks is one nobody would think to question.
        train = make_corpus([LONG, LONG.replace("word0", "other")])
        null = build_corpus(
            [
                make_record(record_id="n1", text=LONG),
                make_record(record_id="n2", text=" ".join(f"z{i}" for i in range(30))),
            ]
        )
        calibration = calibrate(train, null, drop_shared=True)
        assert calibration.dropped == 1
        assert calibration.scored == 1

    def test_a_null_that_is_entirely_shared_is_refused_even_with_drop_shared(self):
        shared = make_corpus([LONG])
        with pytest.raises(ContaminationError, match="every record"):
            calibrate(shared, shared, drop_shared=True)

    @pytest.mark.parametrize("quantile", [0.0, 1.0, -0.5, 2.0])
    def test_an_impossible_quantile_is_refused(
        self, small: Corpus, small_null: Corpus, quantile: float
    ):
        with pytest.raises(ContaminationError, match="quantile"):
            calibrate(small, small_null, quantile=quantile)

    def test_the_default_quantile_is_high_enough_to_leave_a_clean_split_at_zero(self):
        # The reason it is 0.999 and not 0.99: a quantile is a promise about how
        # often disjoint data is called contaminated, and 1% of it is a floor
        # under every measurement the gate makes.
        assert DEFAULT_QUANTILE >= 0.999


class TestReports:
    def test_an_identical_record_is_a_hit(self):
        train = make_corpus([LONG])
        report = contamination_report(train, train, width=5)
        assert report.rate == 1.0
        assert report.hits[0].containment == 1.0

    def test_unrelated_records_are_not(self):
        train = make_corpus([LONG])
        other = make_corpus([" ".join(f"z{index}" for index in range(30))])
        assert contamination_report(train, other, width=5).rate == 0.0

    def test_records_too_short_to_check_are_counted_separately(self):
        # Not counted as clean. "The check did not apply" and "the check passed"
        # are different facts, and this is where they stay different.
        train = make_corpus([LONG])
        short = make_corpus(["four words only here"])
        report = contamination_report(train, short, width=13)
        assert report.not_checkable == 1
        assert report.evaluated == 0
        assert report.coverage == 0.0

    def test_the_rate_is_over_the_checkable_records(self):
        # Dividing by the total would let a split with poor coverage report a
        # small number for the arithmetic reason that most of it was skipped.
        train = make_corpus([LONG])
        mixed = build_corpus(
            [
                make_record(record_id="a", text=LONG),
                make_record(record_id="b", text="far too short"),
            ]
        )
        report = contamination_report(train, mixed, width=5)
        assert report.evaluated == 1
        assert report.rate == 1.0

    def test_a_calibration_supplies_both_the_threshold_and_the_width(
        self, small: Corpus, small_null: Corpus
    ):
        # They belong together: passing one without the other is how a report
        # ends up labelled calibrated while using a number from somewhere else.
        calibration = calibrate(small, small_null, width=7)
        report = contamination_report(small, small_null, calibration=calibration, width=99)
        assert report.width == 7
        assert report.threshold == calibration.threshold
        assert report.calibrated

    def test_a_report_without_a_calibration_says_so(self):
        report = contamination_report(make_corpus([LONG]), make_corpus([LONG]), width=5)
        assert not report.calibrated
        assert "UNCALIBRATED" in report.summary()


class TestEnforce:
    def _clean(self, small: Corpus, small_null: Corpus):
        calibration = calibrate(small, small_null, drop_shared=True)
        return contamination_report(small, small_null, calibration=calibration)

    def test_a_clean_calibrated_report_passes(self, small: Corpus, small_null: Corpus):
        enforce(self._clean(small, small_null))

    def test_an_uncalibrated_report_is_refused_rather_than_guessed_at(self):
        # The whole design: this would rather return "cannot decide" than a
        # verdict it cannot stand behind.
        report = contamination_report(make_corpus([LONG] * 3), make_corpus([LONG] * 3), width=5)
        with pytest.raises(ContaminationError, match="nothing that says what that means"):
            enforce(report)

    def test_the_uncalibrated_rule_can_be_opted_into_deliberately(self):
        train = make_corpus([LONG])
        other = make_corpus([" ".join(f"z{index}" for index in range(30))])
        enforce(contamination_report(train, other, width=5), allow_uncalibrated=True)

    def test_a_contaminated_split_fails_even_when_opted_in(self):
        train = make_corpus([LONG] * 3)
        with pytest.raises(ContaminationError, match="also appear in training"):
            enforce(contamination_report(train, train, width=5), allow_uncalibrated=True)

    def test_poor_coverage_fails_before_anything_else_is_considered(self):
        # A 13-gram check over eight-word records examines almost nothing and
        # reports zero contamination, which reads exactly like a clean result.
        train = make_corpus([LONG])
        short = make_corpus(["four words only here"] * 5)
        with pytest.raises(ContaminationError, match="long enough"):
            enforce(contamination_report(train, short, width=NGRAM_WORDS))

    def test_the_error_names_the_worst_offender(self):
        train = make_corpus([LONG] * 3)
        with pytest.raises(ContaminationError) as caught:
            enforce(contamination_report(train, train, width=5), allow_uncalibrated=True)
        assert "r-000000" in caught.value.remedy

    def test_the_published_threshold_is_still_the_uncalibrated_default(self):
        assert DEFAULT_THRESHOLD == 0.5
