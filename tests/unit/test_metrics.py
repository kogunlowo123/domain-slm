"""Metrics, and the significance test that keeps a claim honest.

The class that earns its place is :class:`TestMcNemar`. On this project's own
corpus the two models differ by two tenths of a point, and without a test of
significance the repository would be reporting noise as a result.
"""

from __future__ import annotations

import numpy as np
import pytest

from dslm.errors import EvaluationError
from dslm.evaluate.metrics import (
    confusion,
    evaluate,
    label_noise_ceiling,
    mcnemar,
    most_confused,
)
from dslm.model.features import LabelSpace
from tests.conftest import make_record

pytestmark = pytest.mark.unit

SPACE = LabelSpace(labels=(21, 32, 34))


def probabilities_for(predictions: np.ndarray, *, confidence: float = 0.9) -> np.ndarray:
    rows = np.full((predictions.size, SPACE.size), (1.0 - confidence) / (SPACE.size - 1))
    rows[np.arange(predictions.size), predictions] = confidence
    return rows


class TestScoring:
    def test_a_perfect_model_scores_one(self):
        targets = np.array([0, 1, 2, 0], dtype=np.int32)
        scored = evaluate("m", targets, probabilities_for(targets), targets, SPACE)
        assert scored.accuracy == 1.0
        assert scored.macro_f1 == 1.0

    def test_macro_f1_punishes_giving_up_on_a_class(self):
        # Accuracy is dominated by the largest classes; a model that abandons a
        # small one can still look fine. Macro F1 is where that becomes visible.
        targets = np.array([0] * 8 + [1, 2], dtype=np.int32)
        always_zero = np.zeros(10, dtype=np.int32)
        scored = evaluate("m", always_zero, probabilities_for(always_zero), targets, SPACE)
        assert scored.accuracy == pytest.approx(0.8)
        assert scored.macro_f1 < 0.4

    def test_the_worst_class_is_reported(self):
        targets = np.array([0] * 8 + [1, 2], dtype=np.int32)
        always_zero = np.zeros(10, dtype=np.int32)
        scored = evaluate("m", always_zero, probabilities_for(always_zero), targets, SPACE)
        assert scored.worst_class is not None
        assert scored.worst_class.f1 == 0.0

    def test_support_is_recorded_per_class(self):
        targets = np.array([0, 0, 1, 2], dtype=np.int32)
        scored = evaluate("m", targets, probabilities_for(targets), targets, SPACE)
        assert [item.support for item in scored.per_class] == [2, 1, 1]

    def test_scoring_nothing_is_refused(self):
        # Reporting 1.0 over zero records is the single most misleading thing
        # this code could do.
        empty = np.array([], dtype=np.int32)
        with pytest.raises(EvaluationError, match="no records"):
            evaluate("m", empty, np.zeros((0, 3)), empty, SPACE)

    def test_a_length_mismatch_is_refused(self):
        with pytest.raises(EvaluationError, match="prediction"):
            evaluate(
                "m",
                np.array([0, 1], dtype=np.int32),
                probabilities_for(np.array([0, 1], dtype=np.int32)),
                np.array([0], dtype=np.int32),
                SPACE,
            )

    def test_confidence_is_reported_alongside_accuracy(self):
        # A model right 92% of the time and 99% confident is differently wrong
        # from one right 92% of the time that knows it.
        targets = np.array([0, 1, 2], dtype=np.int32)
        scored = evaluate("m", targets, probabilities_for(targets, confidence=0.7), targets, SPACE)
        assert scored.mean_confidence == pytest.approx(0.7)


class TestMcNemar:
    def _targets(self) -> np.ndarray:
        return np.zeros(100, dtype=np.int32)

    def test_identical_models_give_no_evidence(self):
        # p = 1.0, and the wording says "indistinguishable" rather than "the
        # same" — which is a different claim and a false one.
        targets = self._targets()
        result = mcnemar("a", targets, "b", targets, targets)
        assert result.p_value == 1.0
        assert not result.significant
        assert "indistinguishable" in result.summary()

    def test_a_large_consistent_difference_is_significant(self):
        targets = self._targets()
        good = targets.copy()
        bad = targets.copy()
        bad[:20] = 1  # twenty records only the first model gets right
        result = mcnemar("good", good, "bad", bad, targets)
        assert result.significant
        assert result.left_only == 20
        assert result.right_only == 0

    def test_a_small_difference_is_not(self):
        # Two records out of a hundred is the case the exact test exists for:
        # the chi-squared approximation needs about 25 discordant pairs.
        targets = self._targets()
        left = targets.copy()
        right = targets.copy()
        left[0] = 1
        right[1] = 1
        result = mcnemar("a", left, "b", right, targets)
        assert result.discordant == 2
        assert not result.significant

    def test_only_disagreements_count(self):
        # Records both models get wrong carry no information about which is
        # better, and a test that counted them would have less power.
        targets = self._targets()
        both_wrong = targets.copy()
        both_wrong[:50] = 1
        result = mcnemar("a", both_wrong, "b", both_wrong, targets)
        assert result.discordant == 0

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(EvaluationError, match="different lengths"):
            mcnemar(
                "a",
                np.zeros(3, dtype=np.int32),
                "b",
                np.zeros(2, dtype=np.int32),
                np.zeros(3, dtype=np.int32),
            )

    def test_the_summary_names_the_better_model_when_there_is_one(self):
        targets = self._targets()
        good = targets.copy()
        bad = targets.copy()
        bad[:20] = 1
        assert mcnemar("good", good, "bad", bad, targets).summary().startswith("good beats bad")


class TestConfusion:
    def test_the_matrix_counts_true_by_predicted(self):
        targets = np.array([0, 0, 1], dtype=np.int32)
        predictions = np.array([0, 1, 1], dtype=np.int32)
        assert confusion(predictions, targets, SPACE) == [[1, 1, 0], [0, 1, 0], [0, 0, 0]]

    def test_most_confused_reports_labels_not_indices(self):
        # Twelve classes is a 144-cell table nobody reads; the five cells that
        # matter are the off-diagonal ones with the largest counts.
        targets = np.array([0, 0, 0, 1], dtype=np.int32)
        predictions = np.array([1, 1, 1, 1], dtype=np.int32)
        assert most_confused(predictions, targets, SPACE) == [(21, 32, 3)]

    def test_a_perfect_model_has_nothing_confused(self):
        targets = np.array([0, 1, 2], dtype=np.int32)
        assert most_confused(targets, targets, SPACE) == []


class TestCeiling:
    def test_it_uses_the_generator_s_own_record_of_what_it_corrupted(self):
        records = [
            make_record(record_id="a"),
            make_record(record_id="b", true_label="32"),
        ]
        assert label_noise_ceiling(records) == pytest.approx(0.5)

    def test_a_corpus_that_does_not_claim_to_know_gets_no_ceiling(self):
        # Every real corpus is in this position, and a ceiling asserted without
        # evidence is worse than none.
        assert label_noise_ceiling([make_record()]) is None
