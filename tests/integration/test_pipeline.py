"""The whole pipeline, and the claims the repository actually makes about it.

The unit layer checks that each piece works. This checks that the pieces, wired
together on the real corpus, produce the numbers the documentation quotes — so
that a claim in the README cannot quietly stop being true.
"""

from __future__ import annotations

import pytest

from dslm.corpus import contamination as contam
from dslm.corpus.curate import apply_curation, find_duplicates
from dslm.corpus.record import Corpus
from dslm.corpus.split import Split, split_corpus
from dslm.corpus.synth import PLANS, generate
from dslm.evaluate.metrics import evaluate, label_noise_ceiling, mcnemar
from dslm.model import features, linear, neural
from dslm.tokenizer import bpe

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def null() -> Corpus:
    """The disjoint corpus the contamination threshold is calibrated against."""
    return generate(PLANS["holdout"])


@pytest.fixture(scope="module")
def full() -> dict[str, object]:
    """The shipped pipeline, end to end, on the shipped corpus.

    Module-scoped: it takes a few seconds and every test below asks a different
    question about the same run.
    """
    corpus = generate(PLANS["main"])
    cleaned = apply_curation(corpus, find_duplicates(corpus.records))
    split = split_corpus(cleaned, seed=1, curation=find_duplicates(cleaned.records))
    tokenizer, _ = bpe.train((record.text for record in split.train), vocab_size=1024)
    space = features.label_space(split.train)
    encoded = {
        part: features.encode(split[part], tokenizer, space) for part in ("train", "dev", "test")
    }
    bayes = linear.train(encoded["train"], tokenizer.vocab_size)
    model, history = neural.train(
        encoded["train"],
        dev=encoded["dev"],
        vocab_size=tokenizer.vocab_size,
        padding_id=tokenizer.vocab[bpe.PADDING],
        tokenizer_digest=tokenizer.digest(),
    )
    counts = encoded["test"].counts(tokenizer.vocab_size)
    predictions = {
        "naive-bayes": bayes.predict(counts),
        "neural": model.predict(encoded["test"].sequences, encoded["test"].lengths),
    }
    scored = {
        "naive-bayes": evaluate(
            "naive-bayes",
            predictions["naive-bayes"],
            bayes.predict_proba(counts),
            encoded["test"].targets,
            space,
            parameters=bayes.parameters,
        ),
        "neural": evaluate(
            "neural",
            predictions["neural"],
            model.predict_proba(encoded["test"].sequences, encoded["test"].lengths),
            encoded["test"].targets,
            space,
            parameters=model.parameters,
        ),
    }
    return {
        "corpus": corpus,
        "split": split,
        "tokenizer": tokenizer,
        "encoded": encoded,
        "scored": scored,
        "predictions": predictions,
        "history": history,
    }


class TestTheModelsAreWorthHaving:
    def test_both_beat_the_majority_class_by_a_wide_margin(self, full):
        split: Split = full["split"]
        majority = max(split.test.label_counts().values()) / len(split.test)
        for metrics in full["scored"].values():
            assert metrics.accuracy > majority + 0.5

    def test_both_land_where_the_documentation_says(self, full):
        # The claim in README.md and docs/model.md. If this fails, one of them
        # is now wrong — which is the point of pinning it.
        for metrics in full["scored"].values():
            assert 0.87 <= metrics.accuracy <= 0.95

    def test_neither_reaches_the_ceiling_the_label_noise_imposes(self, full):
        split: Split = full["split"]
        ceiling = label_noise_ceiling(split.test.records)
        assert ceiling is not None
        for metrics in full["scored"].values():
            assert metrics.accuracy < ceiling

    def test_macro_f1_is_close_to_accuracy_because_the_classes_are_balanced(self, full):
        for metrics in full["scored"].values():
            assert abs(metrics.accuracy - metrics.macro_f1) < 0.03


class TestTheControlEarnsItsPlace:
    def test_the_linear_control_is_far_smaller(self, full):
        scored = full["scored"]
        assert scored["naive-bayes"].parameters < scored["neural"].parameters

    def test_the_two_are_statistically_indistinguishable(self, full):
        # The finding the repository publishes rather than hides. If a change
        # ever makes the neural model genuinely better, this test fails and the
        # documentation has to be updated — which is the correct outcome.
        encoded = full["encoded"]
        result = mcnemar(
            "naive-bayes",
            full["predictions"]["naive-bayes"],
            "neural",
            full["predictions"]["neural"],
            encoded["test"].targets,
        )
        assert not result.significant

    def test_early_stopping_did_something(self, full):
        history = full["history"]
        assert history.dev_accuracies
        assert history.best_epoch >= 1


class TestTheTokenizer:
    def test_it_breaks_a_word_into_few_pieces(self, full):
        split: Split = full["split"]
        measured = bpe.compression(full["tokenizer"], (record.text for record in split.test))
        assert measured.tokens_per_word < 2.0
        assert measured.unknown == 0

    def test_it_was_trained_on_the_training_side_only(self, full):
        # A tokenizer fitted on the whole corpus has seen the test set's
        # vocabulary, which is a mild form of the leak this project is about.
        split: Split = full["split"]
        trained_again, _ = bpe.train((record.text for record in split.train), vocab_size=1024)
        assert trained_again.digest() == full["tokenizer"].digest()


class TestTheContaminationClaims:
    def test_the_published_rule_scores_disjoint_data_no_better_than_the_test_split(
        self, full, null: Corpus
    ):
        # The measurement the whole design rests on. Data that *cannot* be
        # contaminated is not scored lower by the published rule than data that
        # might be — so the rule's number is background, not signal.
        split: Split = full["split"]
        disjoint = contam.contamination_report(split.train, null).rate
        genuine = contam.contamination_report(split.train, split.test).rate
        assert disjoint >= genuine

    def test_the_calibrated_threshold_is_far_above_the_published_one(self, full, null: Corpus):
        split: Split = full["split"]
        calibration = contam.calibrate(split.train, null, drop_shared=True)
        assert calibration.threshold > contam.DEFAULT_THRESHOLD

    def test_the_correct_pipeline_measures_no_contamination(self, full, null: Corpus):
        split: Split = full["split"]
        calibration = contam.calibrate(split.train, null, drop_shared=True)
        report = contam.contamination_report(split.train, split.test, calibration=calibration)
        contam.enforce(report)
        assert report.rate == 0.0

    def test_the_shipped_holdout_collides_on_at_most_a_handful_of_records(self, full, null: Corpus):
        # Two corpora from one grammar with different seeds really can produce
        # the same sentence. The gate reports that rather than hiding it.
        split: Split = full["split"]
        calibration = contam.calibrate(split.train, null, drop_shared=True)
        assert calibration.dropped <= 5


class TestCuration:
    def test_every_planted_duplicate_is_found(self, full):
        # Recall over the planted duplicates, which the generator recorded. The
        # LSH banding is chosen to make this 1.0 rather than 0.95; see the
        # band/row table in dslm.corpus.curate.
        corpus: Corpus = full["corpus"]
        curation = find_duplicates(corpus.records)
        dropped = {record_id for cluster in curation.clusters for record_id in cluster.drop}
        planted = {
            record.record_id for record in corpus if record.meta.get("kind") in {"exact", "near"}
        }
        assert not (planted - dropped)

    def test_a_corpus_with_nothing_planted_is_almost_clean(self):
        # The negative control. A deduplicator that finds duplicates everywhere
        # is not a deduplicator.
        clean = generate(PLANS["clean"])
        curation = find_duplicates(clean.records)
        assert curation.dropped < 0.03 * len(clean)
