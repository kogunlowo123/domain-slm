"""Features, the linear control, and the neural classifier.

Two claims get most of the attention here, because both are load-bearing and
both are easy to get quietly wrong: that the model actually learns (a model that
predicts the majority class scores 11% and a model with a bug can look
plausible), and that a run is reproducible on one machine even though the
project has measured that it is not bit-identical across two.
"""

from __future__ import annotations

import numpy as np
import pytest

from dslm.errors import ModelError
from dslm.model import features, linear, neural
from dslm.model.features import Encoded, LabelSpace
from dslm.tokenizer import bpe
from tests.conftest import make_record

pytestmark = pytest.mark.unit


class TestLabelSpace:
    def test_labels_are_sorted_not_first_seen(self):
        # First-seen would make the mapping depend on the order of the file, so
        # shuffling a corpus would renumber the classes and every stored model
        # would start predicting the wrong chapters.
        space = features.label_space(
            [make_record(record_id="a", label=34), make_record(record_id="b", label=21)]
        )
        assert space.labels == (21, 34)

    def test_an_unseen_label_is_refused_rather_than_mapped_to_the_nearest(self):
        space = LabelSpace(labels=(21, 32))
        with pytest.raises(ModelError, match="not in this model's label space"):
            space.index_of(49)

    def test_an_empty_corpus_has_no_label_space(self):
        with pytest.raises(ModelError, match="no labels"):
            features.label_space([])

    def test_the_digest_changes_with_the_labels(self):
        assert LabelSpace(labels=(21, 32)).digest() != LabelSpace(labels=(21, 33)).digest()


class TestEncoding:
    @staticmethod
    def _tokenizer() -> bpe.Tokenizer:
        tokenizer, _ = bpe.train(
            ["a valve leaked at the gate today and was replaced"] * 8, vocab_size=350
        )
        return tokenizer

    def test_padding_fills_the_tail(self):
        tokenizer = self._tokenizer()
        space = LabelSpace(labels=(32,))
        encoded = features.encode([make_record(text="short one")], tokenizer, space, max_tokens=32)
        padding = tokenizer.vocab[bpe.PADDING]
        assert encoded.sequences[0, -1] == padding
        assert encoded.lengths[0] < 32

    def test_truncation_is_counted_rather_than_hidden(self):
        # A model quietly reading the first 4 tokens of a 40-token record is a
        # model with a data bug and a plausible-looking accuracy.
        tokenizer = self._tokenizer()
        space = LabelSpace(labels=(32,))
        encoded = features.encode(
            [make_record(text="a valve leaked at the gate today and was replaced")],
            tokenizer,
            space,
            max_tokens=4,
        )
        assert encoded.truncated == 1

    def test_counts_accumulate_repeated_tokens(self):
        # `np.add.at`, not fancy-index assignment: a token appearing twice must
        # count twice, and plain assignment keeps only the last.
        tokenizer = self._tokenizer()
        space = LabelSpace(labels=(32,))
        encoded = features.encode([make_record(text="valve valve valve")], tokenizer, space)
        assert encoded.counts(tokenizer.vocab_size).sum() == encoded.lengths[0]

    def test_encoding_nothing_is_refused(self):
        tokenizer = self._tokenizer()
        with pytest.raises(ModelError, match="no records"):
            features.encode([], tokenizer, LabelSpace(labels=(32,)))

    def test_zero_tokens_per_record_is_refused(self):
        tokenizer = self._tokenizer()
        with pytest.raises(ModelError, match="max_tokens"):
            features.encode([make_record()], tokenizer, LabelSpace(labels=(32,)), max_tokens=0)


class TestNaiveBayes:
    def test_it_learns_something_far_above_chance(self, encoded, tokenizer):
        model = linear.train(encoded["train"], tokenizer.vocab_size)
        predictions = model.predict(encoded["test"].counts(tokenizer.vocab_size))
        accuracy = neural.accuracy(predictions, encoded["test"].targets)
        majority = max(np.bincount(encoded["test"].targets)) / encoded["test"].records
        assert accuracy > majority + 0.3

    def test_probabilities_are_a_distribution(self, encoded, tokenizer):
        model = linear.train(encoded["train"], tokenizer.vocab_size)
        probabilities = model.predict_proba(encoded["test"].counts(tokenizer.vocab_size))
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert np.isfinite(probabilities).all()

    def test_probabilities_do_not_overflow_on_a_long_record(self, encoded, tokenizer):
        # These are sums of hundreds of log probabilities. Without subtracting
        # the row maximum, exp overflows and every confidence becomes nan.
        model = linear.train(encoded["train"], tokenizer.vocab_size)
        counts = encoded["train"].counts(tokenizer.vocab_size) * 500
        assert np.isfinite(model.predict_proba(counts)).all()

    def test_a_feature_matrix_of_the_wrong_width_is_refused(self, encoded, tokenizer):
        model = linear.train(encoded["train"], tokenizer.vocab_size)
        with pytest.raises(ModelError, match="columns"):
            model.scores(np.zeros((2, 7)))

    def test_zero_smoothing_is_refused(self, encoded, tokenizer):
        # One unseen token would otherwise veto the class outright.
        with pytest.raises(ModelError, match="smoothing"):
            linear.train(encoded["train"], tokenizer.vocab_size, alpha=0.0)

    def test_a_class_with_no_examples_is_refused(self, tokenizer):
        # It gets a prior of zero and is never predicted, so every record of
        # that class is wrong and the model cannot say why.
        space = LabelSpace(labels=(21, 32))
        encoded = features.encode(
            [make_record(record_id="a", label=21, text="a valve leaked at the gate")],
            tokenizer,
            space,
        )
        with pytest.raises(ModelError, match="no training record"):
            linear.train(encoded, tokenizer.vocab_size)


def separable(classes: int = 4, per_class: int = 60) -> Encoded:
    """A task a working classifier must solve: one distinctive token per class.

    Deliberately not the domain corpus. Whether 268 records are enough to
    learn twelve chapters is a question about sample size, and answering it
    here would make this test fail for a reason that has nothing to do with
    the gradient. The realistic accuracy claim belongs to the integration
    layer, where the whole corpus is available.
    """
    rows = classes * per_class
    sequences = np.zeros((rows, 6), dtype=np.int32)
    targets = np.zeros(rows, dtype=np.int32)
    rng = np.random.default_rng(3)
    for row in range(rows):
        label = row % classes
        # One token that identifies the class, plus shared filler.
        sequences[row] = [1 + label, *rng.integers(10, 20, size=5)]
        targets[row] = label
    return Encoded(
        sequences=sequences,
        lengths=np.full(rows, 6, dtype=np.int32),
        targets=targets,
        truncated=0,
        empty=0,
        label_space=LabelSpace(labels=tuple(21 + index for index in range(classes))),
        max_tokens=6,
    )


class TestNeural:
    def test_it_learns_a_task_it_must_be_able_to_learn(self):
        data = separable()
        model, _ = neural.train(data, epochs=40, weight_decay=1e-5, vocab_size=32)
        accuracy = neural.accuracy(model.predict(data.sequences, data.lengths), data.targets)
        assert accuracy > 0.95

    def test_the_loss_falls(self, encoded, tokenizer):
        _, history = neural.train(
            encoded["train"],
            epochs=6,
            weight_decay=1e-5,
            vocab_size=tokenizer.vocab_size,
            padding_id=tokenizer.vocab[bpe.PADDING],
        )
        assert history.losses[-1] < history.losses[0]

    def test_the_same_seed_gives_the_same_model_on_this_machine(self, encoded, tokenizer):
        # Same-platform reproducibility is exact. Cross-platform is not, and the
        # project measured that rather than assuming either way — see ADR-003.
        kwargs = {
            "epochs": 4,
            "vocab_size": tokenizer.vocab_size,
            "padding_id": tokenizer.vocab[bpe.PADDING],
            "seed": 99,
        }
        one, _ = neural.train(encoded["train"], **kwargs)
        two, _ = neural.train(encoded["train"], **kwargs)
        assert one.weights_digest() == two.weights_digest()

    def test_a_different_seed_gives_a_different_model(self, encoded, tokenizer):
        kwargs = {
            "epochs": 4,
            "vocab_size": tokenizer.vocab_size,
            "padding_id": tokenizer.vocab[bpe.PADDING],
        }
        one, _ = neural.train(encoded["train"], seed=1, **kwargs)
        two, _ = neural.train(encoded["train"], seed=2, **kwargs)
        assert one.weights_digest() != two.weights_digest()

    def test_early_stopping_keeps_the_best_dev_epoch(self, encoded, tokenizer):
        _, history = neural.train(
            encoded["train"],
            dev=encoded["dev"],
            epochs=8,
            vocab_size=tokenizer.vocab_size,
            padding_id=tokenizer.vocab[bpe.PADDING],
        )
        assert history.best_epoch >= 1
        assert history.dev_accuracies[history.best_epoch - 1] == max(history.dev_accuracies)

    def test_without_a_dev_split_nothing_is_selected_on(self, encoded, tokenizer):
        # Selecting on training accuracy would choose the most overfitted model
        # available, so it deliberately does not select at all.
        _, history = neural.train(
            encoded["train"],
            epochs=5,
            vocab_size=tokenizer.vocab_size,
            padding_id=tokenizer.vocab[bpe.PADDING],
        )
        assert history.dev_accuracies == ()
        assert history.best_epoch == 5

    def test_probabilities_are_a_distribution(self, encoded, tokenizer):
        model, _ = neural.train(
            encoded["train"],
            epochs=3,
            vocab_size=tokenizer.vocab_size,
            padding_id=tokenizer.vocab[bpe.PADDING],
        )
        probabilities = model.predict_proba(encoded["test"].sequences, encoded["test"].lengths)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    @pytest.mark.parametrize(("epochs", "batch"), [(0, 32), (4, 0)])
    def test_degenerate_hyperparameters_are_refused(self, encoded, epochs, batch):
        # Zero epochs produces an untrained model that still reports a number.
        with pytest.raises(ModelError, match="at least 1"):
            neural.train(encoded["train"], epochs=epochs, batch=batch)

    def test_accuracy_over_nothing_is_refused(self):
        with pytest.raises(ModelError, match="zero records"):
            neural.accuracy(np.array([], dtype=np.int32), np.array([], dtype=np.int32))


class TestPooling:
    def test_padding_does_not_contribute_to_the_mean(self, encoded, tokenizer):
        # Otherwise a short record is mostly an average of the padding
        # embedding, and the model's view of it depends on max_tokens.
        model, _ = neural.train(
            encoded["train"],
            epochs=2,
            vocab_size=tokenizer.vocab_size,
            padding_id=tokenizer.vocab[bpe.PADDING],
        )
        short = encoded["test"].sequences[:1, :8].copy()
        lengths = np.minimum(encoded["test"].lengths[:1], 8)
        wide = np.full((1, 40), tokenizer.vocab[bpe.PADDING], dtype=np.int32)
        wide[0, : short.shape[1]] = short[0]
        assert np.allclose(model.logits(short, lengths), model.logits(wide, lengths), atol=1e-9)
