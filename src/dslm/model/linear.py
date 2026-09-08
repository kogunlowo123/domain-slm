"""Multinomial naive Bayes: the control the neural model has to beat.

Every claim that a neural model is worth having is a comparison, and most
published ones omit the other side of it. This is the other side.

Naive Bayes over token counts is about forty lines, has one hyperparameter,
trains in milliseconds, and on a well-separated text-classification task it is
genuinely hard to beat. If the neural model does not beat it, that is the
finding — and the honest thing to do is publish it rather than quietly not
measure it.

It is also the exactly-reproducible half of the pair. Training is a sum of
integer counts followed by one logarithm per parameter; there is no iterative
optimisation, so two machines produce the same model up to the last-bit
behaviour of ``log`` alone. The neural model cannot make that claim, and the
difference is documented in ADR-003 rather than glossed.

Laplace smoothing with alpha = 1. The alternative — zero counts giving zero
probability — makes a single unseen token veto a class outright, so a record
containing one novel word gets a posterior of zero for the class it obviously
belongs to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dslm.errors import ModelError
from dslm.model.features import Encoded, LabelSpace

#: Laplace smoothing. One pseudo-count per (class, token); see the module
#: docstring for why zero is not an option.
DEFAULT_ALPHA = 1.0


@dataclass(frozen=True, slots=True)
class NaiveBayes:
    """A trained multinomial naive Bayes classifier."""

    #: (classes, vocab) log P(token | class).
    log_likelihood: np.ndarray
    #: (classes,) log P(class).
    log_prior: np.ndarray
    label_space: LabelSpace
    vocab_size: int
    alpha: float

    @property
    def parameters(self) -> int:
        """How many numbers this model consists of."""
        return int(self.log_likelihood.size + self.log_prior.size)

    def scores(self, counts: np.ndarray) -> np.ndarray:
        """Return (records, classes) log posteriors, unnormalised.

        Unnormalised because the normalising constant is the same for every
        class of a record and cancels in both the argmax and the softmax. What
        it does not cancel in is a *confidence* — so :meth:`predict_proba`
        normalises and this does not, and the two are separate methods rather
        than one with a flag.
        """
        if counts.shape[1] != self.vocab_size:
            raise ModelError(
                f"the feature matrix has {counts.shape[1]} columns and the model "
                f"expects {self.vocab_size}.",
                remedy=(
                    "The tokenizer used to encode this data is not the one the model was "
                    "trained with. Check the tokenizer digest recorded alongside the model."
                ),
            )
        return np.asarray(counts @ self.log_likelihood.T + self.log_prior)

    def predict(self, counts: np.ndarray) -> np.ndarray:
        """Return (records,) predicted class indices."""
        return np.asarray(self.scores(counts).argmax(axis=1), dtype=np.int32)

    def predict_proba(self, counts: np.ndarray) -> np.ndarray:
        """Return (records, classes) posterior probabilities.

        Softmax over the log posteriors, with the row maximum subtracted first.
        Without that subtraction ``exp`` overflows for any record of more than a
        handful of tokens, because these are sums of hundreds of log
        probabilities — and the overflow arrives as ``nan``, which propagates
        into a confidence of ``nan`` and an abstention rate that is silently zero.
        """
        scores = self.scores(counts)
        scores = scores - scores.max(axis=1, keepdims=True)
        exponentiated = np.exp(scores)
        return np.asarray(exponentiated / exponentiated.sum(axis=1, keepdims=True))

    def as_dict(self) -> dict[str, Any]:
        """Serialise the metadata, not the weights."""
        return {
            "kind": "naive-bayes",
            "classes": self.label_space.size,
            "vocab_size": self.vocab_size,
            "alpha": self.alpha,
            "parameters": self.parameters,
            "label_space": self.label_space.as_dict(),
        }


def train(encoded: Encoded, vocab_size: int, *, alpha: float = DEFAULT_ALPHA) -> NaiveBayes:
    """Fit naive Bayes on *encoded*.

    One pass. Sum the token counts per class, add the smoothing, take logs.
    """
    if alpha <= 0:
        raise ModelError(
            f"the smoothing constant is {alpha}.",
            remedy=(
                "With no smoothing a single unseen token gives a class zero probability, "
                "so one novel word vetoes the class a record obviously belongs to."
            ),
        )

    counts = encoded.counts(vocab_size)
    classes = encoded.label_space.size
    per_class = np.zeros((classes, vocab_size), dtype=np.float64)
    class_counts = np.zeros(classes, dtype=np.float64)

    for index in range(classes):
        rows = encoded.targets == index
        class_counts[index] = float(rows.sum())
        if class_counts[index]:
            per_class[index] = counts[rows].sum(axis=0)

    missing = [
        encoded.label_space.label_at(index) for index in range(classes) if class_counts[index] == 0
    ]
    if missing:
        raise ModelError(
            f"no training record carries label(s) {missing}.",
            remedy=(
                "A class with no examples gets a prior of zero and is never predicted, so "
                "every record of that class is wrong and the model cannot say why. Either "
                "remove the class from the label space or supply examples of it."
            ),
        )

    smoothed = per_class + alpha
    log_likelihood = np.log(smoothed) - np.log(smoothed.sum(axis=1, keepdims=True))
    log_prior = np.log(class_counts) - np.log(class_counts.sum())

    return NaiveBayes(
        log_likelihood=log_likelihood,
        log_prior=log_prior,
        label_space=encoded.label_space,
        vocab_size=vocab_size,
        alpha=alpha,
    )
