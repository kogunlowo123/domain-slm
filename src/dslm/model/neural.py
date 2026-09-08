"""The model: token embeddings, mean pooling, one hidden layer, in NumPy.

## What it is

Records in, ATA chapter out. Each token is embedded, the embeddings of a record
are averaged, and the average goes through one hidden layer to a softmax over
classes. Roughly a quarter of a million parameters at the shipped size, about a
megabyte on disk, and it trains on this corpus in a few seconds on one CPU core.

Mean pooling rather than attention, deliberately. A work order is twenty words
long and the task is essentially "which vocabulary is this?" — a bag of learned
embeddings has the capacity that task needs, and a transformer here would be an
architecture chosen to look impressive rather than to fit the problem. When the
linear control is within a point of the neural model, adding attention would be
answering the wrong question. See ADR-002.

## Why NumPy and not a framework

Three reasons, in order of how much they matter.

The reproducibility question — *can two machines train the same model?* — is the
one this repository is actually about, and taking on a framework would answer it
by delegation. Here every floating-point operation is visible in this file.

A deep-learning framework is a 900 MB dependency for a model with three weight
matrices, and the gap between "what the maths says" and "what the code does"
becomes a stack somebody has to trust.

And the whole training loop is thirty lines of the same algebra the derivation
uses, which is the point of a repository meant to be read.

## Reproducibility, measured rather than assumed

Initialisation is **uniform**, not normal. That is not a stylistic choice.
``standard_normal`` uses a ziggurat whose rare tail branch calls ``log``, and
libm's ``log`` differs by one unit in the last place between platforms: measured
here, exactly one value in 32,768 differed between Windows and Linux, and that
single bit then propagates through every subsequent weight. Uniform draws are
integer-to-float only and are bit-identical everywhere.

That fixes initialisation but not training. Measured with the pinned NumPy on
the same two platforms, after thirty steps: **the loss is bit-identical, and
1,344 of 55,872 weights differ, by at most 2.4e-14 relative.** Accumulation
order inside a matrix multiply is a property of the BLAS build, and no seed
controls it.

So the artefacts split into two kinds, and they are gated differently:

- the corpus, the split and the tokenizer are integers and strings, and are
  checked by **digest**;
- the model's weights are floats, and are checked by **metrics within a stated
  tolerance**.

Claiming a byte-exact model across platforms would be a claim this project
measured and found to be false. ADR-003 records it.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from dslm.errors import ModelError
from dslm.model.features import Encoded, LabelSpace

#: Embedding width. Small: the vocabulary is a few thousand tokens over a narrow
#: domain, and the useful distinctions are coarse.
DEFAULT_EMBEDDING = 48
#: Hidden width.
DEFAULT_HIDDEN = 96
#: Passes over the training set. Twenty, with early stopping on dev: the run
#: usually peaks well before the end, and letting it continue costs accuracy
#: rather than gaining it (see the sweep in docs/model.md).
DEFAULT_EPOCHS = 20
#: Records per gradient step.
DEFAULT_BATCH = 64
#: Learning rate for plain SGD. No momentum, no Adam: one more moving part to
#: document and to make reproducible, for a model that converges in seconds.
DEFAULT_LEARNING_RATE = 0.5
#: L2 penalty.
#:
#: 1e-3, chosen on the dev split and worth two full points of accuracy. At
#: 1e-5 the model reaches 100% training accuracy and 83.5% dev — it has the
#: capacity to memorise the corpus's 4% label noise, and given the chance it
#: does. That is the whole argument for the penalty, and it is measured
#: rather than assumed: docs/model.md has the sweep.
DEFAULT_WEIGHT_DECAY = 1e-3
#: The seed. Recorded in the model file, so a run can be repeated exactly on the
#: same platform and to within tolerance on another.
DEFAULT_SEED = 20260908


def _uniform(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    """Glorot-uniform initialisation, drawn flat and reshaped.

    Uniform rather than normal for reproducibility (see the module docstring);
    drawn as one flat vector and reshaped so that the number of variates
    consumed depends only on the parameter count, never on the shape.
    """
    limit = float(np.sqrt(6.0 / (fan_in + fan_out)))
    flat = rng.random(fan_in * fan_out) * 2.0 - 1.0
    return (flat * limit).reshape(fan_in, fan_out)


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    """What happened during training, per epoch."""

    losses: tuple[float, ...]
    accuracies: tuple[float, ...]
    #: Accuracy on the dev split, when one was supplied. Empty otherwise, and
    #: the emptiness is visible rather than filled with the training numbers.
    dev_accuracies: tuple[float, ...]
    epochs: int
    seconds: float
    #: The epoch whose parameters were kept, 1-based. Equal to ``epochs``
    #: when no dev split was supplied, because without one there is nothing
    #: to select on and selecting on training accuracy would choose the most
    #: overfitted model available.
    best_epoch: int = 0

    def summary(self) -> str:
        """One line, for a terminal."""
        final = self.losses[-1] if self.losses else float("nan")
        dev = f", dev {self.dev_accuracies[-1]:.3f}" if self.dev_accuracies else ""
        train = self.accuracies[-1] if self.accuracies else float("nan")
        stopped = (
            f", kept epoch {self.best_epoch}"
            if self.dev_accuracies and self.best_epoch != self.epochs
            else ""
        )
        return (
            f"{self.epochs} epoch(s) in {self.seconds:.1f}s: "
            f"loss {final:.4f}, train {train:.3f}{dev}{stopped}"
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialise for a report."""
        return {
            "epochs": self.epochs,
            "seconds": round(self.seconds, 3),
            "loss": [round(value, 6) for value in self.losses],
            "train_accuracy": [round(value, 6) for value in self.accuracies],
            "dev_accuracy": [round(value, 6) for value in self.dev_accuracies],
            "best_epoch": self.best_epoch,
        }


@dataclass(frozen=True, slots=True)
class Classifier:
    """A trained embedding-and-MLP classifier."""

    #: (vocab, embedding)
    embedding: np.ndarray
    #: (embedding, hidden), (hidden,)
    hidden_weight: np.ndarray
    hidden_bias: np.ndarray
    #: (hidden, classes), (classes,)
    output_weight: np.ndarray
    output_bias: np.ndarray
    label_space: LabelSpace
    padding_id: int
    seed: int
    #: The tokenizer this model was trained against. Checked before scoring:
    #: a model run against a different vocabulary produces confident nonsense.
    tokenizer_digest: str = ""

    @property
    def parameters(self) -> int:
        """How many numbers this model consists of."""
        return int(
            self.embedding.size
            + self.hidden_weight.size
            + self.hidden_bias.size
            + self.output_weight.size
            + self.output_bias.size
        )

    @property
    def vocab_size(self) -> int:
        """The vocabulary this model was trained against."""
        return int(self.embedding.shape[0])

    def weights_digest(self) -> str:
        """A digest over the weights.

        Useful for *same-platform* reproducibility and for detecting a corrupted
        file. **Not** a cross-platform check: the module docstring records the
        measurement showing that trained weights differ in their last bits
        between platforms, so an equality assertion on this would fail in CI for
        a reason that has nothing to do with the code.
        """
        material = b"".join(
            np.ascontiguousarray(array).tobytes()
            for array in (
                self.embedding,
                self.hidden_weight,
                self.hidden_bias,
                self.output_weight,
                self.output_bias,
            )
        )
        return "sha256:" + hashlib.sha256(material).hexdigest()

    def _pooled(self, sequences: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Mean of the embeddings of the real tokens in each record."""
        summed = self.embedding[sequences].sum(axis=1)
        # A record cannot be empty in practice — the byte fallback guarantees at
        # least one token — but dividing by a zero length would give nan and a
        # nan propagates to every prediction, so the floor is not optional.
        divisor = np.maximum(lengths, 1).astype(np.float64)[:, None]
        padding = (
            self.embedding[self.padding_id]
            * (sequences.shape[1] - np.minimum(lengths, sequences.shape[1]))[:, None]
        )
        return np.asarray((summed - padding) / divisor)

    def logits(self, sequences: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Return (records, classes) unnormalised scores."""
        if sequences.shape[0] != lengths.shape[0]:
            raise ModelError(
                "the sequence and length arrays disagree about how many records there are.",
                remedy="Both come from `features.encode`; do not build them separately.",
            )
        hidden = np.tanh(self._pooled(sequences, lengths) @ self.hidden_weight + self.hidden_bias)
        return np.asarray(hidden @ self.output_weight + self.output_bias)

    def predict(self, sequences: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Return (records,) predicted class indices."""
        return np.asarray(self.logits(sequences, lengths).argmax(axis=1), dtype=np.int32)

    def predict_proba(self, sequences: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Return (records, classes) softmax probabilities."""
        return _softmax(self.logits(sequences, lengths))

    def as_dict(self) -> dict[str, Any]:
        """Serialise the metadata, not the weights."""
        return {
            "kind": "neural",
            "classes": self.label_space.size,
            "vocab_size": self.vocab_size,
            "embedding": int(self.embedding.shape[1]),
            "hidden": int(self.hidden_weight.shape[1]),
            "parameters": self.parameters,
            "seed": self.seed,
            "weights_digest": self.weights_digest(),
            "tokenizer_digest": self.tokenizer_digest,
            "label_space": self.label_space.as_dict(),
        }


def _softmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax, with the row maximum subtracted first.

    The subtraction is not an optimisation. Without it ``exp`` overflows on any
    logit above about 710, the result is ``inf``, and ``inf / inf`` is ``nan`` —
    which propagates silently into every downstream confidence.
    """
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return np.asarray(exponentiated / exponentiated.sum(axis=1, keepdims=True))


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """The fraction correct."""
    if not targets.size:
        raise ModelError(
            "accuracy over zero records is not a number.",
            remedy="An empty evaluation set should be reported as such, never as 1.0.",
        )
    return float((predictions == targets).mean())


def train(  # noqa: PLR0913, PLR0915 - hyperparameters are the interface; the loop is the point
    encoded: Encoded,
    *,
    dev: Encoded | None = None,
    embedding: int = DEFAULT_EMBEDDING,
    hidden: int = DEFAULT_HIDDEN,
    epochs: int = DEFAULT_EPOCHS,
    batch: int = DEFAULT_BATCH,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = DEFAULT_SEED,
    vocab_size: int | None = None,
    padding_id: int = 0,
    tokenizer_digest: str = "",
    on_epoch: Callable[[int, float, float], None] | None = None,
) -> tuple[Classifier, TrainingHistory]:
    """Train the classifier by minibatch stochastic gradient descent.

    Plain SGD, cross-entropy, one hidden layer with ``tanh``. The backward pass
    below is written out rather than derived by a framework, and reads in the
    same order as the forward pass reversed, which is the only way to check it
    against the algebra.
    """
    if epochs < 1 or batch < 1:
        raise ModelError(
            f"epochs={epochs} and batch={batch} must both be at least 1.",
            remedy="Zero epochs produces an untrained model that still reports a number.",
        )
    vocabulary = vocab_size if vocab_size is not None else int(encoded.sequences.max()) + 1
    classes = encoded.label_space.size

    rng = np.random.default_rng(seed)
    E = _uniform(rng, vocabulary, embedding)
    W1 = _uniform(rng, embedding, hidden)
    b1 = np.zeros(hidden, dtype=np.float64)
    W2 = _uniform(rng, hidden, classes)
    b2 = np.zeros(classes, dtype=np.float64)

    sequences = encoded.sequences
    lengths = encoded.lengths
    targets = encoded.targets
    records = encoded.records
    divisor = np.maximum(lengths, 1).astype(np.float64)[:, None]
    width = sequences.shape[1]
    # Which positions hold a real token. Padding must not contribute to the mean
    # or a short record is mostly an average of the padding embedding.
    mask = (np.arange(width)[None, :] < lengths[:, None]).astype(np.float64)

    losses: list[float] = []
    accuracies: list[float] = []
    dev_accuracies: list[float] = []
    # Early stopping keeps the parameters from the best dev epoch rather than
    # the last. Without it this model's dev accuracy *falls* over the second
    # half of a run while training accuracy climbs to 1.0 — the capacity goes
    # into memorising the corpus's label noise. Selecting on dev is the
    # standard remedy; selecting on *test* would be the standard mistake, and
    # this function is never given the test split.
    best: tuple[float, int, tuple[np.ndarray, ...]] | None = None
    started = time.perf_counter()

    for epoch in range(epochs):
        order = rng.permutation(records)
        total_loss = 0.0
        correct = 0

        for start in range(0, records, batch):
            rows = order[start : start + batch]
            size = len(rows)
            ids = sequences[rows]
            weights = mask[rows][:, :, None]

            pooled = (E[ids] * weights).sum(axis=1) / divisor[rows]
            pre = pooled @ W1 + b1
            h = np.tanh(pre)
            logits = h @ W2 + b2
            probabilities = _softmax(logits)

            picked = probabilities[np.arange(size), targets[rows]]
            total_loss += float(-np.log(np.maximum(picked, 1e-12)).sum())
            correct += int((logits.argmax(axis=1) == targets[rows]).sum())

            grad_logits = probabilities
            grad_logits[np.arange(size), targets[rows]] -= 1.0
            grad_logits /= size

            grad_W2 = h.T @ grad_logits + weight_decay * W2
            grad_b2 = grad_logits.sum(axis=0)
            grad_h = (grad_logits @ W2.T) * (1.0 - h * h)
            grad_W1 = pooled.T @ grad_h + weight_decay * W1
            grad_b1 = grad_h.sum(axis=0)
            grad_pooled = grad_h @ W1.T

            # Scatter the pooled gradient back to the tokens that produced it.
            # `np.add.at` rather than fancy-index assignment: a token appearing
            # twice in one record must accumulate both contributions, and plain
            # assignment keeps only the last.
            per_token = (grad_pooled / divisor[rows])[:, None, :] * weights
            grad_E = np.zeros_like(E)
            np.add.at(grad_E, ids, per_token)

            E -= learning_rate * grad_E
            W1 -= learning_rate * grad_W1
            b1 -= learning_rate * grad_b1
            W2 -= learning_rate * grad_W2
            b2 -= learning_rate * grad_b2

        losses.append(total_loss / records)
        accuracies.append(correct / records)

        model = Classifier(
            embedding=E,
            hidden_weight=W1,
            hidden_bias=b1,
            output_weight=W2,
            output_bias=b2,
            label_space=encoded.label_space,
            padding_id=padding_id,
            seed=seed,
            tokenizer_digest=tokenizer_digest,
        )
        if dev is not None:
            score = accuracy(model.predict(dev.sequences, dev.lengths), dev.targets)
            dev_accuracies.append(score)
            if best is None or score > best[0]:
                best = (
                    score,
                    epoch + 1,
                    (E.copy(), W1.copy(), b1.copy(), W2.copy(), b2.copy()),
                )
        if on_epoch is not None:
            on_epoch(epoch + 1, losses[-1], dev_accuracies[-1] if dev_accuracies else -1.0)

    if best is not None:
        E, W1, b1, W2, b2 = best[2]
        model = Classifier(
            embedding=E,
            hidden_weight=W1,
            hidden_bias=b1,
            output_weight=W2,
            output_bias=b2,
            label_space=encoded.label_space,
            padding_id=padding_id,
            seed=seed,
            tokenizer_digest=tokenizer_digest,
        )

    history = TrainingHistory(
        losses=tuple(losses),
        accuracies=tuple(accuracies),
        dev_accuracies=tuple(dev_accuracies),
        epochs=epochs,
        seconds=time.perf_counter() - started,
        best_epoch=best[1] if best is not None else epochs,
    )
    return model, history
