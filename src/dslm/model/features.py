"""Turning records into arrays, once, so both models see the same thing.

Two representations, and they exist for different models:

**Counts** — a record becomes a vector of token frequencies. What the linear
control consumes. Order is discarded entirely, which is a real loss and is
exactly why it is worth having as a control: if a model that *can* use order does
not beat one that cannot, the order was not carrying the signal.

**Sequences** — a record becomes its token ids, padded to a fixed length. What
the neural model consumes.

Both are built from one tokenizer and one label ordering, in one place, because
the alternative is two code paths that agree until the day they do not and then
produce a comparison that is not a comparison.

The label ordering is derived from the *training* split and then carried
everywhere. Deriving it from whichever corpus is in hand would mean a model
trained on a split with twelve classes and evaluated on a split with eleven
silently maps class 8 to a different chapter — a bug that shows up as a plausible
accuracy rather than as an error.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from dslm.corpus.record import Corpus, Record
from dslm.errors import ModelError
from dslm.tokenizer.bpe import PADDING, Tokenizer

#: Tokens kept per record. Records longer than this are truncated, and the count
#: of truncated records is reported rather than hidden: a model quietly reading
#: the first 64 tokens of a 200-token record is a model with a data bug.
DEFAULT_MAX_TOKENS = 64


@dataclass(frozen=True, slots=True)
class LabelSpace:
    """The mapping between labels and the indices a model works in."""

    labels: tuple[int, ...]

    @property
    def size(self) -> int:
        """How many classes there are."""
        return len(self.labels)

    def index_of(self, label: int) -> int:
        """The model index for *label*."""
        try:
            return self.labels.index(label)
        except ValueError as exc:
            raise ModelError(
                f"label {label} is not in this model's label space.",
                remedy=(
                    f"The model was trained on {list(self.labels)}. A record with an "
                    "unseen label cannot be scored against it, and mapping it to the "
                    "nearest class would invent a prediction."
                ),
            ) from exc

    def label_at(self, index: int) -> int:
        """The label for a model index."""
        return self.labels[index]

    def digest(self) -> str:
        """A content address, so a model and a report cannot disagree silently."""
        material = ",".join(str(label) for label in self.labels)
        return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]

    def as_dict(self) -> dict[str, Any]:
        """Serialise."""
        return {"labels": list(self.labels), "digest": self.digest()}


def label_space(corpus: Corpus | Sequence[Record]) -> LabelSpace:
    """Derive the label space from a corpus, sorted.

    Sorted rather than first-seen: first-seen makes the mapping depend on the
    order of the file, so shuffling a corpus would renumber the classes and every
    stored model would start predicting the wrong chapters.
    """
    labels = sorted({record.label for record in corpus})
    if not labels:
        raise ModelError(
            "the corpus has no labels, so there is nothing to classify.",
            remedy="Check that the corpus loaded; the reader reports rejected lines.",
        )
    return LabelSpace(labels=tuple(labels))


@dataclass(frozen=True, slots=True)
class Encoded:
    """A corpus as arrays, plus what was lost encoding it."""

    #: (records, max_tokens) int32 token ids, right-padded.
    sequences: np.ndarray
    #: (records,) int32 lengths before padding.
    lengths: np.ndarray
    #: (records,) int32 label indices.
    targets: np.ndarray
    #: Records whose token sequence was cut to fit. Reported, never hidden.
    truncated: int
    #: Records with no tokens at all. Cannot happen from a valid record, and is
    #: reported rather than assumed away.
    empty: int
    label_space: LabelSpace
    max_tokens: int

    @property
    def records(self) -> int:
        """How many records were encoded."""
        return int(self.sequences.shape[0])

    def counts(self, vocab_size: int) -> np.ndarray:
        """Return the (records, vocab) float64 count matrix.

        Built on demand rather than stored: for this corpus it is 5,000 x 2,048
        doubles, which is 80 MB, and only the linear model wants it.
        """
        matrix = np.zeros((self.records, vocab_size), dtype=np.float64)
        for row in range(self.records):
            length = int(self.lengths[row])
            if length:
                np.add.at(matrix[row], self.sequences[row, :length], 1.0)
        return matrix

    def as_dict(self) -> dict[str, Any]:
        """Serialise for a report."""
        return {
            "records": self.records,
            "max_tokens": self.max_tokens,
            "truncated": self.truncated,
            "empty": self.empty,
            "median_length": int(np.median(self.lengths)) if self.records else 0,
            "label_space": self.label_space.as_dict(),
        }


def encode(
    corpus: Corpus | Sequence[Record],
    tokenizer: Tokenizer,
    space: LabelSpace,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Encoded:
    """Encode *corpus* into arrays using *tokenizer* and *space*."""
    records = list(corpus)
    if not records:
        raise ModelError(
            "there are no records to encode.",
            remedy="An empty split produces a model with no evidence and a metric over nothing.",
        )
    if max_tokens < 1:
        raise ModelError(
            f"max_tokens is {max_tokens}.",
            remedy="A record encoded to zero tokens carries no information.",
        )

    padding = tokenizer.vocab[PADDING]
    sequences = np.full((len(records), max_tokens), padding, dtype=np.int32)
    lengths = np.zeros(len(records), dtype=np.int32)
    targets = np.zeros(len(records), dtype=np.int32)
    truncated = 0
    empty = 0

    for row, record in enumerate(records):
        ids = tokenizer.encode(record.text)
        if not ids:
            empty += 1
        if len(ids) > max_tokens:
            truncated += 1
            ids = ids[:max_tokens]
        sequences[row, : len(ids)] = ids
        lengths[row] = len(ids)
        targets[row] = space.index_of(record.label)

    return Encoded(
        sequences=sequences,
        lengths=lengths,
        targets=targets,
        truncated=truncated,
        empty=empty,
        label_space=space,
        max_tokens=max_tokens,
    )
