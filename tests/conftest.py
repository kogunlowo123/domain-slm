"""Shared fixtures.

Two sizes of corpus, deliberately. Most tests use a small generated corpus of a
few hundred records so the suite stays fast; the handful of tests that make
claims about the *shipped* corpus use the real thing, session-scoped so the cost
is paid once.

Credential-shaped strings do not appear here at all: this project handles no
credentials, reads no tokens, and authenticates to nothing. The secret scanner
therefore has nothing in this repository to allowlist, which is the strongest
version of that control.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from dslm.corpus.curate import apply_curation, find_duplicates
from dslm.corpus.record import Corpus, Record, build_corpus
from dslm.corpus.split import Split, split_corpus
from dslm.corpus.synth import PLANS, Plan, generate
from dslm.model import features
from dslm.tokenizer import bpe

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"

#: Small enough that a whole pipeline runs in well under a second.
SMALL = Plan(name="small", records=360, seed=11)
#: No planted duplicates: the deduplicator's negative control.
SMALL_CLEAN = Plan(
    name="small-clean", records=360, seed=12, duplicate_rate=0.0, near_duplicate_rate=0.0
)
#: Disjoint from SMALL by seed, for calibrating a contamination threshold.
SMALL_NULL = Plan(
    name="small-null", records=240, seed=13, duplicate_rate=0.0, near_duplicate_rate=0.0
)


def make_record(
    text: str = "shock strut pressure low at the gate. serviced with nitrogen.",
    *,
    label: int = 32,
    record_id: str = "r-000001",
    **meta: str,
) -> Record:
    """A valid record, with only the fields a test cares about spelled out."""
    return Record(
        record_id=record_id,
        text=text,
        label=label,
        source="test",
        license="CC0-1.0",
        meta=dict(meta),
    )


def make_corpus(texts: Sequence[str], *, labels: Sequence[int] | None = None) -> Corpus:
    """A corpus from bare strings, for tests about text rather than about records."""
    chosen = labels if labels is not None else [32] * len(texts)
    return build_corpus(
        make_record(text, label=label, record_id=f"r-{index:06d}")
        for index, (text, label) in enumerate(zip(texts, chosen, strict=True))
    )


@pytest.fixture(scope="session")
def small() -> Corpus:
    """A small generated corpus, with duplicates planted in it."""
    return generate(SMALL)


@pytest.fixture(scope="session")
def small_clean() -> Corpus:
    """The same size, with no planted duplicates."""
    return generate(SMALL_CLEAN)


@pytest.fixture(scope="session")
def small_null() -> Corpus:
    """A corpus disjoint from ``small`` by construction."""
    return generate(SMALL_NULL)


@pytest.fixture(scope="session")
def pipeline(small: Corpus) -> Split:
    """A correctly built split: deduplicate, then split by cluster."""
    cleaned = apply_curation(small, find_duplicates(small.records))
    return split_corpus(cleaned, seed=1, curation=find_duplicates(cleaned.records))


@pytest.fixture(scope="session")
def tokenizer(pipeline: Split) -> bpe.Tokenizer:
    """A tokenizer trained on the training side only.

    On the training side *only*, and that is not a detail. A tokenizer fitted on
    the whole corpus has seen the test set's vocabulary, which is a mild form of
    the leak this repository is about.
    """
    trained, _ = bpe.train((record.text for record in pipeline.train), vocab_size=512)
    return trained


@pytest.fixture(scope="session")
def space(pipeline: Split) -> features.LabelSpace:
    """The label space, derived from training."""
    return features.label_space(pipeline.train)


@pytest.fixture
def encoded(
    pipeline: Split, tokenizer: bpe.Tokenizer, space: features.LabelSpace
) -> dict[str, features.Encoded]:
    """Every part of the split, encoded."""
    return {
        part: features.encode(pipeline[part], tokenizer, space) for part in ("train", "dev", "test")
    }


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A scratch directory for tests that write files."""
    return tmp_path


@pytest.fixture(scope="session")
def shipped() -> Corpus:
    """The corpus that actually ships, for claims about the corpus that ships."""
    return generate(PLANS["main"])
