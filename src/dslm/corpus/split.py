"""Splitting a corpus into train, dev and test.

A split is where most reported accuracies go wrong, and it goes wrong quietly.
Three failures, in the order they are usually made:

**Splitting before deduplicating.** A record and its near-duplicate land on
opposite sides, the model memorises one and is tested on the other, and the
reported number is a memorisation score wearing an accuracy's clothes. So this
module takes the *clusters* found by :mod:`dslm.corpus.curate` and keeps every
member of a cluster on the same side. Deduplicating first and splitting after is
not sufficient on its own — the duplicate is gone, but a *related* record may
remain — which is why grouping is done here rather than assumed upstream.

**Splitting randomly when the data has groups.** If the same aircraft, shop or
author appears on both sides, the model can key on the group. Grouped splitting
is available and off by default, because whether a group leaks is a property of
the domain rather than of the code, and defaulting it on would silently make
every split smaller and stranger than asked for.

**Splitting non-deterministically.** A split that differs between runs makes
every comparison between two models meaningless. The seed is required, recorded
in the split's own metadata, and part of its digest.

None of this is sufficient. The check that actually catches the mistake is the
contamination gate in :mod:`dslm.corpus.contamination`, which measures overlap
between the splits this module produced rather than trusting that it did its
job.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from random import Random
from typing import Any

from dslm.corpus.curate import Curation
from dslm.corpus.record import Corpus, Record, build_corpus
from dslm.errors import SplitError

#: The names a split has. Fixed rather than configurable: three parts with these
#: meanings is the convention every consumer of this repository already knows,
#: and a fourth would need a reason nobody has yet had.
PARTS = ("train", "dev", "test")

#: Default proportions. Dev exists so that a threshold, an early-stopping point
#: or a hyperparameter can be chosen without touching test. A repository with no
#: dev split has either not tuned anything or has tuned on test.
DEFAULT_RATIOS = (0.8, 0.1, 0.1)

#: How far the ratios may sum from 1 before they are refused. Floating-point
#: slack only: 0.8 + 0.1 + 0.1 is not exactly 1.0 in binary.
RATIO_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class Split:
    """A corpus divided into train, dev and test, and how it was divided."""

    train: Corpus
    dev: Corpus
    test: Corpus
    seed: int
    grouped_by: str
    #: How many clusters were kept together. Zero when no curation was supplied,
    #: which is a fact worth seeing rather than a default worth hiding.
    grouped_clusters: int

    def __getitem__(self, part: str) -> Corpus:
        if part not in PARTS:
            raise SplitError(
                f"{part!r} is not a part of a split.",
                remedy=f"The parts are {', '.join(PARTS)}.",
            )
        return getattr(self, part)  # type: ignore[no-any-return]

    @property
    def total(self) -> int:
        """How many records are in the split, across all three parts."""
        return len(self.train) + len(self.dev) + len(self.test)

    def digest(self) -> str:
        """A content address over the assignment, not over the records.

        Two splits of the same corpus with different seeds have the same records
        and different digests, which is what makes this worth reporting: it
        answers "was this model evaluated on the same test set as that one?".
        """
        material = "\n".join(
            f"{part}:{record.record_id}"
            for part in PARTS
            for record in sorted(self[part].records, key=lambda r: r.record_id)
        )
        return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        """Serialise for a report."""
        return {
            "seed": self.seed,
            "grouped_by": self.grouped_by,
            "grouped_clusters": self.grouped_clusters,
            "digest": self.digest(),
            "sizes": {part: len(self[part]) for part in PARTS},
            "label_counts": {part: self[part].label_counts() for part in PARTS},
        }

    def summary(self) -> str:
        """One line, for a terminal."""
        sizes = ", ".join(f"{part} {len(self[part])}" for part in PARTS)
        grouping = f", grouped by {self.grouped_by}" if self.grouped_by else ""
        return f"{self.total} record(s): {sizes} (seed {self.seed}{grouping})"


def _groups(  # noqa: C901 - two grouping sources, merged transitively
    records: Sequence[Record], curation: Curation | None, group_by: str
) -> list[list[int]]:
    """Assign every record to a group; records in one group share a side.

    Two sources of grouping, combined: the duplicate clusters, and an optional
    metadata key. A record can be pulled into a group by either, and the groups
    are merged transitively — if A duplicates B and B shares an aircraft with C,
    all three stay together. That is stricter than either rule alone, and being
    stricter than necessary costs a slightly uneven split while being looser
    costs a wrong number.
    """
    index_of = {record.record_id: index for index, record in enumerate(records)}
    parent = list(range(len(records)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    if curation is not None:
        for cluster in curation.clusters:
            members = [cluster.keep, *cluster.drop]
            present = [index_of[rid] for rid in members if rid in index_of]
            for other in present[1:]:
                union(present[0], other)

    if group_by:
        by_value: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            value = record.meta.get(group_by)
            if value is None:
                raise SplitError(
                    f"record {record.record_id!r} has no {group_by!r} in its metadata.",
                    remedy=(
                        "Grouped splitting needs the key on every record. A record "
                        "without it would be assigned by itself, which is exactly the "
                        "leak the grouping exists to prevent."
                    ),
                )
            by_value.setdefault(value, []).append(index)
        for indices in by_value.values():
            for other in indices[1:]:
                union(indices[0], other)

    grouped: dict[int, list[int]] = {}
    for index in range(len(records)):
        grouped.setdefault(find(index), []).append(index)
    return [sorted(members) for _, members in sorted(grouped.items())]


def split_corpus(
    corpus: Corpus,
    *,
    seed: int,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    curation: Curation | None = None,
    group_by: str = "",
) -> Split:
    """Divide *corpus* into train, dev and test.

    Groups are shuffled and then filled greedily into whichever part is furthest
    below its target share. Greedy rather than proportional because groups have
    different sizes: assigning them by proportion leaves the last few groups
    with nowhere sensible to go, and the part that ends up short is always test.
    """
    if len(ratios) != len(PARTS):
        raise SplitError(
            f"expected {len(PARTS)} ratios, got {len(ratios)}.",
            remedy=f"One for each of {', '.join(PARTS)}.",
        )
    if any(ratio <= 0 for ratio in ratios):
        raise SplitError(
            "every ratio must be positive.",
            remedy=(
                "A zero-sized part is not a split, it is an omission. Leave the part "
                "out of the report rather than creating an empty one."
            ),
        )
    if abs(sum(ratios) - 1.0) > RATIO_TOLERANCE:
        raise SplitError(
            f"the ratios sum to {sum(ratios):g}, not 1.",
            remedy="Ratios are shares of the corpus and must sum to exactly one.",
        )
    if not corpus.records:
        raise SplitError(
            "cannot split an empty corpus.",
            remedy="Check the reader's 'unreadable' count; every line may have been rejected.",
        )

    groups = _groups(corpus.records, curation, group_by)
    # nosec B311 - a split has to be *reproducible*, which is the opposite of
    # the property a cryptographic generator provides. Nothing here is a secret.
    rng = Random(seed)  # noqa: S311  # nosec B311
    rng.shuffle(groups)

    targets = [ratio * len(corpus.records) for ratio in ratios]
    buckets: list[list[int]] = [[], [], []]
    for group in groups:
        # The part furthest below its target, measured as a shortfall in
        # records. Ties break towards the earlier part, which is deterministic.
        deficits = [targets[i] - len(buckets[i]) for i in range(len(PARTS))]
        chosen = max(range(len(PARTS)), key=lambda i: deficits[i])
        buckets[chosen].extend(group)

    empty = [PARTS[i] for i, bucket in enumerate(buckets) if not bucket]
    if empty:
        raise SplitError(
            f"the split left {', '.join(empty)} empty.",
            remedy=(
                "Too few groups for the requested ratios. Either the corpus is very "
                "small, or grouping has merged most of it into one group — check the "
                "duplicate cluster sizes."
            ),
        )

    parts: list[Corpus] = []
    for name, bucket in zip(PARTS, buckets, strict=True):
        parts.append(
            build_corpus(
                [corpus.records[index] for index in sorted(bucket)],
                source=f"{corpus.source}#{name}",
                metadata={**corpus.metadata, "split": name, "split_seed": str(seed)},
            )
        )

    return Split(
        train=parts[0],
        dev=parts[1],
        test=parts[2],
        seed=seed,
        grouped_by=group_by,
        grouped_clusters=len(curation.clusters) if curation is not None else 0,
    )
