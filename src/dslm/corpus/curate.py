"""Curation: normalise, find near-duplicates, and say what the corpus is.

Three jobs, in order, and the order matters.

**Normalise first.** ``Serviced  the STRUT`` and ``serviced the strut`` are the
same work order. Comparing them before normalising means the deduplicator is
partly a case-sensitivity detector, and every downstream count is off by an
amount nobody can quantify afterwards.

**Then find near-duplicates, not exact ones.** A hash set finds records that are
byte-identical, which in maintenance text is the easy and rare case. The
expensive case is the same write-up with the station code changed, or a
different part number, or one extra word — for training purposes one record, and
for a hash set two. That is what MinHash is here for, and the shipped corpus
plants both kinds at known rates so the difference can be measured rather than
asserted.

**Then write down what you have.** The data card is generated from the records,
never typed by hand. A card that is written by hand is a card that is right on
the day it is written.

Nothing here deletes anything. ``curate`` reports clusters and marks the records
it would drop; the caller decides. A curation step that silently removed a
quarter of a corpus would be the single most consequential unlogged action in
the pipeline.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from dslm.corpus.record import Corpus, Record, build_corpus

#: Shingle width, in words.
#:
#: **Three, not the conventional five, and this is the most consequential
#: parameter in the file.** The five-word shingle at a 0.8 threshold comes from
#: web-scale document deduplication, where a document is hundreds of words long
#: and a one-word edit is a rounding error. A maintenance record is about twenty
#: words, and there the same setting cannot detect a one-word edit *at all*.
#:
#: The arithmetic says why, exactly. A record of ``n`` words yields
#: ``S = n - w + 1`` shingles. Changing one word destroys at most ``w`` of them
#: on each side, so the best achievable similarity between a record and its
#: one-word variant is::
#:
#:     J = (S - w) / (S + w)
#:
#: For a 21-word record that is 0.78 at w=3, 0.70 at w=4 and **0.63 at w=5** —
#: below the conventional threshold, so a five-word shingle at 0.8 reports that
#: a corpus of near-duplicates contains none. Measured over the shipped corpus
#: the formula predicts the observed median to three decimal places at every
#: width. See docs/curation.md.
SHINGLE_WORDS = 3

#: MinHash signature length. 128 permutations estimate a Jaccard similarity to
#: roughly ±0.09 at one standard deviation, which is ample when the gap between
#: the two populations is 0.727 against 0.471 (below).
PERMUTATIONS = 128

#: LSH banding: 32 bands of 4 rows.
#:
#: Chosen to match ``DEFAULT_THRESHOLD``, not by convention. The probability that
#: two records share at least one whole band is ``1 - (1 - J**rows)**bands``, an
#: S-curve whose midpoint sits at ``(1/bands)**(1/rows)``. Everything depends on
#: putting that midpoint *below* the threshold being enforced:
#:
#: ======  ======  ==========  ==========  ==========
#: bands   rows    midpoint    P at J=0.6  P at J=0.8
#: ======  ======  ==========  ==========  ==========
#: 16      8       0.707       0.237       0.947
#: **32**  **4**   **0.420**   **0.988**   **1.000**
#: ======  ======  ==========  ==========  ==========
#:
#: The first row is the textbook default and it was the original setting here.
#: With a 0.6 threshold it finds a pair sitting *at* that threshold barely one
#: time in four — a deduplicator that misses three quarters of exactly the cases
#: it was configured to catch. The unit test that pins this is the one that
#: caught it.
#:
#: The cost of 32 bands is more candidate pairs to verify. That is the right
#: side of the trade: verification is an exact Jaccard over two small sets, and
#: banding exists to avoid the 18 million comparisons that checking every pair
#: of 6,000 records would need — not to be as cheap as possible.
BANDS = 32
ROWS = PERMUTATIONS // BANDS

#: Two records in the same band bucket are compared exactly; this is the
#: similarity at or above which they are called duplicates.
#:
#: Chosen by measurement over the shipped corpus rather than by convention. At
#: ``SHINGLE_WORDS = 3`` a planted one-word variant scores **0.727 or above**
#: (median 0.778) and an unrelated pair scores **0.471 or below** (99.9th
#: percentile 0.306). 0.6 sits in the gap with margin on both sides, and the
#: test that pins these two populations apart is the one that would fail if
#: either the grammar or the shingle width changed underneath it.
DEFAULT_THRESHOLD = 0.6

#: A cluster needs two members to be a cluster. Named because the bare 2
#: appears twice below and means the same thing both times.
MIN_CLUSTER = 2

_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r"[^\w\s-]+")


def normalise(text: str) -> str:
    """Return the comparison form of *text*.

    Lowercased, NFC-composed, punctuation removed, whitespace collapsed. Used
    for **comparison only** — the record keeps its original text, because the
    tokenizer and the model should see what a technician actually wrote.
    """
    folded = unicodedata.normalize("NFC", text).lower()
    folded = _PUNCTUATION.sub(" ", folded)
    return _WHITESPACE.sub(" ", folded).strip()


def shingles(text: str, *, width: int = SHINGLE_WORDS) -> frozenset[str]:
    """Return the set of overlapping *width*-word shingles of *text*.

    A record shorter than the shingle width yields one shingle: the whole
    record. Returning the empty set instead would make every short record
    identical to every other short record, which is the opposite of the truth.
    """
    words = normalise(text).split()
    if len(words) <= width:
        return frozenset({" ".join(words)}) if words else frozenset()
    return frozenset(
        " ".join(words[index : index + width]) for index in range(len(words) - width + 1)
    )


#: A Mersenne prime just above 2**61. Used as the modulus of the permutation
#: family below; a prime modulus is what makes ``(a*h + b) mod p`` a permutation
#: of the residues rather than a map that collapses them.
_PRIME = (1 << 61) - 1


def _permutations(count: int) -> tuple[tuple[int, int], ...]:
    """Return *count* (a, b) coefficient pairs for the hash family.

    Derived from a fixed seed, so the family is the same on every machine and
    forever. Not ``random``: a signature computed today has to match one
    computed next year, or the committed corpus's report is irreproducible.
    """
    coefficients: list[tuple[int, int]] = []
    for index in range(count):
        material = hashlib.blake2b(str(index).encode(), digest_size=16).digest()
        a = int.from_bytes(material[:8], "big") % (_PRIME - 1) + 1
        b = int.from_bytes(material[8:], "big") % _PRIME
        coefficients.append((a, b))
    return tuple(coefficients)


#: Computed once. Building it per call was the original implementation and made
#: signing 6,000 records take twelve seconds.
_FAMILY = _permutations(PERMUTATIONS)


def _base_hash(value: str) -> int:
    """One 64-bit hash of a shingle.

    blake2b rather than Python's ``hash``: ``hash`` is randomised per process by
    default, so a signature computed today would not match one computed
    tomorrow and every report over the committed corpus would be
    irreproducible.
    """
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), "big") % _PRIME


def signature(text: str, *, family: tuple[tuple[int, int], ...] = _FAMILY) -> tuple[int, ...]:
    """Return the MinHash signature of *text*.

    Hash each shingle **once**, then apply an affine permutation per position.
    The naive alternative — a fresh cryptographic hash per (shingle,
    permutation) pair — is 128 times more hashing and made this the slowest step
    in the pipeline by two orders of magnitude. The estimator is the same one;
    only the cost differs.
    """
    grams = shingles(text)
    if not grams:
        return tuple([_PRIME] * len(family))
    base = [_base_hash(gram) for gram in grams]
    return tuple(min((a * h + b) % _PRIME for h in base) for a, b in family)


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Exact Jaccard similarity. Used to confirm a candidate pair, never to find one."""
    if not left and not right:
        return 1.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


@dataclass(frozen=True, slots=True)
class Cluster:
    """A group of records the curator considers one record.

    ``keep`` is the first by id, which is arbitrary and deliberately so: any
    rule that preferred, say, the longest text would be a silent editorial
    decision about which write-up is the real one.
    """

    keep: str
    drop: tuple[str, ...]
    #: The lowest exact similarity between the kept record and any dropped one.
    #: Reported so a reviewer can see how close the call was.
    min_similarity: float

    @property
    def size(self) -> int:
        """How many records are in the cluster, including the one kept."""
        return len(self.drop) + 1

    def as_dict(self) -> dict[str, Any]:
        """Serialise for a report."""
        return {
            "keep": self.keep,
            "drop": list(self.drop),
            "size": self.size,
            "min_similarity": round(self.min_similarity, 4),
        }


@dataclass(frozen=True, slots=True)
class Curation:
    """What curation found. Nothing has been removed yet."""

    clusters: tuple[Cluster, ...]
    kept: int
    #: Records that are byte-identical after normalisation. A subset of what the
    #: clusters cover, reported separately because the two failures have
    #: different causes: exact duplicates are usually a broken ingest, near
    #: duplicates are usually how people write.
    exact_duplicates: int
    threshold: float
    comparisons: int

    @property
    def dropped(self) -> int:
        """How many records curation would remove."""
        return sum(len(cluster.drop) for cluster in self.clusters)

    @property
    def total(self) -> int:
        """How many records went in."""
        return self.kept + self.dropped

    def summary(self) -> str:
        """One line, for a terminal."""
        if not self.clusters:
            return f"no duplicates at Jaccard >= {self.threshold:g} over {self.total} record(s)"
        share = self.dropped / self.total if self.total else 0.0
        return (
            f"{self.dropped} of {self.total} record(s) are duplicates "
            f"({share:.1%}) in {len(self.clusters)} cluster(s); "
            f"{self.exact_duplicates} exact, {self.dropped - self.exact_duplicates} near"
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialise for a report."""
        return {
            "total": self.total,
            "kept": self.kept,
            "dropped": self.dropped,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.dropped - self.exact_duplicates,
            "clusters": len(self.clusters),
            "threshold": self.threshold,
            "candidate_pairs_compared": self.comparisons,
            "largest_cluster": max((c.size for c in self.clusters), default=0),
        }


def find_duplicates(  # noqa: C901 - banding, union-find and clustering, in that order
    records: Sequence[Record], *, threshold: float = DEFAULT_THRESHOLD
) -> Curation:
    """Cluster near-duplicate records using MinHash with LSH banding.

    Banding is what makes this tractable. Comparing every pair of 6,000 records
    is 18 million comparisons; banding compares only the pairs that agree on a
    whole band of the signature, which for this corpus is a few thousand. The
    cost is a small false-negative rate at the threshold, quantified in the
    band/row constants above and preferable to an approach that does not finish.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")

    grams = [shingles(record.text) for record in records]
    signatures = [signature(record.text) for record in records]

    #: band index -> bucket key -> record indices
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for index, sig in enumerate(signatures):
        for band in range(BANDS):
            key = (band, sig[band * ROWS : (band + 1) * ROWS])
            buckets.setdefault(key, []).append(index)

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

    similarity: dict[tuple[int, int], float] = {}
    compared: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < MIN_CLUSTER:
            continue
        for position, left in enumerate(members):
            for right in members[position + 1 :]:
                pair = (left, right) if left < right else (right, left)
                if pair in compared:
                    continue
                compared.add(pair)
                score = jaccard(grams[pair[0]], grams[pair[1]])
                if score >= threshold:
                    similarity[pair] = score
                    union(*pair)

    groups: dict[int, list[int]] = {}
    for index in range(len(records)):
        groups.setdefault(find(index), []).append(index)

    normalised = [normalise(record.text) for record in records]
    clusters: list[Cluster] = []
    exact = 0
    for root, members in sorted(groups.items()):
        if len(members) < MIN_CLUSTER:
            continue
        ordered = sorted(members)
        keep, drop = ordered[0], ordered[1:]
        scores = [
            similarity.get((min(keep, other), max(keep, other)), jaccard(grams[keep], grams[other]))
            for other in drop
        ]
        exact += sum(1 for other in drop if normalised[other] == normalised[keep])
        clusters.append(
            Cluster(
                keep=records[keep].record_id,
                drop=tuple(records[other].record_id for other in drop),
                min_similarity=min(scores) if scores else 1.0,
            )
        )
        del root

    dropped = sum(len(cluster.drop) for cluster in clusters)
    return Curation(
        clusters=tuple(clusters),
        kept=len(records) - dropped,
        exact_duplicates=exact,
        threshold=threshold,
        comparisons=len(compared),
    )


def apply_curation(corpus: Corpus, curation: Curation) -> Corpus:
    """Return a corpus with the duplicate records removed.

    Separate from :func:`find_duplicates` on purpose. Finding is a measurement
    and removing is an edit, and a function that did both would make it
    impossible to look at what would be removed before removing it.
    """
    drop = {record_id for cluster in curation.clusters for record_id in cluster.drop}
    kept = [record for record in corpus.records if record.record_id not in drop]
    metadata = dict(corpus.metadata)
    metadata["curated"] = "true"
    metadata["curation_dropped"] = str(len(corpus.records) - len(kept))
    metadata["curation_threshold"] = f"{curation.threshold:g}"
    return build_corpus(kept, source=corpus.source, metadata=metadata)


def data_card(corpus: Corpus, curation: Curation | None = None) -> dict[str, Any]:
    """Generate the data card from the records themselves.

    Everything here is computed. A data card typed by hand is a card that is
    correct on the day it is written and wrong from the first change afterwards,
    which is worse than none because it is believed.
    """
    lengths = sorted(len(record.text.split()) for record in corpus.records)
    counts = corpus.label_counts()
    total = len(corpus.records)
    card: dict[str, Any] = {
        "records": total,
        "digest": corpus.digest(),
        "labels": {
            "classes": len(counts),
            "counts": counts,
            "majority_share": round(max(counts.values()) / total, 4) if counts else 0.0,
        },
        "licenses": sorted({record.license for record in corpus.records}),
        "sources": sorted({record.source for record in corpus.records}),
        "text": {
            "words_min": lengths[0] if lengths else 0,
            "words_median": lengths[len(lengths) // 2] if lengths else 0,
            "words_max": lengths[-1] if lengths else 0,
            "unique_texts": len({record.text for record in corpus.records}),
        },
        "metadata": dict(corpus.metadata),
        "unreadable_lines": len(corpus.unreadable),
    }
    if curation is not None:
        card["duplicates"] = curation.as_dict()
    return card


def render_data_card(card: dict[str, Any]) -> str:
    """Render a data card as Markdown, for a repository or a job summary."""
    lines = [
        "# Data card",
        "",
        f"**{card['records']} record(s)**, digest `{card['digest']}`.",
        "",
        "## Provenance",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Sources | {', '.join(card['sources']) or '—'} |",
        f"| Licences | {', '.join(card['licenses']) or '—'} |",
    ]
    metadata = card.get("metadata") or {}
    if metadata.get("generated") == "true":
        lines.append("| Generated | yes — see the note below |")
    lines.extend(["", "## Labels", "", "| Label | Records |", "| --- | --- |"])
    lines.extend(f"| {label} | {count} |" for label, count in card["labels"]["counts"].items())
    lines.extend(
        [
            "",
            (
                f"{card['labels']['classes']} classes. The largest holds "
                f"{card['labels']['majority_share']:.1%} of the corpus, which is the "
                "accuracy a majority-class classifier would reach."
            ),
            "",
            "## Text",
            "",
            (
                f"- {card['text']['words_min']} to {card['text']['words_max']} "
                f"words, median {card['text']['words_median']}"
            ),
            f"- {card['text']['unique_texts']} distinct strings",
        ]
    )
    duplicates = card.get("duplicates")
    if duplicates:
        lines.extend(
            [
                "",
                "## Duplicates",
                "",
                (
                    f"- {duplicates['dropped']} of {duplicates['total']} records are "
                    f"duplicates at Jaccard >= {duplicates['threshold']:g}"
                ),
                f"- {duplicates['exact_duplicates']} exact, {duplicates['near_duplicates']} near",
                f"- {duplicates['clusters']} clusters, largest {duplicates['largest_cluster']}",
            ]
        )
    if metadata.get("note"):
        lines.extend(["", "## Note", "", metadata["note"]])
    return "\n".join(lines) + "\n"


def normalise_corpus(records: Iterable[Record]) -> tuple[Record, ...]:
    """Return records whose text has been whitespace-collapsed and NFC-composed.

    Not lowercased and not stripped of punctuation: those are *comparison*
    transformations, and applying them to the stored text would train the model
    on something no technician ever wrote.
    """
    out: list[Record] = []
    for record in records:
        cleaned = _WHITESPACE.sub(" ", unicodedata.normalize("NFC", record.text)).strip()
        out.append(
            record if cleaned == record.text else record.model_copy(update={"text": cleaned})
        )
    return tuple(out)
