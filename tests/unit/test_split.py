"""Splitting.

The tests that matter are the ones about *leakage*: a split that puts a record
and its near-duplicate on opposite sides produces a reported accuracy that is
partly a memorisation score, and nothing downstream can tell.
"""

from __future__ import annotations

import pytest

from dslm.corpus.curate import find_duplicates
from dslm.corpus.record import Corpus, build_corpus
from dslm.corpus.split import PARTS, split_corpus
from dslm.errors import SplitError
from tests.conftest import make_record

pytestmark = pytest.mark.unit


WORDS = [
    "hydraulic seepage",
    "generator fault",
    "cabin overheat",
    "tyre wear",
    "pitot blockage",
    "door seal damage",
    "apu hot start",
    "bleed trip",
    "radar dropout",
    "flap asymmetry",
    "battery discharge",
    "light failure",
]
ACTIONS = [
    "replaced the unit",
    "cleaned the connector",
    "re-torqued the fitting",
    "rigged the mechanism",
    "serviced the reservoir",
    "swapped the module",
]


def corpus_of(count: int, *, labels: int = 4, meta: str = "") -> Corpus:
    return build_corpus(
        make_record(
            record_id=f"r-{index:04d}",
            text=f"record {index} describing a component fault that was rectified on site",
            label=21 + (index % labels),
            **({meta: f"g{index % 20}"} if meta else {}),
        )
        for index in range(count)
    )


class TestShape:
    def test_every_record_lands_in_exactly_one_part(self):
        split = split_corpus(corpus_of(200), seed=1)
        assert split.total == 200
        seen = [record.record_id for part in PARTS for record in split[part]]
        assert len(seen) == len(set(seen)) == 200

    def test_the_default_ratios_are_roughly_honoured(self):
        split = split_corpus(corpus_of(1000), seed=1)
        assert 750 <= len(split.train) <= 850
        assert 50 <= len(split.dev) <= 150
        assert 50 <= len(split.test) <= 150

    def test_a_split_is_deterministic_under_its_seed(self):
        one = split_corpus(corpus_of(200), seed=7)
        two = split_corpus(corpus_of(200), seed=7)
        assert one.digest() == two.digest()

    def test_a_different_seed_gives_a_different_assignment(self):
        one = split_corpus(corpus_of(200), seed=7)
        two = split_corpus(corpus_of(200), seed=8)
        assert one.digest() != two.digest()

    def test_the_digest_is_over_the_assignment_not_the_records(self):
        # It answers "was this model evaluated on the same test set as that
        # one?", which a digest over the records could not.
        assert split_corpus(corpus_of(200), seed=7).digest() != (
            split_corpus(corpus_of(200), seed=9).digest()
        )


class TestLeakage:
    def test_duplicate_clusters_stay_on_one_side(self):
        # The failure this module exists to prevent. Without grouping, a record
        # and its duplicate land on opposite sides and the model is tested on
        # something it memorised.
        text = (
            "A320 LHR: main gear — shock strut pressure low at the gate. "
            "serviced the strut with nitrogen to the extension chart and checked for leaks."
        )
        # Thirty copies of one record, then 170 that are genuinely unlike each
        # other. Appending an index to the same sentence would NOT do: those
        # differ by one token in twenty-four and are near-duplicates of each
        # other, so the whole corpus collapses into one cluster — which is the
        # deduplicator being right and the fixture being wrong.
        records = [make_record(record_id=f"dup-{index:03d}", text=text) for index in range(30)]
        records += [
            make_record(
                record_id=f"uniq-{index:03d}",
                text=(
                    f"{WORDS[index % len(WORDS)]} {index} reported during the check; "
                    f"{ACTIONS[index % len(ACTIONS)]} and returned to service at stand {index}"
                ),
            )
            for index in range(170)
        ]
        corpus = build_corpus(records)
        curation = find_duplicates(corpus.records)
        split = split_corpus(corpus, seed=3, curation=curation)

        where = {record.record_id: part for part in PARTS for record in split[part]}
        for cluster in curation.clusters:
            sides = {where[member] for member in (cluster.keep, *cluster.drop)}
            assert len(sides) == 1, f"cluster {cluster.keep} straddles {sides}"

    def test_grouping_by_metadata_keeps_a_group_together(self):
        corpus = corpus_of(300, meta="shop")
        split = split_corpus(corpus, seed=3, group_by="shop")
        for part in PARTS:
            groups = {record.meta["shop"] for record in split[part]}
            others = {
                record.meta["shop"] for other in PARTS if other != part for record in split[other]
            }
            assert not (groups & others)

    def test_a_record_missing_the_grouping_key_is_refused(self):
        # Assigning it by itself is exactly the leak the grouping prevents.
        corpus = corpus_of(50)
        with pytest.raises(SplitError, match="no 'shop'"):
            split_corpus(corpus, seed=1, group_by="shop")


class TestRefusals:
    def test_ratios_must_sum_to_one(self):
        with pytest.raises(SplitError, match="sum to"):
            split_corpus(corpus_of(100), seed=1, ratios=(0.8, 0.1, 0.2))

    def test_a_zero_sized_part_is_refused(self):
        # A zero-sized part is not a split, it is an omission.
        with pytest.raises(SplitError, match="positive"):
            split_corpus(corpus_of(100), seed=1, ratios=(0.9, 0.1, 0.0))

    def test_an_empty_corpus_is_refused(self):
        with pytest.raises(SplitError, match="empty corpus"):
            split_corpus(Corpus(records=()), seed=1)

    def test_a_corpus_too_small_to_fill_three_parts_says_so(self):
        with pytest.raises(SplitError, match="empty"):
            split_corpus(corpus_of(2), seed=1)

    def test_an_unknown_part_is_refused(self):
        split = split_corpus(corpus_of(100), seed=1)
        with pytest.raises(SplitError, match="not a part"):
            split["validation"]


class TestReporting:
    def test_the_summary_names_every_part(self):
        summary = split_corpus(corpus_of(200), seed=1).summary()
        for part in PARTS:
            assert part in summary

    def test_the_report_records_the_grouping(self):
        corpus = corpus_of(300, meta="shop")
        report = split_corpus(corpus, seed=1, group_by="shop").as_dict()
        assert report["grouped_by"] == "shop"
        assert set(report["sizes"]) == set(PARTS)

    def test_zero_clusters_is_reported_rather_than_hidden(self):
        # Whether grouping happened is a fact worth seeing.
        assert split_corpus(corpus_of(200), seed=1).as_dict()["grouped_clusters"] == 0
