"""The corpus generator.

Determinism is the whole contract: a committed corpus can be regenerated and
checked rather than taken on trust, and that is only true if the same plan gives
the same records on every machine, forever.

The rest are about the corpus being *hard* enough. An earlier version of this
grammar gave every chapter a disjoint vocabulary, and both models scored 100.00%
— a number that measured the grammar rather than the model. These tests pin the
two properties that fixed it.
"""

from __future__ import annotations

import pytest

from dslm.corpus.synth import (
    AIRCRAFT,
    CHAPTERS,
    GENERIC_COMPONENTS,
    LICENSE,
    PLANS,
    SOURCE,
    Plan,
    generate,
    plan_digest,
)

pytestmark = pytest.mark.unit

TINY = Plan(name="tiny", records=240, seed=5)


class TestDeterminism:
    def test_the_same_plan_gives_the_same_corpus(self):
        assert generate(TINY).digest() == generate(TINY).digest()

    def test_including_the_record_ids(self):
        assert [record.record_id for record in generate(TINY)] == [
            record.record_id for record in generate(TINY)
        ]

    def test_a_different_seed_gives_a_different_corpus(self):
        other = Plan(name="tiny", records=TINY.records, seed=TINY.seed + 1)
        assert generate(TINY).digest() != generate(other).digest()

    def test_the_plan_digest_moves_with_the_plan(self):
        assert plan_digest([TINY]) != plan_digest([Plan(name="tiny", records=241, seed=5)])


class TestProvenance:
    def test_every_record_carries_a_source_and_a_licence(self):
        # Required fields, not a note in a README. By the time anyone asks, the
        # person who assembled the corpus has usually left.
        corpus = generate(TINY)
        assert {record.source for record in corpus} == {SOURCE}
        assert {record.license for record in corpus} == {LICENSE}

    def test_the_corpus_says_it_was_generated(self):
        metadata = generate(TINY).metadata
        assert metadata["generated"] == "true"
        assert "Synthesised" in metadata["note"]
        assert "captured from a real" in metadata["note"]

    def test_the_planted_rates_are_recorded_in_the_metadata(self):
        metadata = generate(TINY).metadata
        assert int(metadata["planted_exact_duplicates"]) > 0
        assert int(metadata["planted_near_duplicates"]) > 0


class TestTheTaskIsHardEnough:
    def test_the_chapters_share_vocabulary(self):
        # Without this the task is keyword lookup, both models score 100%, and
        # the comparison between them is vacuous.
        corpus = generate(Plan(name="wide", records=1200, seed=6))
        texts = " ".join(record.text for record in corpus)
        assert any(component in texts for component in GENERIC_COMPONENTS)

    def test_a_generic_component_appears_under_more_than_one_chapter(self):
        corpus = generate(Plan(name="wide", records=1200, seed=6))
        for component in GENERIC_COMPONENTS:
            labels = {record.label for record in corpus if component in record.text}
            if len(labels) > 1:
                return
        pytest.fail("no chapter-neutral component appeared under two chapters")

    def test_label_noise_is_planted_and_recoverable(self):
        # The text describes the true chapter and only the recorded label is
        # wrong — which is how a mis-chaptered work order actually looks, and
        # what puts a genuine ceiling under any achievable accuracy.
        corpus = generate(Plan(name="noisy", records=1200, seed=6, label_noise=0.1))
        corrupted = [record for record in corpus if "true_label" in record.meta]
        assert corrupted
        assert all(int(record.meta["true_label"]) != record.label for record in corrupted)

    def test_no_label_noise_means_no_corrupted_records(self):
        corpus = generate(Plan(name="clean", records=240, seed=6, label_noise=0.0))
        assert not [record for record in corpus if "true_label" in record.meta]


class TestShape:
    def test_every_chapter_appears(self):
        corpus = generate(Plan(name="wide", records=1200, seed=6))
        assert set(corpus.labels) == {chapter.number for chapter in CHAPTERS}

    def test_the_classes_are_roughly_balanced(self):
        # So that a majority-class baseline is weak and accuracy means something.
        counts = generate(Plan(name="wide", records=1200, seed=6)).label_counts()
        assert max(counts.values()) < 2 * min(counts.values())

    def test_records_carry_bounded_metadata(self):
        corpus = generate(TINY)
        assert {record.meta["aircraft"] for record in corpus} <= set(AIRCRAFT)

    def test_a_plan_with_no_room_for_fresh_records_is_refused(self):
        with pytest.raises(ValueError, match="no fresh records"):
            generate(Plan(name="bad", records=10, duplicate_rate=0.6, near_duplicate_rate=0.6))

    def test_the_description_states_what_was_planted(self):
        description = TINY.describe()
        assert "duplicates" in description
        assert "label noise" in description


class TestTheShippedPlans:
    @pytest.mark.parametrize("name", sorted(PLANS))
    def test_each_one_generates(self, name: str):
        corpus = generate(PLANS[name])
        assert len(corpus) == PLANS[name].records

    @pytest.mark.parametrize("name", sorted(PLANS))
    def test_each_one_says_what_it_shows(self, name: str):
        assert PLANS[name].metadata["shows"]

    def test_each_one_has_a_different_seed(self):
        seeds = [plan.seed for plan in PLANS.values()]
        assert len(set(seeds)) == len(seeds)

    def test_the_holdout_is_almost_disjoint_from_the_main_corpus(self):
        # "Almost": two corpora from one grammar with different seeds really can
        # produce the same sentence, and the contamination gate reports that
        # collision rather than hiding it. One in 1,500 is what a finite
        # vocabulary does.
        main = {record.fingerprint() for record in generate(PLANS["main"])}
        holdout = generate(PLANS["holdout"])
        shared = [record for record in holdout if record.fingerprint() in main]
        assert len(shared) < 0.01 * len(holdout)
