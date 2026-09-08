"""Normalisation, shingling and near-duplicate detection.

The class that matters most is :class:`TestTheShingleArithmetic`. The shingle
width is the most consequential parameter in the module, it was chosen by
measurement, and the derivation that justifies it is checkable — so it is
checked, rather than left as a comment somebody could quietly contradict.
"""

from __future__ import annotations

import pytest

from dslm.corpus.curate import (
    DEFAULT_THRESHOLD,
    SHINGLE_WORDS,
    data_card,
    find_duplicates,
    jaccard,
    normalise,
    normalise_corpus,
    render_data_card,
    shingles,
    signature,
)
from dslm.corpus.record import build_corpus
from tests.conftest import make_corpus, make_record

pytestmark = pytest.mark.unit


class TestNormalisation:
    def test_case_and_spacing_are_folded_for_comparison(self):
        assert normalise("Serviced  the STRUT") == normalise("serviced the strut")

    def test_punctuation_is_dropped_for_comparison(self):
        assert normalise("valve, replaced.") == normalise("valve replaced")

    def test_hyphens_survive_because_they_are_part_of_domain_tokens(self):
        # "32-41-00" is a manual reference. Splitting it would destroy the
        # structure a domain tokenizer exists to learn.
        assert "32-41-00" in normalise("AMM 32-41-00 refers")

    def test_stored_text_is_not_lowercased_by_normalise_corpus(self):
        # normalise() is for *comparison*. Applying it to the stored text would
        # train the model on something no technician ever wrote.
        record = make_record(text="Replaced  the  valve")
        (cleaned,) = normalise_corpus([record])
        assert cleaned.text == "Replaced the valve"


class TestShingles:
    def test_a_short_record_yields_one_shingle_rather_than_none(self):
        # Returning the empty set would make every short record identical to
        # every other short record, which is the opposite of the truth.
        assert shingles("two words", width=5) == frozenset({"two words"})

    def test_an_empty_record_yields_nothing(self):
        assert shingles("   ") == frozenset()

    def test_shingles_overlap(self):
        assert shingles("a b c d", width=2) == frozenset({"a b", "b c", "c d"})

    def test_jaccard_of_identical_sets_is_one(self):
        assert jaccard(shingles("a valve leaked"), shingles("a valve leaked")) == 1.0

    def test_jaccard_of_disjoint_sets_is_zero(self):
        assert jaccard(shingles("a valve leaked"), shingles("quite different words here")) == 0.0


class TestTheShingleArithmetic:
    """The derivation behind SHINGLE_WORDS = 3, checked rather than asserted.

    A record of ``n`` words yields ``S = n - w + 1`` shingles. Changing one word
    destroys at most ``w`` of them on each side, so the best achievable
    similarity between a record and its one-word variant is ``(S - w)/(S + w)``.

    That is why the conventional five-word shingle fails here: on a twenty-word
    record it caps at 0.63, *below* the conventional 0.8 threshold, so the check
    reports that a corpus full of near-duplicates contains none.
    """

    @staticmethod
    def _predicted(words: int, width: int) -> float:
        shingle_count = words - width + 1
        return (shingle_count - width) / (shingle_count + width)

    @pytest.mark.parametrize("width", [3, 4, 5])
    def test_the_formula_predicts_a_one_word_edit(self, width: int):
        original = " ".join(f"w{index}" for index in range(21))
        edited = original.replace("w10", "changed")
        measured = jaccard(shingles(original, width=width), shingles(edited, width=width))
        assert measured == pytest.approx(self._predicted(21, width), abs=0.02)

    def test_the_conventional_width_cannot_reach_the_conventional_threshold(self):
        # The finding, stated as a test: at width 5 on a record this length, a
        # one-word edit cannot score 0.8 however similar the records are.
        assert self._predicted(21, 5) < 0.8

    def test_the_chosen_width_can(self):
        assert self._predicted(21, SHINGLE_WORDS) > DEFAULT_THRESHOLD


class TestSignatures:
    def test_a_signature_is_stable_across_processes(self):
        # blake2b rather than Python's hash(), which is salted per process. A
        # signature that changed daily would make every committed report
        # irreproducible.
        assert signature("a valve leaked") == signature("a valve leaked")

    def test_different_texts_give_different_signatures(self):
        assert signature("a valve leaked") != signature("a pump seized")

    def test_an_empty_text_has_a_signature_rather_than_crashing(self):
        assert len(signature("")) == len(signature("something"))


class TestFindingDuplicates:
    def test_exact_duplicates_are_found(self):
        corpus = make_corpus(
            [
                "shock strut pressure low at the gate serviced with nitrogen to the chart",
                "shock strut pressure low at the gate serviced with nitrogen to the chart",
                "generator dropped offline in cruise replaced the unit and did a load check",
            ]
        )
        curation = find_duplicates(corpus.records)
        assert curation.dropped == 1
        assert curation.exact_duplicates == 1

    def test_a_one_word_variant_is_found_too(self):
        # The case a hash set misses entirely, and the reason MinHash is here.
        # The text is a realistic twenty-one words: at thirteen the same edit
        # scores 0.571 and is genuinely below the threshold, which is the
        # arithmetic in TestTheShingleArithmetic rather than a miss.
        base = (
            "A320 LHR: main gear — shock strut pressure low at the gate. "
            "serviced the strut with nitrogen to the extension chart and checked for leaks."
        )
        corpus = make_corpus([base, base.replace("LHR", "AMS")])
        curation = find_duplicates(corpus.records)
        assert curation.dropped == 1
        assert curation.exact_duplicates == 0

    def test_a_one_word_variant_of_a_short_record_is_below_the_threshold(self):
        # Not a bug, and worth pinning so nobody "fixes" it: a thirteen-word
        # record has eleven shingles, a one-word edit destroys three of them
        # on each side, and 8/14 = 0.571 really is below 0.6. The remedy for a
        # corpus of short records is a smaller width, not a lower threshold.
        base = "shock strut pressure low at LHR serviced with nitrogen to the extension chart"
        corpus = make_corpus([base, base.replace("LHR", "AMS")])
        assert find_duplicates(corpus.records).dropped == 0
        assert find_duplicates(corpus.records, threshold=0.55).dropped == 1

    def test_unrelated_records_are_not_clustered(self):
        corpus = make_corpus(
            [
                "shock strut pressure low at the gate serviced with nitrogen to the chart",
                "transponder no reply reported by air traffic control cleaned the connector",
                "wing anti-ice valve failed to open on selection replaced the valve",
            ]
        )
        assert find_duplicates(corpus.records).dropped == 0

    def test_nothing_is_removed_by_finding(self):
        # Finding is a measurement; removing is an edit. A function that did both
        # would make it impossible to see what would go before it went.
        corpus = make_corpus(["a valve leaked here today"] * 3)
        curation = find_duplicates(corpus.records)
        assert curation.dropped == 2
        assert len(corpus.records) == 3

    def test_the_kept_record_is_the_first_by_id(self):
        corpus = make_corpus(["a valve leaked here today at the gate"] * 3)
        (cluster,) = find_duplicates(corpus.records).clusters
        assert cluster.keep == "r-000000"
        assert cluster.drop == ("r-000001", "r-000002")

    def test_an_impossible_threshold_is_refused(self):
        with pytest.raises(ValueError, match="threshold"):
            find_duplicates([make_record()], threshold=0.0)

    def test_a_single_record_has_no_duplicates(self):
        assert find_duplicates([make_record()]).dropped == 0


class TestDataCard:
    def test_the_card_is_computed_from_the_records(self):
        corpus = build_corpus(
            [
                make_record(record_id="a", label=21, text="pack flow control valve stuck open"),
                make_record(record_id="b", label=32, text="shock strut pressure low at the gate"),
            ]
        )
        card = data_card(corpus)
        assert card["records"] == 2
        assert card["labels"]["counts"] == {21: 1, 32: 1}
        assert card["licenses"] == ["CC0-1.0"]

    def test_the_majority_share_is_the_accuracy_of_guessing(self):
        corpus = build_corpus(
            [make_record(record_id=f"r{index}", label=21 if index else 32) for index in range(4)]
        )
        assert card_majority(corpus) == pytest.approx(0.75)

    def test_it_renders_as_markdown(self):
        corpus = make_corpus(["a valve leaked at the gate today"])
        rendered = render_data_card(data_card(corpus, find_duplicates(corpus.records)))
        assert rendered.startswith("# Data card")
        assert "## Provenance" in rendered
        assert "## Duplicates" in rendered


def card_majority(corpus) -> float:
    return float(data_card(corpus)["labels"]["majority_share"])
