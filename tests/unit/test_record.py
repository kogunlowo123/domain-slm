"""Records and corpora.

The validation here is the boundary between "a file someone gave us" and
everything downstream, so most of these are about what is *refused*.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dslm.corpus.record import MAX_TEXT_CHARS, Corpus, Record, build_corpus, require_records
from dslm.errors import CorpusError, RecordError
from tests.conftest import make_record

pytestmark = pytest.mark.unit


class TestValidation:
    def test_a_valid_record_is_accepted(self):
        record = make_record()
        assert record.label == 32
        assert record.license == "CC0-1.0"

    def test_provenance_is_required(self):
        # The field most often omitted and least often recoverable afterwards.
        with pytest.raises(ValidationError):
            Record(record_id="r", text="a valve", label=32, license="CC0-1.0")  # type: ignore[call-arg]

    def test_a_licence_is_required(self):
        with pytest.raises(ValidationError):
            Record(record_id="r", text="a valve", label=32, source="test")  # type: ignore[call-arg]

    def test_an_unknown_field_is_refused_rather_than_carried(self):
        # extra="forbid". A field nobody declared is a field nothing validates,
        # and "prompt" is exactly the one that would appear.
        with pytest.raises(ValidationError):
            Record(
                record_id="r",
                text="a valve",
                label=32,
                source="test",
                license="CC0-1.0",
                prompt="what is wrong with the aircraft",  # type: ignore[call-arg]
            )

    def test_whitespace_only_text_is_refused(self):
        with pytest.raises(ValidationError):
            make_record(text="   \n  ")

    def test_text_must_be_nfc(self):
        # Two strings that look identical and differ in composition are two
        # vocabulary entries, two n-grams, and a missed duplicate.
        decomposed = "cabin témoin light"  # combining acute
        with pytest.raises(ValidationError):
            make_record(text=decomposed)

    def test_a_record_longer_than_the_limit_is_refused_not_truncated(self):
        with pytest.raises(ValidationError):
            make_record(text="valve " * (MAX_TEXT_CHARS // 3))

    def test_a_label_outside_the_ata_range_is_refused(self):
        with pytest.raises(ValidationError):
            make_record(label=412)

    def test_records_are_frozen(self):
        with pytest.raises(ValidationError):
            make_record().text = "something else"


class TestFingerprints:
    def test_the_fingerprint_covers_the_text_and_the_label(self):
        assert make_record(text="a").fingerprint() != make_record(text="b").fingerprint()
        assert make_record(label=32).fingerprint() != make_record(label=33).fingerprint()

    def test_it_does_not_cover_the_provenance_or_the_id(self):
        # Correcting a licence or renaming a record does not change what was
        # trained, and a drift check that fired on those is one people disable.
        left = make_record(record_id="a")
        right = left.model_copy(update={"record_id": "b", "source": "elsewhere"})
        assert left.fingerprint() == right.fingerprint()


class TestCorpora:
    def test_duplicate_ids_are_refused(self):
        with pytest.raises(CorpusError, match="more than once"):
            build_corpus([make_record(record_id="same"), make_record(record_id="same")])

    def test_the_digest_ignores_order(self):
        # What a model learns from is the set. A shuffle that moved the digest
        # would make every legitimate reordering look like a data change.
        one = make_record(record_id="a", text="alpha valve")
        two = make_record(record_id="b", text="beta valve")
        assert build_corpus([one, two]).digest() == build_corpus([two, one]).digest()

    def test_the_digest_moves_when_a_record_changes(self):
        one = make_record(record_id="a", text="alpha valve")
        two = make_record(record_id="b", text="beta valve")
        changed = two.model_copy(update={"text": "gamma valve"})
        assert build_corpus([one, two]).digest() != build_corpus([one, changed]).digest()

    def test_label_counts_are_sorted(self):
        corpus = build_corpus(
            [
                make_record(record_id="a", label=34),
                make_record(record_id="b", label=21),
                make_record(record_id="c", label=21),
            ]
        )
        assert corpus.label_counts() == {21: 2, 34: 1}
        assert corpus.labels == (21, 34)

    def test_licences_are_reported(self):
        corpus = build_corpus([make_record(record_id="a"), make_record(record_id="b")])
        assert corpus.licenses == ("CC0-1.0",)

    def test_an_empty_corpus_is_refused_where_it_would_be_meaningless(self):
        # An empty corpus produces a tokenizer with no merges and an accuracy of
        # 1.0 over zero examples. Both are numbers; neither is a measurement.
        with pytest.raises(RecordError, match="no records"):
            require_records(Corpus(records=()), what="training a tokenizer")

    def test_require_records_passes_a_populated_corpus(self):
        require_records(build_corpus([make_record()]), what="anything")
