"""The byte-pair encoder.

The most important test in this file is
:meth:`TestTheFastPath.test_it_agrees_with_the_reference_implementation`.
:func:`train` is a 35x optimisation of :func:`train_reference`, and an
optimisation whose only evidence is that it runs faster is a rewrite nobody can
trust.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dslm.errors import TokenizerError
from dslm.tokenizer import bpe

pytestmark = pytest.mark.unit

TEXTS = [
    "shock strut pressure low at the gate serviced with nitrogen to the chart",
    "shock strut seal replaced and the gear retraction checked on jacks",
    "generator dropped offline in cruise replaced the unit and carried out a load check",
    "generator control unit replaced after an intermittent bus fault was recorded",
    "wing anti-ice valve failed to open on selection replaced the valve",
    "amm 32-41-00 refers for the brake assembly replacement and taxi check",
] * 6


@pytest.fixture(scope="module")
def trained() -> bpe.Tokenizer:
    """One tokenizer for the encoding tests.

    Module scope rather than class scope: pytest deprecates a class-scoped
    fixture written as an instance method, and this project turns warnings
    into errors.
    """
    tokenizer, _ = bpe.train(TEXTS, vocab_size=400)
    return tokenizer


class TestTraining:
    def test_it_produces_a_vocabulary_of_about_the_requested_size(self):
        tokenizer, report = bpe.train(TEXTS, vocab_size=400)
        assert tokenizer.vocab_size <= 400
        assert report.merges > 0

    def test_training_is_deterministic(self):
        # Every operation is over integers and strings, so this is exact rather
        # than approximate — which is what lets the tokenizer be digest-gated
        # while the model can only be gated on metrics.
        assert bpe.train(TEXTS, vocab_size=400)[0].digest() == (
            bpe.train(TEXTS, vocab_size=400)[0].digest()
        )

    def test_the_special_tokens_come_first_and_keep_their_ids(self):
        # Fixed ids, so a model trained against one vocabulary and run against
        # another fails on the digest rather than quietly on a shifted id.
        tokenizer, _ = bpe.train(TEXTS, vocab_size=400)
        assert tokenizer.vocab[bpe.PADDING] == 0
        assert tokenizer.vocab[bpe.UNKNOWN] == 1

    def test_every_byte_is_representable(self):
        tokenizer, _ = bpe.train(TEXTS, vocab_size=400)
        assert all(f"<0x{byte:02X}>" in tokenizer.vocab for byte in range(256))

    def test_unfilled_merges_are_reported_rather_than_ignored(self):
        # Asking for 8,000 tokens and getting 400 is a fact about the corpus.
        _, report = bpe.train(TEXTS[:6], vocab_size=5000)
        assert report.unfilled > 0
        assert "unfilled" in report.summary()

    def test_an_empty_corpus_is_refused(self):
        with pytest.raises(TokenizerError, match="no words"):
            bpe.train([], vocab_size=400)

    @pytest.mark.parametrize("size", [10, 500_000])
    def test_an_out_of_range_vocabulary_is_refused(self, size: int):
        with pytest.raises(TokenizerError, match="outside"):
            bpe.train(TEXTS, vocab_size=size)

    def test_a_vocabulary_too_small_for_the_alphabet_is_refused(self):
        # The 256 byte tokens are always present, so a corpus with a wide
        # alphabet of its own can exhaust the smallest permitted vocabulary
        # before a single merge is made. Refused, rather than silently
        # producing a tokenizer with no merges in it.
        wide = ["".join(chr(0x4E00 + index) for index in range(80))] * 4
        with pytest.raises(TokenizerError, match="alphabet"):
            bpe.train(wide, vocab_size=bpe.MIN_VOCAB)


class TestTheFastPath:
    def test_it_agrees_with_the_reference_implementation(self):
        # The optimisation is incremental pair counting. This is the only thing
        # that says it computes the same answer.
        fast, _ = bpe.train(TEXTS, vocab_size=450)
        slow, _ = bpe.train_reference(TEXTS, vocab_size=450)
        assert fast.merges == slow.merges
        assert fast.digest() == slow.digest()

    def test_they_agree_on_a_second_corpus_too(self):
        other = [text.replace("valve", "actuator") for text in TEXTS]
        assert bpe.train(other, vocab_size=420)[0].merges == (
            bpe.train_reference(other, vocab_size=420)[0].merges
        )


class TestEncoding:
    def test_encoding_round_trips_to_the_normalised_form(self, trained: bpe.Tokenizer):
        text = "Shock Strut pressure low"
        assert trained.decode(trained.encode(text)) == "shock strut pressure low"

    def test_it_never_emits_the_unknown_token(self, trained: bpe.Tokenizer):
        # The byte fallback guarantees it. A tokenizer that can produce <unk>
        # loses information silently, and the loss is blamed on the model.
        exotic = "圧力 low ✈ strut"
        assert trained.vocab[bpe.UNKNOWN] not in trained.encode(exotic)

    def test_an_unseen_word_still_encodes(self, trained: bpe.Tokenizer):
        assert trained.encode("quixotic")

    def test_encoding_is_stable(self, trained: bpe.Tokenizer):
        text = "shock strut pressure low at the gate"
        assert trained.encode(text) == trained.encode(text)

    def test_the_end_of_word_marker_separates_a_suffix_from_a_word(self, trained: bpe.Tokenizer):
        assert trained.encode("valve") != trained.encode("valves")


class TestPreTokenisation:
    def test_it_lowercases_and_splits_on_whitespace(self):
        assert bpe.pre_tokenise("Shock  STRUT low") == ["shock", "strut", "low"]

    def test_it_keeps_the_punctuation_that_carries_domain_meaning(self):
        # "32-41-00" is a manual reference and "p/n" is a part number.
        assert bpe.pre_tokenise("amm 32-41-00 p/n 356-1194") == [
            "amm",
            "32-41-00",
            "p/n",
            "356-1194",
        ]


class TestPersistence:
    def test_a_tokenizer_survives_a_round_trip(self, tmp_path: Path):
        tokenizer, _ = bpe.train(TEXTS, vocab_size=400)
        loaded = bpe.load(tokenizer.save(tmp_path / "tok.json"))
        assert loaded.digest() == tokenizer.digest()
        assert loaded.encode("shock strut") == tokenizer.encode("shock strut")

    def test_a_missing_file_is_refused(self, tmp_path: Path):
        with pytest.raises(TokenizerError, match="could not be opened"):
            bpe.load(tmp_path / "absent.json")

    def test_malformed_json_is_refused(self, tmp_path: Path):
        path = tmp_path / "tok.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(TokenizerError, match="not valid JSON"):
            bpe.load(path)

    def test_an_edited_tokenizer_is_refused_by_its_own_digest(self, tmp_path: Path):
        # A model trained against a different vocabulary produces ids that mean
        # something else, and nothing downstream would notice.
        import json

        tokenizer, _ = bpe.train(TEXTS, vocab_size=400)
        path = tokenizer.save(tmp_path / "tok.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["merges"] = payload["merges"][:-1]
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(TokenizerError, match="digest"):
            bpe.load(path)

    def test_a_newer_schema_is_refused(self, tmp_path: Path):
        import json

        path = tmp_path / "tok.json"
        path.write_text(json.dumps({"dslm_tokenizer": 99}), encoding="utf-8")
        with pytest.raises(TokenizerError, match="newer version"):
            bpe.load(path)


class TestCompression:
    def test_it_measures_tokens_per_record(self):
        tokenizer, _ = bpe.train(TEXTS, vocab_size=400)
        measured = bpe.compression(tokenizer, TEXTS)
        assert measured.records == len(TEXTS)
        assert measured.tokens_per_record > 0
        assert measured.unknown == 0

    def test_a_domain_tokenizer_beats_one_trained_on_other_text(self):
        # The claim the module exists to support, measured rather than asserted.
        # The comparison is against a tokenizer of the *same size* trained on
        # different text — not against a word split, which would be a different
        # kind of thing wearing the same units.
        other = [
            "the quick brown fox jumps over the lazy dog again and again today",
            "she sells sea shells on the sea shore every single summer morning",
            "it was the best of times it was the worst of times and so on",
        ] * 12
        domain, _ = bpe.train(TEXTS, vocab_size=400)
        general, _ = bpe.train(other, vocab_size=400)
        assert bpe.compression(domain, TEXTS).tokens_per_record < (
            bpe.compression(general, TEXTS).tokens_per_record
        )

    def test_measuring_over_nothing_is_refused(self):
        tokenizer, _ = bpe.train(TEXTS, vocab_size=400)
        with pytest.raises(TokenizerError, match="nothing to measure"):
            bpe.compression(tokenizer, [])
