"""Reading and writing corpora, against real files."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from dslm.corpus import record as record_module
from dslm.corpus.record import (
    MAX_LINE_BYTES,
    Corpus,
    build_corpus,
    is_compressed,
    open_text,
    read_corpus,
)
from dslm.errors import CorpusError
from tests.conftest import make_corpus, make_record

pytestmark = pytest.mark.integration


class TestRoundTrip:
    def test_a_corpus_survives_being_written_and_read(self, workspace: Path):
        corpus = make_corpus(["a valve leaked", "a pump seized", "a light failed"])
        recovered = read_corpus(corpus.write(workspace / "c.jsonl"))
        assert len(recovered) == 3
        assert recovered.digest() == corpus.digest()

    def test_metadata_survives(self, workspace: Path):
        corpus = build_corpus([make_record()], metadata={"plan": "test"})
        corpus.write(workspace / "c.jsonl")
        assert read_corpus(workspace / "c.jsonl").metadata["plan"] == "test"

    def test_the_file_is_json_lines(self, workspace: Path):
        # One header, then one record per line. What makes a truncated file lose
        # one record rather than all of them.
        make_corpus(["one", "two"]).write(workspace / "c.jsonl")
        lines = (workspace / "c.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        assert json.loads(lines[0])["dslm_corpus"] == 1
        assert all(json.loads(line)["record_id"] for line in lines[1:])

    def test_optional_fields_are_omitted_rather_than_written_empty(self, workspace: Path):
        make_corpus(["one"]).write(workspace / "c.jsonl")
        second = (workspace / "c.jsonl").read_text(encoding="utf-8").splitlines()[1]
        assert "meta" not in json.loads(second)


class TestCompression:
    def test_a_gzipped_corpus_round_trips(self, workspace: Path):
        corpus = make_corpus(["a valve leaked", "a pump seized"])
        recovered = read_corpus(corpus.write(workspace / "c.jsonl.gz"))
        assert recovered.digest() == corpus.digest()

    def test_the_file_really_is_gzip(self, workspace: Path):
        make_corpus(["one"]).write(workspace / "c.jsonl.gz")
        assert (workspace / "c.jsonl.gz").read_bytes()[:2] == b"\x1f\x8b"

    def test_writing_the_same_corpus_twice_gives_the_same_bytes(self, workspace: Path):
        # gzip stamps the current time by default, which would make every
        # regenerated corpus differ from the committed one for a reason no
        # reader could see — and turn the drift check into noise.
        corpus = make_corpus(["a valve leaked", "a pump seized"])
        first = corpus.write(workspace / "a.jsonl.gz").read_bytes()
        second = corpus.write(workspace / "b.jsonl.gz").read_bytes()
        assert first == second

    def test_compression_does_not_change_the_content_address(self, workspace: Path):
        corpus = make_corpus(["a valve leaked", "a pump seized"])
        corpus.write(workspace / "plain.jsonl")
        corpus.write(workspace / "packed.jsonl.gz")
        assert read_corpus(workspace / "plain.jsonl").digest() == (
            read_corpus(workspace / "packed.jsonl.gz").digest()
        )

    def test_it_is_the_suffix_that_decides(self):
        # Sniffing the magic bytes teaches people to ship gzip named .jsonl, and
        # then something downstream reading by name gets bytes it cannot parse.
        assert is_compressed("c.jsonl.gz")
        assert not is_compressed("c.jsonl")
        assert not is_compressed("c.gz.jsonl")

    def test_a_decompression_bomb_is_refused(self, workspace: Path, monkeypatch):
        # Accepting compressed input means accepting this. Checking stat() alone
        # would let a 5 KB file expand until the runner is killed, which reaches
        # an operator as a flaky build rather than as an attack.
        path = workspace / "bomb.jsonl.gz"
        with gzip.open(path, "wb") as handle:
            handle.write(b"0" * (5 * 1024 * 1024))
        # Above the compressed size on disk and far below the expanded size, so
        # it is the stream guard that fires rather than the cheap stat() check.
        monkeypatch.setattr(record_module, "MAX_CORPUS_BYTES", 64 * 1024)
        with pytest.raises(CorpusError, match="expands to more than"):
            read_corpus(path)

    def test_open_text_reads_both_forms(self, workspace: Path):
        make_corpus(["one"]).write(workspace / "c.jsonl")
        make_corpus(["one"]).write(workspace / "c.jsonl.gz")
        with open_text(workspace / "c.jsonl") as plain, open_text(workspace / "c.jsonl.gz") as gz:
            assert plain.read() == gz.read()


class TestDamagedFiles:
    def _write(self, path: Path, *lines: str) -> Path:
        header = json.dumps({"dslm_corpus": 1, "records": len(lines)})
        path.write_text("\n".join([header, *lines]) + "\n", encoding="utf-8")
        return path

    def test_a_malformed_record_is_counted_not_fatal(self, workspace: Path):
        # A corpus is written by a process that can be killed. Refusing the whole
        # file because of one truncated record throws away the rest of it.
        first = json.dumps(make_record(record_id="a").as_dict())
        second = json.dumps(make_record(record_id="b").as_dict())
        path = self._write(workspace / "c.jsonl", first, "{not json", second)
        corpus = read_corpus(path)
        assert len(corpus) == 2
        assert len(corpus.unreadable) == 1

    def test_the_reason_is_recorded_with_the_line_number(self, workspace: Path):
        path = self._write(workspace / "c.jsonl", json.dumps({"record_id": "x"}))
        (number, reason) = read_corpus(path).unreadable[0]
        assert number == 2
        assert reason

    def test_a_missing_header_is_fatal(self, workspace: Path):
        # Without it there is no record of what this file claims to be, and every
        # check downstream would be against a guess.
        path = workspace / "c.jsonl"
        path.write_text("not json at all\n", encoding="utf-8")
        with pytest.raises(CorpusError, match="does not start with a corpus header"):
            read_corpus(path)

    def test_a_header_without_the_marker_is_fatal(self, workspace: Path):
        path = workspace / "c.jsonl"
        path.write_text(json.dumps({"records": 0}) + "\n", encoding="utf-8")
        with pytest.raises(CorpusError, match="no corpus header"):
            read_corpus(path)

    def test_a_newer_schema_is_refused(self, workspace: Path):
        path = workspace / "c.jsonl"
        path.write_text(json.dumps({"dslm_corpus": 99}) + "\n", encoding="utf-8")
        with pytest.raises(CorpusError, match="newer version"):
            read_corpus(path)

    def test_an_empty_file_is_refused(self, workspace: Path):
        path = workspace / "c.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(CorpusError, match="is empty"):
            read_corpus(path)

    def test_an_enormous_line_is_rejected_rather_than_read(self, workspace: Path):
        path = self._write(workspace / "c.jsonl", '{"x": "' + "y" * MAX_LINE_BYTES + '"}')
        corpus = read_corpus(path)
        assert "longer than" in corpus.unreadable[0][1]

    def test_a_missing_file_names_how_to_make_one(self, workspace: Path):
        # The remedy, not the message. A message says what is wrong; a remedy
        # says what to do, and the reader is rarely the person who wrote it.
        with pytest.raises(CorpusError) as caught:
            read_corpus(workspace / "absent.jsonl")
        assert "dslm synth" in caught.value.remedy

    def test_blank_lines_are_ignored_rather_than_counted_as_damage(self, workspace: Path):
        good = json.dumps(make_record().as_dict())
        path = self._write(workspace / "c.jsonl", good, "", "   ")
        corpus = read_corpus(path)
        assert len(corpus) == 1
        assert corpus.unreadable == ()


class TestTheShippedGenerator:
    def test_a_generated_corpus_survives_the_round_trip(self, workspace: Path, small: Corpus):
        recovered = read_corpus(small.write(workspace / "c.jsonl.gz"))
        assert recovered.digest() == small.digest()
        assert recovered.metadata["generated"] == "true"

    def test_the_provenance_reaches_the_file(self, workspace: Path, small: Corpus):
        small.write(workspace / "c.jsonl.gz")
        recovered = read_corpus(workspace / "c.jsonl.gz")
        assert recovered.licenses == ("CC0-1.0",)
        assert all(record.source for record in recovered)
