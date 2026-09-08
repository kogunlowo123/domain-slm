"""Adversarial cases. A failure here is a security regression, not a bug.

This project's threat model is unusual for the series, and the tests reflect
that: it handles no credentials, opens no socket and executes nothing from a
file. What it *does* handle is somebody else's data and somebody else's files,
so the surface that remains is resource exhaustion, hostile input, and the
integrity of the verdict itself.

The claims below are deliberately precise. "No secrets leak" would be trivially
true here and therefore worthless; what is asserted instead is that the specific
controls exist and are wired up.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from dslm.corpus import record as record_module
from dslm.corpus.contamination import contamination_report, enforce
from dslm.corpus.record import MAX_LINE_BYTES, MAX_TEXT_CHARS, Record, read_corpus
from dslm.errors import ContaminationError, CorpusError, ModelError
from dslm.evaluate.report import Evaluation, render_json, render_markdown
from dslm.model import io as model_io
from tests.conftest import make_corpus, make_record

pytestmark = pytest.mark.security


class TestTheRecordCannotCarryFreeText:
    """The structural control: there is nowhere to put a prompt.

    A telemetry or corpus schema leaks user text through the field somebody
    added because the debugging was hard. Here there is no such field, and
    ``extra="forbid"`` means adding one is refused at validation rather than
    carried through to every consumer.
    """

    @pytest.mark.parametrize(
        "field", ["prompt", "completion", "response", "raw", "notes", "user_id", "email"]
    )
    def test_an_extra_field_is_refused(self, field: str):
        payload: dict[str, Any] = {
            "record_id": "r",
            "text": "a valve leaked",
            "label": 32,
            "source": "test",
            "license": "CC0-1.0",
            field: "something sensitive",
        }
        with pytest.raises(ValidationError):
            Record(**payload)

    def test_the_metadata_bag_is_strings_only(self):
        # Not a place to smuggle a nested document.
        with pytest.raises(ValidationError):
            make_record(nested={"prompt": "hidden"})  # type: ignore[arg-type]

    def test_text_is_bounded(self):
        # An unbounded text field is a transcript field with a different name.
        with pytest.raises(ValidationError):
            make_record(text="x " * MAX_TEXT_CHARS)


class TestResourceExhaustion:
    def test_a_decompression_bomb_is_refused(self, workspace: Path, monkeypatch):
        # Accepting .gz input means accepting this. Without the stream guard a
        # small file expands until the runner is killed, which reaches an
        # operator as a flaky build rather than as an attack.
        path = workspace / "bomb.jsonl.gz"
        with gzip.open(path, "wb") as handle:
            handle.write(b"0" * (8 * 1024 * 1024))
        monkeypatch.setattr(record_module, "MAX_CORPUS_BYTES", 64 * 1024)
        with pytest.raises(CorpusError, match="expands to more than"):
            read_corpus(path)

    def test_an_oversized_file_is_refused_before_it_is_read(self, workspace: Path, monkeypatch):
        path = workspace / "c.jsonl"
        path.write_text(json.dumps({"dslm_corpus": 1}) + "\n", encoding="utf-8")
        monkeypatch.setattr(record_module, "MAX_CORPUS_BYTES", 4)
        with pytest.raises(CorpusError, match="the limit is"):
            read_corpus(path)

    def test_one_enormous_line_does_not_take_the_reader_down_with_it(self, workspace: Path):
        path = workspace / "c.jsonl"
        header = json.dumps({"dslm_corpus": 1})
        payload = '{"x": "' + "y" * (MAX_LINE_BYTES * 2) + '"}'
        good = json.dumps(make_record().as_dict())
        path.write_text("\n".join([header, payload, good]) + "\n", encoding="utf-8")
        corpus = read_corpus(path)
        assert len(corpus) == 1
        assert corpus.unreadable


class TestNothingIsExecutedOrImported:
    def test_the_corpus_reader_does_not_evaluate_its_input(self, workspace: Path):
        # json.loads, never eval. The record is validated into a frozen model
        # with a closed set of fields.
        path = workspace / "c.jsonl"
        header = json.dumps({"dslm_corpus": 1})
        hostile = json.dumps(
            {
                "record_id": "r",
                "text": "__import__('os').system('echo pwned')",
                "label": 32,
                "source": "test",
                "license": "CC0-1.0",
            }
        )
        path.write_text("\n".join([header, hostile]) + "\n", encoding="utf-8")
        corpus = read_corpus(path)
        # The text is data. It is stored, tokenised and counted, and never run.
        assert corpus.records[0].text.startswith("__import__")

    def test_a_model_artefact_is_numpy_arrays_and_json_not_a_pickle(self, workspace: Path):
        # np.savez, not pickle: loading a pickle from an untrusted path executes
        # whatever it was told to. `allow_pickle` is left at its safe default.
        source = Path(model_io.__file__).read_text(encoding="utf-8")
        assert "pickle" not in source
        assert "allow_pickle=True" not in source


class TestTheVerdictCannotBeQuietlyWeakened:
    def test_a_model_run_against_a_different_tokenizer_is_refused(
        self, workspace: Path, encoded, tokenizer
    ):
        # Not an error anyone would otherwise see: token 412 simply means a
        # different string, and the result is an accuracy several points lower
        # that reads as a bad model rather than as a mismatch.
        from dslm.model import neural

        model, _ = neural.train(
            encoded["train"],
            epochs=2,
            vocab_size=tokenizer.vocab_size,
            padding_id=tokenizer.vocab["<pad>"],
            tokenizer_digest=tokenizer.digest(),
        )
        model_io.save(model, workspace / "m")
        with pytest.raises(ModelError, match="different tokenizer"):
            model_io.load(workspace / "m", tokenizer_digest="sha256:something-else")

    def test_a_truncated_model_is_refused_rather_than_loaded(
        self, workspace: Path, encoded, tokenizer
    ):
        from dslm.model import neural

        model, _ = neural.train(encoded["train"], epochs=1, vocab_size=tokenizer.vocab_size)
        model_io.save(model, workspace / "m")
        metadata = json.loads((workspace / "m" / "model.json").read_text(encoding="utf-8"))
        metadata["shapes"]["embedding"] = [1, 1]
        (workspace / "m" / "model.json").write_text(json.dumps(metadata), encoding="utf-8")
        with pytest.raises(ModelError, match="shape"):
            model_io.load(workspace / "m")

    def test_a_contaminated_evaluation_reports_no_metric_at_all(self):
        # The strongest form of the control. Not "a metric with a warning": the
        # number does not appear in the document, so it cannot be quoted out of
        # context by anything downstream.
        train = make_corpus([" ".join(f"w{index}" for index in range(30))] * 4)
        report = contamination_report(train, train, width=5)
        with pytest.raises(ContaminationError):
            enforce(report, allow_uncalibrated=True)

        evaluation = Evaluation(
            metrics={},
            contamination=report,
            verdict=None,
            comparisons=(),
            corpus_digest="sha256:c",
            split_digest="sha256:s",
            tokenizer_digest="sha256:t",
            refused="the evaluation set also appears in training",
        )
        document = json.loads(render_json(evaluation))
        assert document["passed"] is False
        assert document["trustworthy"] is False
        assert "metrics" not in document
        assert "label_noise_ceiling" not in document

    def test_the_markdown_summary_prints_no_number_either(self):
        train = make_corpus([" ".join(f"w{index}" for index in range(30))] * 4)
        evaluation = Evaluation(
            metrics={},
            contamination=contamination_report(train, train, width=5),
            verdict=None,
            comparisons=(),
            corpus_digest="sha256:c",
            split_digest="sha256:s",
            tokenizer_digest="sha256:t",
            refused="the evaluation set also appears in training",
        )
        rendered = render_markdown(evaluation)
        assert rendered.startswith("# Evaluation: REFUSED")
        assert "Accuracy" not in rendered

    def test_deleting_the_refusal_would_change_the_document(self):
        # The test that would fail if the control were removed: the same
        # evaluation without `refused` publishes its metrics.
        train = make_corpus([" ".join(f"w{index}" for index in range(30))] * 4)
        report = contamination_report(train, train, width=5)
        without = Evaluation(
            metrics={},
            contamination=report,
            verdict=None,
            comparisons=(),
            corpus_digest="sha256:c",
            split_digest="sha256:s",
            tokenizer_digest="sha256:t",
        )
        assert "metrics" in json.loads(render_json(without))


class TestNoNetworkAndNoCredentials:
    def test_the_package_opens_no_socket(self):
        # Stated as a property of the source rather than of one run: there is no
        # HTTP client, no socket, and nothing to configure an endpoint on.
        root = Path(record_module.__file__).resolve().parent.parent
        sources = [path.read_text(encoding="utf-8") for path in root.rglob("*.py")]
        joined = "\n".join(sources)
        for forbidden in ("import socket", "import requests", "urllib.request", "httpx"):
            assert forbidden not in joined

    def test_there_is_no_credential_setting_to_get_wrong(self):
        from dslm.config import Settings

        fields = set(Settings.model_fields)
        assert not {"api_key", "token", "secret", "credentials"} & fields
