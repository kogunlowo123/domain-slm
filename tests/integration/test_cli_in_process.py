"""The command line, driven in-process.

This layer exists because of a defect found repeatedly in this series: **a
subprocess is a different interpreter**. A CLI exercised only end to end
measures as 0% covered, and every error path in it is code nothing has ever
executed under assertion. The end-to-end layer still exists — it is the only
thing that checks exit codes and the stdout/stderr split for real — but it
cannot be the only one.

Everything here runs the real `main()` with a real argument vector against real
files in a temporary directory. Nothing is mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dslm.cli import main
from dslm.corpus.record import read_corpus
from dslm.errors import EXIT_ERROR, EXIT_GATE_FAILED, EXIT_OK

pytestmark = pytest.mark.integration


def run(*argv: str) -> int:
    return main(list(argv))


def out(capsys: pytest.CaptureFixture[str]) -> Any:
    return json.loads(capsys.readouterr().out)


@pytest.fixture
def built(workspace: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    """A whole pipeline, run through the command line, in a scratch directory."""
    assert (
        run("synth", "--plan", "main", "--records", "900", "--out", str(workspace / "c.jsonl.gz"))
        == EXIT_OK
    )
    assert (
        run(
            "synth",
            "--plan",
            "holdout",
            "--records",
            "400",
            "--out",
            str(workspace / "null.jsonl.gz"),
        )
        == EXIT_OK
    )
    assert (
        run(
            "curate",
            "--corpus",
            str(workspace / "c.jsonl.gz"),
            "--out",
            str(workspace / "clean.jsonl.gz"),
        )
        == EXIT_OK
    )
    assert (
        run(
            "split",
            "--corpus",
            str(workspace / "clean.jsonl.gz"),
            "--out-dir",
            str(workspace / "splits"),
        )
        == EXIT_OK
    )
    assert (
        run(
            "train",
            "--splits",
            str(workspace / "splits"),
            "--out-dir",
            str(workspace / "model"),
            "--vocab-size",
            "512",
            "--epochs",
            "5",
        )
        == EXIT_OK
    )
    capsys.readouterr()
    return workspace


class TestSynth:
    def test_it_writes_a_corpus_and_reports_its_digest(
        self, workspace: Path, capsys: pytest.CaptureFixture[str]
    ):
        assert (
            run(
                "synth",
                "--plan",
                "main",
                "--records",
                "300",
                "--out",
                str(workspace / "c.jsonl.gz"),
            )
            == EXIT_OK
        )
        report = out(capsys)
        assert report["records"] == 300
        assert report["digest"].startswith("sha256:")
        assert read_corpus(workspace / "c.jsonl.gz").digest() == report["digest"]

    def test_the_seed_can_be_overridden(self, workspace: Path, capsys: pytest.CaptureFixture[str]):
        run("synth", "--records", "200", "--seed", "1", "--out", str(workspace / "a.jsonl.gz"))
        one = out(capsys)["digest"]
        run("synth", "--records", "200", "--seed", "2", "--out", str(workspace / "b.jsonl.gz"))
        assert out(capsys)["digest"] != one


class TestCheck:
    def test_an_unmodified_corpus_matches_its_plan(
        self, workspace: Path, capsys: pytest.CaptureFixture[str]
    ):
        run("synth", "--plan", "holdout", "--out", str(workspace / "c.jsonl.gz"))
        capsys.readouterr()
        assert (
            run("check", "--plan", "holdout", "--corpus", str(workspace / "c.jsonl.gz")) == EXIT_OK
        )

    def test_a_corpus_from_another_plan_does_not(
        self, workspace: Path, capsys: pytest.CaptureFixture[str]
    ):
        run("synth", "--plan", "holdout", "--out", str(workspace / "c.jsonl.gz"))
        capsys.readouterr()
        assert (
            run("check", "--plan", "clean", "--corpus", str(workspace / "c.jsonl.gz")) == EXIT_ERROR
        )
        assert "no longer matches" in capsys.readouterr().err


class TestCurate:
    def test_it_reports_duplicates_and_writes_a_card(
        self, workspace: Path, capsys: pytest.CaptureFixture[str]
    ):
        run("synth", "--records", "400", "--out", str(workspace / "c.jsonl.gz"))
        capsys.readouterr()
        assert (
            run(
                "curate",
                "--corpus",
                str(workspace / "c.jsonl.gz"),
                "--out",
                str(workspace / "clean.jsonl.gz"),
                "--card",
                str(workspace / "CARD.md"),
            )
            == EXIT_OK
        )
        report = out(capsys)
        assert report["duplicates"]["dropped"] > 0
        assert (workspace / "CARD.md").read_text(encoding="utf-8").startswith("# Data card")
        assert len(read_corpus(workspace / "clean.jsonl.gz")) < 400

    def test_nothing_is_written_unless_asked(
        self, workspace: Path, capsys: pytest.CaptureFixture[str]
    ):
        # Curation is a measurement; removing is an edit. The edit is opt-in.
        run("synth", "--records", "300", "--out", str(workspace / "c.jsonl.gz"))
        capsys.readouterr()
        run("curate", "--corpus", str(workspace / "c.jsonl.gz"))
        assert not (workspace / "clean.jsonl.gz").exists()


class TestSplit:
    def test_it_writes_three_parts(self, workspace: Path, capsys: pytest.CaptureFixture[str]):
        run("synth", "--records", "600", "--out", str(workspace / "c.jsonl.gz"))
        capsys.readouterr()
        assert (
            run(
                "split",
                "--corpus",
                str(workspace / "c.jsonl.gz"),
                "--out-dir",
                str(workspace / "s"),
            )
            == EXIT_OK
        )
        report = out(capsys)
        assert set(report["sizes"]) == {"train", "dev", "test"}
        for part in ("train", "dev", "test"):
            assert (workspace / "s" / f"{part}.jsonl.gz").is_file()

    def test_impossible_ratios_are_refused(
        self, workspace: Path, capsys: pytest.CaptureFixture[str]
    ):
        run("synth", "--records", "300", "--out", str(workspace / "c.jsonl.gz"))
        capsys.readouterr()
        code = run(
            "split",
            "--corpus",
            str(workspace / "c.jsonl.gz"),
            "--out-dir",
            str(workspace / "s"),
            "--train",
            "0.5",
            "--dev",
            "0.2",
            "--test",
            "0.2",
        )
        assert code == EXIT_ERROR
        assert "sum to" in capsys.readouterr().err


class TestTrain:
    def test_it_trains_both_models(self, built: Path, capsys: pytest.CaptureFixture[str]):
        # Both, always. A command that trained only the neural one would make
        # the comparison optional, which is how a control stops being run.
        assert (
            run(
                "train",
                "--splits",
                str(built / "splits"),
                "--out-dir",
                str(built / "m2"),
                "--vocab-size",
                "512",
                "--epochs",
                "3",
            )
            == EXIT_OK
        )
        report = out(capsys)
        assert report["naive_bayes"]["kind"] == "naive-bayes"
        assert report["neural"]["kind"] == "neural"
        assert (built / "m2" / "model.npz").is_file()
        assert (built / "m2" / "tokenizer.json").is_file()

    def test_the_model_records_the_tokenizer_it_was_trained_against(self, built: Path):
        metadata = json.loads((built / "model" / "model.json").read_text(encoding="utf-8"))
        tokenizer = json.loads((built / "model" / "tokenizer.json").read_text(encoding="utf-8"))
        assert metadata["tokenizer_digest"] == tokenizer["digest"]


class TestEvaluate:
    def test_a_clean_pipeline_passes_and_reports_both_models(
        self, built: Path, capsys: pytest.CaptureFixture[str]
    ):
        code = run(
            "evaluate",
            "--splits",
            str(built / "splits"),
            "--model",
            str(built / "model"),
            "--null",
            str(built / "null.jsonl.gz"),
            "--drop-shared",
        )
        assert code == EXIT_OK
        report = out(capsys)
        assert report["passed"] is True
        assert set(report["metrics"]) == {"naive-bayes", "neural"}
        assert report["contamination"]["calibrated"] is True

    def test_without_a_null_it_refuses_and_reports_no_metric_at_all(
        self, built: Path, capsys: pytest.CaptureFixture[str]
    ):
        # Not "metrics with a caveat". A number that exists is a number somebody
        # quotes out of context.
        code = run("evaluate", "--splits", str(built / "splits"), "--model", str(built / "model"))
        assert code == EXIT_GATE_FAILED
        report = out(capsys)
        assert report["passed"] is False
        assert "metrics" not in report
        assert "refused" in report

    def test_the_uncalibrated_rule_can_be_opted_into(
        self, built: Path, capsys: pytest.CaptureFixture[str]
    ):
        code = run(
            "evaluate",
            "--splits",
            str(built / "splits"),
            "--model",
            str(built / "model"),
            "--allow-uncalibrated",
        )
        assert code == EXIT_OK
        assert out(capsys)["contamination"]["calibrated"] is False

    def test_a_baseline_round_trips_and_then_holds(
        self, built: Path, capsys: pytest.CaptureFixture[str]
    ):
        common = [
            "evaluate",
            "--splits",
            str(built / "splits"),
            "--model",
            str(built / "model"),
            "--null",
            str(built / "null.jsonl.gz"),
            "--drop-shared",
            "--baseline",
            str(built / "baseline.json"),
        ]
        assert run(*common, "--update-baseline") == EXIT_OK
        capsys.readouterr()
        assert (built / "baseline.json").is_file()
        assert run(*common) == EXIT_OK
        assert out(capsys)["baseline"]["passed"] is True

    def test_no_baseline_is_reported_as_not_compared_rather_than_as_passing(
        self, built: Path, capsys: pytest.CaptureFixture[str]
    ):
        code = run(
            "evaluate",
            "--splits",
            str(built / "splits"),
            "--model",
            str(built / "model"),
            "--null",
            str(built / "null.jsonl.gz"),
            "--drop-shared",
            "--baseline",
            str(built / "absent.json"),
        )
        assert code == EXIT_OK
        assert "baseline" not in out(capsys)

    def test_reports_are_written_where_asked(self, built: Path, capsys: pytest.CaptureFixture[str]):
        run(
            "evaluate",
            "--splits",
            str(built / "splits"),
            "--model",
            str(built / "model"),
            "--null",
            str(built / "null.jsonl.gz"),
            "--drop-shared",
            "--json-out",
            str(built / "r.json"),
            "--junit-out",
            str(built / "r.xml"),
            "--markdown-out",
            str(built / "r.md"),
        )
        assert json.loads((built / "r.json").read_text(encoding="utf-8"))["passed"] is True
        assert (built / "r.xml").read_text(encoding="utf-8").startswith("<?xml")
        assert (built / "r.md").read_text(encoding="utf-8").startswith("# Evaluation")

    def test_the_report_goes_to_stdout_or_a_file_never_both(
        self, built: Path, capsys: pytest.CaptureFixture[str]
    ):
        run(
            "evaluate",
            "--splits",
            str(built / "splits"),
            "--model",
            str(built / "model"),
            "--null",
            str(built / "null.jsonl.gz"),
            "--drop-shared",
            "--json-out",
            str(built / "r.json"),
        )
        assert capsys.readouterr().out == ""


class TestContamination:
    def test_it_measures_two_corpora(self, built: Path, capsys: pytest.CaptureFixture[str]):
        code = run(
            "contamination",
            "--train",
            str(built / "splits" / "train.jsonl.gz"),
            "--evaluate",
            str(built / "splits" / "test.jsonl.gz"),
            "--null",
            str(built / "null.jsonl.gz"),
            "--drop-shared",
        )
        assert code == EXIT_OK
        assert out(capsys)["calibrated"] is True

    def test_a_corpus_against_itself_fails(self, built: Path, capsys: pytest.CaptureFixture[str]):
        code = run(
            "contamination",
            "--train",
            str(built / "splits" / "train.jsonl.gz"),
            "--evaluate",
            str(built / "splits" / "train.jsonl.gz"),
            "--allow-uncalibrated",
        )
        assert code == EXIT_GATE_FAILED


class TestClassify:
    def test_it_labels_text(self, built: Path, capsys: pytest.CaptureFixture[str]):
        code = run(
            "classify",
            "--model",
            str(built / "model"),
            "--text",
            "shock strut pressure low at the gate, serviced with nitrogen",
        )
        assert code == EXIT_OK
        results = out(capsys)
        assert len(results) == 1
        assert 0 <= results[0]["confidence"] <= 1

    def test_classifying_nothing_is_refused(self, built: Path, capsys: pytest.CaptureFixture[str]):
        assert run("classify", "--model", str(built / "model")) == EXIT_ERROR
        assert "nothing to classify" in capsys.readouterr().err


class TestDoctor:
    def test_it_runs_a_whole_pipeline_with_no_files(self, capsys: pytest.CaptureFixture[str]):
        assert run("doctor") == EXIT_OK
        assert "self-check      ok" in capsys.readouterr().out

    def test_it_can_also_inspect_a_corpus(self, built: Path, capsys: pytest.CaptureFixture[str]):
        assert run("doctor", "--corpus", str(built / "c.jsonl.gz")) == EXIT_OK
        printed = capsys.readouterr().out
        assert "licences" in printed
        assert "digest" in printed


class TestErrorHandling:
    def test_a_missing_corpus_exits_three_with_a_remedy(
        self, workspace: Path, capsys: pytest.CaptureFixture[str]
    ):
        # Exit 3, not 2: "the tool could not run" and "a gate failed" are
        # different facts, and a pipeline conflating them retries until green.
        assert run("doctor", "--corpus", str(workspace / "absent.jsonl.gz")) == EXIT_ERROR
        captured = capsys.readouterr().err
        assert "could not be opened" in captured
        assert "dslm synth" in captured

    def test_a_broken_environment_variable_exits_three_rather_than_tracing_back(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        # The trap: settings loaded above main's try escape as a traceback and
        # exit 1, while the unit tests that call load() directly still pass.
        monkeypatch.setenv("DSLM_GATE__TOLERANCE", "not-a-number")
        assert run("doctor") == EXIT_ERROR
        assert "settings are invalid" in capsys.readouterr().err

    def test_a_misspelt_section_is_refused_too(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ):
        # extra="forbid" never sees a misspelt *section*: pydantic does not
        # build a `gates` key for it to reject.
        monkeypatch.setenv("DSLM_GATES__TOLERANCE", "0.01")
        assert run("doctor") == EXIT_ERROR
        assert "DSLM_GATES__TOLERANCE" in capsys.readouterr().err
