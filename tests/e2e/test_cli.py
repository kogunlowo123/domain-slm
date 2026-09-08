"""The command line as a real process.

The in-process layer is where the coverage and most of the assertions live. This
layer exists for the two things it alone can check: **the exit code the operating
system actually sees**, and the split between stdout and stderr. Both are the
interface a CI job consumes, and neither is observable from inside the
interpreter that produced them.

Kept small on purpose — every case here costs a process launch.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dslm.errors import EXIT_ERROR, EXIT_GATE_FAILED, EXIT_OK, EXIT_USAGE

pytestmark = pytest.mark.e2e


def dslm(*argv: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "dslm", *argv],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
        cwd=cwd,
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A whole pipeline, built once, by real subprocesses."""
    workspace = tmp_path_factory.mktemp("e2e")
    assert (
        dslm("synth", "--records", "900", "--out", str(workspace / "c.jsonl.gz")).returncode
        == EXIT_OK
    )
    assert (
        dslm(
            "synth",
            "--plan",
            "holdout",
            "--records",
            "400",
            "--out",
            str(workspace / "null.jsonl.gz"),
        ).returncode
        == EXIT_OK
    )
    assert (
        dslm(
            "curate",
            "--corpus",
            str(workspace / "c.jsonl.gz"),
            "--out",
            str(workspace / "clean.jsonl.gz"),
        ).returncode
        == EXIT_OK
    )
    assert (
        dslm(
            "split",
            "--corpus",
            str(workspace / "clean.jsonl.gz"),
            "--out-dir",
            str(workspace / "splits"),
        ).returncode
        == EXIT_OK
    )
    assert (
        dslm(
            "train",
            "--splits",
            str(workspace / "splits"),
            "--out-dir",
            str(workspace / "model"),
            "--vocab-size",
            "512",
            "--epochs",
            "5",
        ).returncode
        == EXIT_OK
    )
    return workspace


class TestExitCodes:
    def test_a_clean_evaluation_exits_zero(self, built: Path):
        result = dslm(
            "evaluate",
            "--splits",
            str(built / "splits"),
            "--model",
            str(built / "model"),
            "--null",
            str(built / "null.jsonl.gz"),
            "--drop-shared",
        )
        assert result.returncode == EXIT_OK

    def test_a_refused_evaluation_exits_two(self, built: Path):
        # Two, not three. "A gate failed" and "the tool could not run" are
        # different facts and a pipeline that conflates them retries until green.
        result = dslm(
            "evaluate", "--splits", str(built / "splits"), "--model", str(built / "model")
        )
        assert result.returncode == EXIT_GATE_FAILED

    def test_a_missing_file_exits_three(self, built: Path):
        result = dslm("doctor", "--corpus", str(built / "absent.jsonl.gz"))
        assert result.returncode == EXIT_ERROR

    def test_an_unknown_command_exits_one(self):
        assert dslm("nonsense").returncode == EXIT_USAGE

    def test_a_missing_required_flag_exits_one(self):
        assert dslm("synth").returncode == EXIT_USAGE

    def test_version_exits_zero(self):
        result = dslm("--version")
        assert result.returncode == EXIT_OK
        assert "dslm" in result.stdout


class TestTheStreamSplit:
    def test_the_report_is_on_stdout_and_parses(self, built: Path):
        result = dslm(
            "evaluate",
            "--splits",
            str(built / "splits"),
            "--model",
            str(built / "model"),
            "--null",
            str(built / "null.jsonl.gz"),
            "--drop-shared",
        )
        document = json.loads(result.stdout)
        assert document["passed"] is True

    def test_the_commentary_is_on_stderr_where_it_cannot_corrupt_the_report(self, built: Path):
        result = dslm(
            "evaluate",
            "--splits",
            str(built / "splits"),
            "--model",
            str(built / "model"),
            "--null",
            str(built / "null.jsonl.gz"),
            "--drop-shared",
        )
        assert "contaminated" in result.stderr
        json.loads(result.stdout)

    def test_an_error_goes_to_stderr_with_its_remedy(self, built: Path):
        result = dslm("doctor", "--corpus", str(built / "absent.jsonl.gz"))
        assert result.stdout == ""
        assert "could not be opened" in result.stderr
        assert "dslm synth" in result.stderr

    def test_output_is_utf8_even_on_a_legacy_console(self, built: Path):
        # Windows consoles default to a legacy code page, and this project's
        # summaries contain en dashes. Mojibake in a JSON report is a report
        # that will not parse.
        result = dslm(
            "evaluate",
            "--splits",
            str(built / "splits"),
            "--model",
            str(built / "model"),
            "--null",
            str(built / "null.jsonl.gz"),
            "--drop-shared",
            "--markdown-out",
            str(built / "r.md"),
        )
        assert result.returncode == EXIT_OK
        assert "�" not in (built / "r.md").read_text(encoding="utf-8")


class TestTheWholePipelineFromAFreshDirectory:
    def test_synth_check_round_trips_through_a_real_process(self, tmp_path: Path):
        path = tmp_path / "c.jsonl.gz"
        assert dslm("synth", "--plan", "clean", "--out", str(path)).returncode == EXIT_OK
        assert dslm("check", "--plan", "clean", "--corpus", str(path)).returncode == EXIT_OK

    def test_a_corpus_that_drifted_is_caught(self, tmp_path: Path):
        path = tmp_path / "c.jsonl.gz"
        dslm("synth", "--plan", "clean", "--records", "300", "--out", str(path))
        result = dslm("check", "--plan", "clean", "--corpus", str(path))
        assert result.returncode == EXIT_ERROR
        assert "no longer matches" in result.stderr

    def test_classify_answers_from_the_built_model(self, built: Path):
        result = dslm(
            "classify",
            "--model",
            str(built / "model"),
            "--text",
            "shock strut pressure low at the gate, serviced with nitrogen",
        )
        assert result.returncode == EXIT_OK
        assert json.loads(result.stdout)[0]["label"] in range(100)

    def test_doctor_works_with_no_arguments_at_all(self):
        # The first thing anyone runs, and it must not need a file.
        result = dslm("doctor")
        assert result.returncode == EXIT_OK
        assert "self-check      ok" in result.stdout
