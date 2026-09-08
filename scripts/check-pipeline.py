#!/usr/bin/env python
"""Assert the contamination gate fires, and reports the rate it was given.

A gate that has only ever been observed passing is indistinguishable from a
`true` in a shell script. This runs the real command line against the shipped
corpus and asserts four things:

1. the correctly built pipeline **passes** — the control, without which every
   result below could be a fact about the tool rather than about the data;
2. contamination injected at a **known rate** makes it exit **2**, not merely
   non-zero — 3 would mean the tool broke;
3. the rate the gate *reports* is an estimate of the rate that was injected,
   within a stated tolerance. This is the difference between a flag and a
   measurement, and it is the claim the repository is built on;
4. asked to gate without a null corpus, it **refuses** rather than guessing.

Run through ``python tasks.py check-pipeline``. CI runs it on every push.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from dslm.corpus.record import build_corpus, read_corpus

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
SPLITS = EXAMPLES / "splits"
MODEL = EXAMPLES / "model"
NULL = EXAMPLES / "holdout.jsonl.gz"

EXIT_OK = 0
EXIT_GATE_FAILED = 2

#: injected fraction -> how far the reported rate may be from it.
INJECTIONS: dict[float, float] = {0.05: 0.02, 0.10: 0.03, 0.25: 0.03}


def dslm(*argv: str) -> tuple[int, str, str]:
    """Run the command line and return (exit code, stdout, stderr)."""
    result = subprocess.run(  # noqa: S603 - a fixed argument vector
        [sys.executable, "-m", "dslm", *argv],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
        cwd=ROOT,
        encoding="utf-8",
    )
    return result.returncode, result.stdout, result.stderr


def report_of(stdout: str, stderr: str) -> dict[str, Any]:
    try:
        return dict(json.loads(stdout))
    except json.JSONDecodeError:
        print(f"the gate produced no report:\n{stderr}", file=sys.stderr)
        raise


def evaluate(*extra: str) -> tuple[int, dict[str, Any]]:
    code, stdout, stderr = dslm(
        "evaluate",
        "--splits",
        str(SPLITS),
        "--model",
        str(MODEL),
        *extra,
    )
    return code, report_of(stdout, stderr)


def contaminated_split(rate: float, into: Path) -> Path:
    """Copy the shipped split, replacing *rate* of test with training records."""
    into.mkdir(parents=True, exist_ok=True)
    train = read_corpus(SPLITS / "train.jsonl.gz")
    dev = read_corpus(SPLITS / "dev.jsonl.gz")
    test = read_corpus(SPLITS / "test.jsonl.gz")

    count = int(len(test) * rate)
    leaked = [
        record.model_copy(update={"record_id": f"leak-{index:05d}"})
        for index, record in enumerate(train.records[:count])
    ]
    poisoned = build_corpus([*list(test.records)[count:], *leaked])

    train.write(into / "train.jsonl.gz")
    dev.write(into / "dev.jsonl.gz")
    poisoned.write(into / "test.jsonl.gz")
    return into


def main() -> int:
    failures = 0
    calibrated = ["--null", str(NULL), "--drop-shared"]

    # 1. The control.
    code, report = evaluate(*calibrated)
    if code == EXIT_OK and report["passed"]:
        measured = report["contamination"]["rate"]
        print(f"ok    the shipped pipeline passes; contamination {measured:.2%}")
    else:
        print(f"FAIL  the shipped pipeline did not pass: exit {code}", file=sys.stderr)
        failures += 1

    # 4. No null, no verdict. Checked early because it needs no fixture.
    code, report = evaluate()
    if code == EXIT_GATE_FAILED and not report.get("metrics"):
        print("ok    without a null corpus it refuses, and reports no metric at all")
    else:
        print(
            f"FAIL  uncalibrated run: exit {code}, metrics present: {'metrics' in report}",
            file=sys.stderr,
        )
        failures += 1

    # 2 and 3. Injected contamination is detected and measured.
    #
    # `var/` is gitignored, so on a fresh clone it does not exist and
    # TemporaryDirectory(dir=...) raises FileNotFoundError. Created here
    # rather than assumed: this script's whole job is to run on a checkout
    # nobody has run anything else in yet.
    scratch_root = ROOT / "var"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch_root) as scratch:
        for rate, tolerance in INJECTIONS.items():
            splits = contaminated_split(rate, Path(scratch) / f"inject-{int(rate * 100)}")
            code, stdout, stderr = dslm(
                "evaluate",
                "--splits",
                str(splits),
                "--model",
                str(MODEL),
                *calibrated,
            )
            report = report_of(stdout, stderr)
            measured = report["contamination"]["rate"]

            if code != EXIT_GATE_FAILED:
                print(
                    f"FAIL  {rate:.0%} injected: exit {code}, expected {EXIT_GATE_FAILED}",
                    file=sys.stderr,
                )
                failures += 1
            elif abs(measured - rate) > tolerance:
                print(
                    f"FAIL  {rate:.0%} injected: gate measured {measured:.2%}, "
                    f"outside the {tolerance:.0%} tolerance",
                    file=sys.stderr,
                )
                failures += 1
            elif report.get("metrics"):
                print(
                    f"FAIL  {rate:.0%} injected: a metric was reported anyway",
                    file=sys.stderr,
                )
                failures += 1
            else:
                print(
                    f"ok    {rate:.0%} injected: exit 2, measured {measured:.2%}, "
                    "no metric reported"
                )

    if failures:
        print(f"\n{failures} check(s) failed", file=sys.stderr)
        return 1
    print(
        f"\nthe gate passes clean data, refuses without a null, and measures "
        f"{len(INJECTIONS)} injected rate(s) to within tolerance"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
