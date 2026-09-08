"""The task table. Run through ``python tasks.py <name>``.

Stdlib only, because a task runner that needs its own dependency installed
before it can install dependencies is a bootstrap problem nobody asked for.

``uv run`` picks the group each task actually needs rather than installing
everything: the documentation build gets a renderer and not pytest, and the
example commands get the project alone.
"""

from __future__ import annotations

from collections.abc import Sequence

IMAGE = "domain-slm"

EXAMPLES = "examples"
CORPUS = f"{EXAMPLES}/corpus.jsonl.gz"
CLEAN = f"{EXAMPLES}/clean.jsonl.gz"
NULL = f"{EXAMPLES}/holdout.jsonl.gz"
SPLITS = f"{EXAMPLES}/splits"
MODEL = f"{EXAMPLES}/model"
BASELINE = f"{EXAMPLES}/baseline.json"
CARD = f"{EXAMPLES}/DATA-CARD.md"


def _run(*args: str) -> list[str]:
    return ["uv", "run", *args]


def _dslm(*args: str) -> list[str]:
    return _run("python", "-m", "dslm", *args)


TASKS: dict[str, tuple[str, list[Sequence[str]]]] = {
    "setup": (
        "Install the project and its development tooling.",
        [["uv", "sync", "--locked", "--group", "dev", "--group", "docs"]],
    ),
    "fmt": ("Format.", [_run("ruff", "format", ".")]),
    "lint": (
        "Lint and check formatting.",
        [_run("ruff", "check", "."), _run("ruff", "format", "--check", ".")],
    ),
    "typecheck": ("Type check under mypy --strict.", [_run("mypy")]),
    "test": (
        "Run the whole test suite with the coverage gate.",
        [_run("pytest", "--cov", "--cov-report=term-missing", "--cov-fail-under=90")],
    ),
    "test-unit": ("Unit tests only.", [_run("pytest", "-m", "unit")]),
    "test-integration": ("Integration tests only.", [_run("pytest", "-m", "integration")]),
    "test-security": ("Security tests only.", [_run("pytest", "-m", "security")]),
    "test-e2e": (
        "End-to-end tests only: the CLI as a real process.",
        [_run("pytest", "-m", "e2e")],
    ),
    "test-meta": (
        "The gate's own negative controls: break one thing, assert it goes red.",
        [_run("pytest", "-m", "meta")],
    ),
    # -- the shipped example, rebuilt from nothing ------------------------
    "corpus": (
        "Regenerate the example corpus, its holdout and its data card.",
        [
            _dslm("synth", "--plan", "main", "--out", CORPUS),
            _dslm("synth", "--plan", "holdout", "--out", NULL),
            _dslm("curate", "--corpus", CORPUS, "--out", CLEAN, "--card", CARD),
            _dslm("split", "--corpus", CLEAN, "--out-dir", SPLITS),
        ],
    ),
    "corpus-check": (
        "Do the committed corpora still match their plans? The CI check.",
        [
            _dslm("check", "--plan", "main", "--corpus", CORPUS),
            _dslm("check", "--plan", "holdout", "--corpus", NULL),
        ],
    ),
    "train": (
        "Train the tokenizer and both models on the committed split.",
        [_dslm("train", "--splits", SPLITS, "--out-dir", MODEL)],
    ),
    "evaluate": (
        "Score both models, with the contamination and regression gates.",
        [
            _dslm(
                "evaluate",
                "--splits",
                SPLITS,
                "--model",
                MODEL,
                "--null",
                NULL,
                "--drop-shared",
                "--baseline",
                BASELINE,
                "--json-out",
                "reports/evaluation.json",
                "--junit-out",
                "reports/evaluation.xml",
                "--markdown-out",
                "reports/evaluation.md",
            )
        ],
    ),
    "baseline": (
        "Re-record the committed baseline. Commit the result on its own.",
        [
            _dslm(
                "evaluate",
                "--splits",
                SPLITS,
                "--model",
                MODEL,
                "--null",
                NULL,
                "--drop-shared",
                "--baseline",
                BASELINE,
                "--update-baseline",
            )
        ],
    ),
    "check-pipeline": (
        "Assert the gate fires on injected contamination and measures its rate.",
        [_run("python", "scripts/check-pipeline.py")],
    ),
    "contamination": (
        "Measure the overlap between the committed train and test splits.",
        [
            _dslm(
                "contamination",
                "--train",
                f"{SPLITS}/train.jsonl.gz",
                "--evaluate",
                f"{SPLITS}/test.jsonl.gz",
                "--null",
                NULL,
                "--drop-shared",
            )
        ],
    ),
    "doctor": (
        "Report what this installation would do, and self-check end to end.",
        [_dslm("doctor", "--corpus", CORPUS)],
    ),
    "examples": (
        "Run every example. They are documentation that executes.",
        [
            _run("python", "examples/quickstart.py"),
            _run("python", "examples/contamination_demo.py"),
            _run("python", "examples/tokenizer_demo.py"),
        ],
    ),
    "site": (
        "Build the documentation site into _site.",
        [_run("--only-group", "docs", "python", "scripts/build_site.py", "--output", "_site")],
    ),
    "security": (
        "Local security scans.",
        [
            _run("bandit", "-c", "pyproject.toml", "-r", "src", "-f", "screen"),
            [
                "uv",
                "export",
                "--locked",
                "--no-emit-project",
                "--no-hashes",
                "--output-file",
                "requirements.audit.txt",
            ],
            [
                "uv",
                "tool",
                "run",
                "pip-audit",
                "--strict",
                "--no-deps",
                "--requirement",
                "requirements.audit.txt",
            ],
        ],
    ),
    "docker-build": (
        "Build the container image.",
        [["docker", "build", "-t", f"{IMAGE}:local", "."]],
    ),
    "smoke": (
        "Build the image and run the smoke test against it.",
        [
            ["docker", "build", "-t", f"{IMAGE}:local", "."],
            ["bash", "scripts/smoke-test.sh", f"{IMAGE}:local"],
        ],
    ),
}

#: The order CI runs things in, cheapest gate first. Formatting and typing fail
#: in seconds; the container build takes minutes. A developer who broke an
#: import should learn that before the suite has finished collecting.
ALL = (
    "lint",
    "typecheck",
    "test",
    "corpus-check",
    "evaluate",
    "check-pipeline",
    "examples",
    "site",
)

# Expanded here rather than special-cased in the runner: tasks.py runs whatever
# command list it finds, and a task carrying an empty one would print nothing
# and exit 0. Nothing about that looks wrong on a terminal, which is what makes
# it worth catching.
TASKS["all"] = (
    "Everything CI runs, in the order CI runs it.",
    [step for name in ALL for step in TASKS[name][1]],
)
