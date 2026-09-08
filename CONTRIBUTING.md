# Contributing

Thanks for taking the time to contribute. This document describes the local
workflow, the quality bar enforced in CI, and how changes are reviewed.

## Prerequisites

- Python 3.12 (the project pins `>=3.12,<3.13`)
- [uv](https://docs.astral.sh/uv/) 0.10 or newer
- Docker (optional, only needed for container work)

`make` is convenient on Linux and macOS but is not required. `python tasks.py`
is the cross-platform entry point and the source of truth; the `Makefile`
delegates to it.

## Getting set up

```bash
python tasks.py setup          # uv sync --locked --group dev --group docs
python tasks.py doctor         # proves the install works end to end
```

There is no credential to configure. This project reads files and writes
reports; it has no HTTP client and no socket anywhere in the package, and a test
asserts that over the source.

Never commit a populated `.env`. `.gitignore` excludes it and CI runs a secret
scan over both the working tree and the full git history.

## Development loop

| Task | Command |
| --- | --- |
| Format | `python tasks.py fmt` |
| Lint | `python tasks.py lint` |
| Type check | `python tasks.py typecheck` |
| Unit tests | `python tasks.py test-unit` |
| Integration tests | `python tasks.py test-integration` |
| Security tests | `python tasks.py test-security` |
| End-to-end tests | `python tasks.py test-e2e` |
| The gate's own negative controls | `python tasks.py test-meta` |
| Full suite + coverage gate | `python tasks.py test` |
| Regenerate the example corpus | `python tasks.py corpus` |
| Do the committed corpora still match? | `python tasks.py corpus-check` |
| Retrain the shipped model | `python tasks.py train` |
| Score it, with the gates | `python tasks.py evaluate` |
| Re-record the baseline | `python tasks.py baseline` |
| Assert the gate fires on injected contamination | `python tasks.py check-pipeline` |
| Run every example | `python tasks.py examples` |
| Documentation site | `python tasks.py site` |
| Local security scans | `python tasks.py security` |
| Container image + smoke test | `python tasks.py smoke` |
| Everything CI runs, in order | `python tasks.py all` |

Run `python tasks.py --list` for the full list.

## Quality bar

A change is mergeable when all of the following hold:

1. `ruff check` and `ruff format --check` are clean.
2. `mypy --strict` reports no errors.
3. The full test suite passes and line coverage is at least **90%**.
4. `python tasks.py check-pipeline` passes: the shipped pipeline is clean, the
   gate refuses without a null corpus, and contamination injected at 5%, 10% and
   25% is both caught and **measured to within tolerance**.
5. `python tasks.py corpus-check` passes: the committed corpora still match the
   plans that generated them.
6. `bandit` and `pip-audit` report no unresolved findings. If a dependency
   vulnerability has no upstream fix, add a justified, dated entry to
   `security/audit-exceptions.md` and the identifier to
   `security/audit-ignores.txt`.
7. No secret scanner finding, in the tree or in history.
8. Every example still runs. They are documentation that executes; one that
   stops working is a README that lies.
9. The documentation site builds. The builder fails on a broken internal link,
   so a renamed file fails the pull request rather than the deployment.

## Tests

Five layers, each runnable on its own:

- `unit` — pure logic, no I/O.
- `integration` — real files, the whole pipeline, and **the CLI in-process**.
- `security` — adversarial cases. **A failure here is a security regression**,
  not a bug.
- `e2e` — the command line as a real subprocess, so exit codes and the
  stdout/stderr split are actually asserted.
- `meta` — the gate's negative controls. See below.

New behaviour needs a test at the lowest layer that can express it. Tests that
assert nothing meaningful are rejected in review.

### Six rules specific to this repository

**The CLI needs an in-process layer as well as an end-to-end one.** A subprocess
is a different interpreter: invisible to coverage, with error paths nothing has
ever executed under assertion. Adding that layer here found a real bug in its
first run — `--records` and `--seed` were broken, because `Plan` is a slots
dataclass with no `__dict__` and the override used one.

**And an end-to-end layer as well as an in-process one.** Only a real process
shows the exit code the operating system sees. That layer caught `argparse`
exiting 2 on a usage error, colliding with this tool's *a gate failed*.

**A gate test needs a known answer, not just a known direction.** `tests/meta/`
injects contamination at 10%, 25% and 50% and asserts the gate reports
approximately those rates back. A gate that fires is worth something; a gate
whose number estimates the thing it measures is worth much more, and only a test
with a known answer can tell them apart.

**Every gate test needs a control at the same settings.** Without it, "the gate
fired" could be a fact about the tool rather than about the data.

**An optimisation needs the implementation it replaced.** `bpe.train` is a 35×
speedup of `bpe.train_reference`, and the test suite asserts they produce
identical merges. An optimisation whose only evidence is that it runs faster is
a rewrite nobody can trust.

**A parameter chosen by measurement gets the measurement asserted.** The shingle
width, the LSH banding and the calibration quantile were all chosen against
data, and the tests pin the properties that made the choice — not the choice
itself.

## Commits and pull requests

- Use [Conventional Commits](https://www.conventionalcommits.org/): `feat:`,
  `fix:`, `docs:`, `refactor:`, `test:`, `build:`, `ci:`, `chore:`.
- Keep the subject line under 72 characters and use the imperative mood.
- One logical change per pull request.
- Describe the behaviour change, the risk, and how you verified it.
- Update `CHANGELOG.md` under `## [Unreleased]` for anything user-visible.

## Changing the corpus or the model

The example artefacts in `examples/` are generated, and CI checks that the
corpora still match their plans. After changing the generator or a plan:

```bash
python tasks.py corpus            # regenerate the corpus, holdout, splits, card
python tasks.py corpus-check      # confirm they match
python tasks.py train             # retrain
python tasks.py baseline          # re-record, in its own commit
python tasks.py check-pipeline    # confirm the gate still behaves
```

Commit the regenerated artefacts in the same pull request, and the **baseline in
its own commit** so that a changed number is visible in review rather than
buried in a diff of regenerated files.

Corpora are written gzipped with a zero timestamp in the gzip member, so the
same corpus produces the same bytes and a diff means a real change.

## Changing a number this repository publishes

Several documented figures are asserted by tests — the model accuracies, the
tokenizer ratio, the contamination measurements, the shingle formula. That is
deliberate: a claim in a README that nothing checks is a claim that quietly
stops being true.

If your change moves one of them, the test fails. **Update the documentation in
the same pull request**, and say in the description what moved and why. Do not
widen the assertion to make it pass.

## Adding a model, a gate or a plan

- **A model** — it must be scored by the same `evaluate` path as the others and
  appear in the pairwise significance comparison. A model that is not compared
  is a model whose value nobody measured.
- **A gate** — it must be able to *refuse*, and there must be a `meta` test that
  makes it refuse, with a control. A gate that has only ever been observed
  passing is indistinguishable from `exit 0`.
- **A plan** — give it a distinct seed and a `shows` entry saying what it
  demonstrates, and add it to `corpus-check`.

## Reporting security issues

Do not open a public issue for a vulnerability. Follow the process in
[SECURITY.md](SECURITY.md).
