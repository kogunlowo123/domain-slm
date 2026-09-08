# Wiring it into CI

Every stage is a command that exits non-zero when something is wrong.

| Exit | Meaning | What a pipeline should do |
| --- | --- | --- |
| 0 | Every gate that could be applied held | Continue |
| 1 | The command line was used wrongly | Fix the invocation |
| 2 | A gate failed — contamination, or a regression | **Fail the build** |
| 3 | The tool could not produce a verdict | **Fail the build** |

Both 2 and 3 fail, and they stay distinguishable. A contaminated corpus and a
missing file are different facts, and a pipeline that treats "non-zero" as one
thing is a pipeline where a broken step gets retried until it goes green.

**`argparse` exits 2 by default**, which this tool documents as *a gate failed* —
so the parser is subclassed to exit 1 instead. Without that, a mistyped flag and
a contaminated corpus would be indistinguishable to the thing most likely to be
reading. An end-to-end test caught it; nothing in-process could have.

## GitHub Actions

```yaml
- name: Evaluate, with the gates
  run: |
    uv run dslm evaluate \
      --splits splits/ --model model/ \
      --null holdout.jsonl.gz --drop-shared \
      --baseline baseline.json \
      --json-out reports/evaluation.json \
      --junit-out reports/evaluation.xml \
      --markdown-out reports/evaluation.md

- name: Publish the summary
  if: always()
  run: cat reports/evaluation.md >> "$GITHUB_STEP_SUMMARY"

- name: Upload the reports
  if: always()
  uses: actions/upload-artifact@v7
  with:
    name: evaluation-reports
    path: reports/
```

`if: always()` on both — the reports are most useful on the run that failed.

## The three report formats

### JUnit XML — the one that matters

```bash
dslm evaluate ... --junit-out reports/evaluation.xml
```

A failed gate appears next to the unit tests, with the same red mark, in
whatever dashboard the team already looks at. Nobody has to be taught a new tool.

One test case **per gate**, so `contamination` and `regression` are
distinguishable rather than both being "the evaluation job failed".

And **no baseline is a `skipped` case, never a passing one**. A green suite must
not be readable as "no regression" when nothing was compared.

### Markdown — the job summary

Verdict on the first line. When the contamination gate failed, the reason comes
first and **no accuracy is printed at all** — not a number with a warning above
it, because a warning above a number is read as a number.

### JSON — for anything else

```bash
jq -r '.passed' reports/evaluation.json
```

`passed` is the first key, so a script that greps rather than parses can answer
the only question that matters. The document carries the corpus, split and
tokenizer digests, so a report and the inputs that produced it can be matched up
months later without trusting a commit message.

## The baseline

A committed file recording what the models scored when somebody last looked. Its
purpose is to turn "the model got worse" from a thing somebody notices into a
thing the build says.

```bash
# Record it, in its own commit
dslm evaluate ... --baseline baseline.json --update-baseline --note "why"

# Compare against it, on every run
dslm evaluate ... --baseline baseline.json
```

Three properties worth knowing:

**One-sided with a tolerance.** A drop of more than 0.005 fails. Requiring the
exact number would fail on every platform whose BLAS accumulates differently —
[measured, and real](reproducibility.md).

**An improvement is reported and does not fail**, but is flagged as needing the
baseline updated. A silent improvement is how a baseline stops meaning anything.

**Comparing across a changed split is refused, not warned about.** The baseline
records the split digest, and two accuracies measured on different test sets are
not a comparison. Re-record the baseline in its own commit so the change of
split is visible in review.

## Choosing a null corpus in CI

The contamination gate needs one, and refuses to decide without it. In a
pipeline that usually means a file committed alongside the splits — data held
out before any of this ran, or generated, as this repository does.

`--drop-shared` handles the case where the null overlaps training on a handful
of records and **reports how many**. It is off by default because a null corpus
that silently shrinks is one nobody would think to question.

## Failing the build for the right reason

The trap worth naming: **a gate that has only ever been observed passing is
indistinguishable from `exit 0`.**

This repository's CI answers that with `scripts/check-pipeline.py`, which
asserts four things:

1. the correctly built pipeline **passes** — the control, without which every
   result below could be a fact about the tool;
2. contamination injected at a known rate makes it exit **2**, not merely
   non-zero;
3. the rate the gate *reports* is an estimate of the rate injected, within a
   stated tolerance;
4. asked to gate without a null, it **refuses** rather than guessing.

```
ok    the shipped pipeline passes; contamination 0.00%
ok    without a null corpus it refuses, and reports no metric at all
ok    5% injected: exit 2, measured 4.89%, no metric reported
ok    10% injected: exit 2, measured 9.96%, no metric reported
ok    25% injected: exit 2, measured 25.00%, no metric reported
```

Worth stealing: keep one input you know is bad, assert the gate rejects it, and
assert a known-good input at the same settings does not.

## Container

```bash
docker run --rm --network none \
  -v "$PWD/work:/app/work" \
  domain-slm:local \
  evaluate --splits work/splits --model work/model --null work/holdout.jsonl.gz
```

`--network none` is safe and is how the smoke test runs every command: this
package has no HTTP client and no socket at all.

The image runs as uid 10001, so a bind-mounted output directory must be writable
by it. On Linux a bind mount carries the host's ownership straight through, so
`chmod 0777` it once — Docker Desktop on Windows ignores ownership, which is why
this failure appears only in CI.

## What CI runs here

| Workflow | What it does |
| --- | --- |
| `ci.yml` | Lint, mypy, five test layers, coverage gate, the self-gate, examples, wheel |
| `security.yml` | gitleaks over full history, bandit, pip-audit, CodeQL, trivy |
| `docker.yml` | Build, trivy image scan, non-root check, smoke test inside the image |
| `pages.yml` | Build the documentation site and deploy it |

`python tasks.py all` runs the same sequence locally, cheapest gate first.

## See also

- [Contamination](contamination.md) — what the main gate measures
- [Reproducibility](reproducibility.md) — why the tolerance exists
