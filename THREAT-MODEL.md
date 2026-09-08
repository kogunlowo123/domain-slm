# Threat model

What this is trusted with, what it refuses to be trusted with, and what it does
not defend against.

## What this is

A command-line tool. It reads files, computes, writes reports, and exits. It
listens on no port, has no database, holds no state between runs, and — unlike
every other project in this series — **has no network path at all**. There is no
HTTP client, no socket, and no credential setting anywhere in the package. The
test suite asserts that over the source rather than over one run.

That removes most of the surface a reviewer looks for first. What remains is
about **data**: somebody else's files going in, and the integrity of the verdict
coming out.

## Assets

| Asset | Why it matters |
| --- | --- |
| The corpus | Somebody's operational text. Regulated in many domains. |
| Provenance and licence | A corpus whose licence cannot be established cannot be published. |
| **The verdict** | An evaluation that can be made to say "clean" is worse than none. |
| The runner | The process has a checkout and a token. |

The verdict is the unusual one, and it is the primary asset here. Everything
else in this document is ordinary input handling; the reason this project exists
is that a reported accuracy is a claim, and a claim that can be quietly weakened
is worse than no claim at all.

## Trust boundary

```
  untrusted ─────────────────────────────────┐   trusted
                                             │
  corpus file   ────────────────────────────▶│  Record validation (pydantic, extra=forbid)
  tokenizer     ────────────────────────────▶│  Digest checked against its own contents
  model dir     ────────────────────────────▶│  Shapes, tokenizer digest, label space checked
  baseline      ────────────────────────────▶│  Schema checked; split digest must match
  environment   ────────────────────────────▶│  Settings; unknown names and sections refused
                                             │
                                             ├─▶ stdout / files
                                             └─▶ exit code
```

Nothing crosses that boundary outward except a report and an exit code.

---

## T1 — The verdict is quietly weakened

The threat this project is built around. Five ways it happens, and what stops
each.

| How | Control |
| --- | --- |
| The evaluation set overlaps training | The contamination gate, calibrated against a null (ADR-001) |
| The threshold is meaningless for this data | Refuses to gate without a null rather than guessing |
| The check could not be applied | Coverage floor: too-short records fail the gate, never pass it |
| No baseline exists | Reported as `skipped`, never as a passing case |
| A model stops being measured | A model missing from a comparison fails the regression gate |

**The strongest form of the control:** when the gate fails, the metric is *not
computed and withheld* — it does not appear in the document at all. A number
that exists is a number somebody quotes out of context. `tests/security/` asserts
that the JSON has no `metrics` key and the Markdown contains no accuracy, and
includes a test that would fail if the refusal were removed.

**Residual risk.** Somebody can edit the objectives of the evaluation — lower a
tolerance, drop a model — in a pull request. That is a review problem, and the
baseline's recorded digests are what make it visible in a diff.

---

## T2 — Operational text leaks through the corpus schema

**The realistic version.** Not a bug in a redactor: an engineer adds a field to
carry the technician's free-text note, or the customer's name, because the
debugging was hard.

**Control.** A record has `record_id`, `text`, `label`, `source`, `license` and
a bounded string metadata bag. `extra="forbid"` refuses anything else at
validation — `prompt`, `email`, `user_id` and four more are asserted refused in
the security tests. `text` is bounded at 4,096 characters, so the field that
does exist cannot become a transcript by accident.

**Residual risk.** `text` is the work order, and a work order can contain
whatever somebody typed into it. This tool cannot fix that, and does not claim
to; what it does is refuse to grow *new* places for it.

---

## T3 — A hostile or damaged corpus file

| Attack | Control |
| --- | --- |
| Truncated mid-line | One JSON document per line; unreadable lines counted and reported |
| A 200 MB line | `MAX_LINE_BYTES` (128 KB) — that is not a record |
| A 40 GB corpus | `MAX_CORPUS_BYTES` (512 MB), checked before reading |
| **A small gzip that expands to 40 GB** | The same limit applied to the *decompressed stream* |
| A record from a newer schema | Refused by version, with the version named |
| Code in a text field | It is data: `json.loads`, never `eval`, and nothing is executed |

The decompression bomb is the one worth naming. Accepting `.gz` input means
accepting it, and checking `stat()` alone would let one through — the runner is
then killed by the OOM killer, which reaches an operator as "the build is
flaky" rather than as an attack.

**Residual risk.** A corpus can contain plausible but false records. No amount
of validation detects a lying producer; see T5.

---

## T4 — A model artefact is not what it claims to be

**The realistic version.** A model directory assembled from two runs, or a
tokenizer regenerated after the model was trained. Neither produces an error —
token 412 simply means a different string, and the result is an accuracy several
points lower that reads as a bad model rather than as a mismatch.

**Controls.** Three, all at load time: the metadata and the weights must agree
about shapes; the model's recorded tokenizer digest must match the tokenizer
supplied; and the label space must be present, because without it a predicted
index cannot be turned back into a chapter.

**Not a pickle.** Weights are `np.savez` arrays and JSON metadata. Loading a
pickle from an untrusted path executes whatever it was told to; `allow_pickle`
is left at its safe default and the security tests assert the string does not
appear in the module.

---

## T5 — The data is wrong rather than malformed

Out of scope, and stated rather than implied. If 4% of the labels are wrong, the
model learns from wrong labels and the reported accuracy is capped — which is
exactly what the shipped corpus does on purpose, so that the ceiling is visible
in every report rather than assumed away.

What the tool does do is refuse to *pretend*: unreadable lines are counted, the
label-noise ceiling is reported beside the accuracy when the corpus records one,
and a difference between two models is reported with a p-value rather than as a
winner.

---

## T6 — Supply chain

| Concern | Control |
| --- | --- |
| Dependency count | Five: numpy, pydantic, pydantic-settings, structlog, pyyaml |
| Transitive drift | `uv.lock`, committed; CI installs with `--locked` |
| Known vulnerabilities | `pip-audit --strict --no-deps` over the exported lock |
| Static analysis | `bandit` over `src/`, and CodeQL on push |
| Secrets in history | `gitleaks` over the full history, not only the diff |
| A deep-learning framework's transitive tree | Avoided entirely (ADR-007) |

Exceptions are recorded in `security/audit-exceptions.md` with a reason and a
date, never as a bare ignore.

---

## T7 — The CI runner

The tool executes no subprocess, evaluates no expression from a file, and
imports nothing named by configuration. YAML is not used for input at all here;
the corpus is JSON Lines and the model is `.npz`.

Workflow permissions are least-privilege per job, and the container runs as a
non-root user with no build toolchain in the final layer.

---

## Not defended against

- **A malicious maintainer.** Commit signing and branch protection live outside
  this repository.
- **A corpus you are not allowed to have.** The tool records the licence you
  give it; it cannot verify that you were entitled to the text.
- **Adversarial examples against the classifier.** It is a bag-of-embeddings
  chapter classifier, not a security control, and nothing here treats its output
  as one.
- **Denial of service.** There is no service. The size limits bound a single
  run, and nothing else is claimed.

## Reporting

See [SECURITY.md](SECURITY.md). Please do not open a public issue for a
vulnerability.
