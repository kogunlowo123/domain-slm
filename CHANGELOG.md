# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [0.1.0] — 2026-09-08

First release. Everything below is new.

### The contamination gate

- `dslm evaluate` measures how much of the evaluation set also appears in
  training **before** computing any metric, and above the limit reports no
  metric at all — no `metrics` key in the JSON, no accuracy in the Markdown, a
  failing case in the JUnit XML.
- The threshold is **calibrated against a null corpus**, not fixed (ADR-001).
  The published "13-gram containment >= 0.5" rule scores provably disjoint data
  at 1.80% against the genuine test split's 1.25% on this corpus — it cannot
  order the two, so a threshold on its output decides on noise.
- With a calibrated threshold the reported rate becomes an estimate: 0%, 5%,
  10% and 25% injected contamination measure 0.00%, 4.89%, 9.96% and 25.00%.
- Without a null corpus the tool **refuses to gate** rather than guessing.
  `--allow-uncalibrated` opts back in deliberately.
- A check that could not be applied is never reported as passed: a coverage
  floor fails a split whose records are mostly too short to examine.

### Curation and splitting

- Near-duplicate detection with MinHash and LSH banding, with the shingle width
  and threshold **derived from the record length** rather than copied (ADR-005).
  The conventional 5-word shingle at 0.8 cannot detect a one-word edit at all on
  a 21-word record.
- The banding is chosen to match the threshold: 32 bands of 4 rows rather than
  the textbook 16 x 8, which took recall over planted duplicates from 0.950 to
  **1.000**.
- Curation reports; removal is a separate, opt-in act (ADR-009).
- The data card is generated from the records, never written by hand.
- Splits keep every duplicate cluster on one side, with optional grouping by a
  metadata key, merged transitively.

### The models

- A neural classifier — token embeddings, mean pooling, one hidden layer —
  written out in NumPy. 55,020 parameters; trains in five seconds on one CPU
  core.
- A multinomial naive Bayes control, trained and scored **always** (ADR-002),
  and compared with McNemar's exact test.
- The result, published rather than hidden: naive Bayes 92.03% against the
  neural model's 92.39% with a fifth of the parameters, p = 0.79 —
  statistically indistinguishable.
- Early stopping on the dev split, keeping the best epoch's parameters and
  reporting which. Weight decay at 1e-3, chosen on dev and worth two points:
  without it the model reaches 100% training accuracy and 83.5% dev by
  memorising the corpus's label noise.

### The tokenizer

- Byte-pair encoding trained from scratch, with a byte-level fallback so the
  unknown token is never emitted.
- 2.37x fewer tokens per record than a same-sized tokenizer trained on
  out-of-domain text.
- Incremental pair counting: a 35x speedup over the straightforward
  implementation, which is kept in the module and asserted to produce identical
  merges.

### Reproducibility

- Measured rather than assumed (ADR-003). Corpus, split and tokenizer are
  exactly reproducible and gated by **digest**; model weights are not, and are
  gated on **metrics within a tolerance**.
- Uniform initialisation rather than normal: `standard_normal`'s ziggurat tail
  calls `log`, and libm's `log` differs by one ULP between platforms — one value
  in 32,768, which then propagates through every weight.
- With that fixed, training still diverges: after 30 steps the loss is
  bit-identical and 1,344 of 55,872 weights differ by at most 2.4e-14 relative.

### The corpus

- `dslm synth` generates aircraft maintenance work orders from a seeded grammar,
  with `source` and `license` as **required fields** on every record.
- Duplicates and label noise are planted at known rates, so the deduplicator and
  the gate can be asserted to find a known answer.
- The task is deliberately hard: chapters share vocabulary and 4% of records are
  mislabelled, which puts a real ceiling (96.1%) under any accuracy claim.
- `dslm check` verifies a committed corpus still matches its plan, by digest.

### Reports and exit codes

- JSON with `passed` as the first key, JUnit XML where an uncompared baseline is
  a `skipped` case rather than a passing one, and a Markdown job summary.
- Exit **0** held, **1** usage, **2** a gate failed, **3** could not run — and
  `argparse` is subclassed so that a usage error does not exit 2 and collide
  with "a gate failed" (ADR-008).

### Quality

- 345 tests across five layers — unit, integration, security, end-to-end and
  meta — at 95.79% line coverage against a 90% gate.
- `scripts/check-pipeline.py` asserts the gate passes clean data, refuses
  without a null, and measures three injected contamination rates to within
  tolerance. CI runs it on every push.
- CI: ruff, mypy `--strict`, five test layers, a coverage gate, the self-gate,
  every example, a wheel that is installed and invoked, gitleaks over the full
  history, bandit, pip-audit, CodeQL, trivy, and a container smoke test that
  runs the gates inside the built image with the network removed.

[Unreleased]: https://github.com/kogunlowo123/domain-slm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kogunlowo123/domain-slm/releases/tag/v0.1.0
