#!/usr/bin/env python
"""The whole pipeline in one file, from nothing to a scored, gated result.

Documentation that executes. Run it:

    python examples/quickstart.py

Nothing here is read from disk and nothing reaches a network. Every artefact is
generated, so a fresh clone produces the same numbers on the first run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dslm.corpus import contamination as contam
from dslm.corpus.curate import apply_curation, find_duplicates
from dslm.corpus.split import split_corpus
from dslm.corpus.synth import PLANS, generate
from dslm.evaluate.metrics import evaluate, label_noise_ceiling, mcnemar
from dslm.model import features, linear, neural
from dslm.tokenizer import bpe


def heading(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


def main() -> int:
    heading("1. Generate a corpus, with provenance on every record")
    corpus = generate(PLANS["main"])
    print(f"  {len(corpus)} records, {len(corpus.labels)} chapters")
    print(f"  licences: {', '.join(corpus.licenses)}")
    print(f"  {corpus.metadata['note']}")

    heading("2. Find the near-duplicates before splitting, not after")
    curation = find_duplicates(corpus.records)
    print(f"  {curation.summary()}")
    cleaned = apply_curation(corpus, curation)

    heading("3. Split, keeping every duplicate cluster on one side")
    split = split_corpus(cleaned, seed=1, curation=find_duplicates(cleaned.records))
    print(f"  {split.summary()}")

    heading("4. Train a tokenizer on the training side only")
    tokenizer, report = bpe.train((r.text for r in split.train), vocab_size=1024)
    print(f"  {report.summary()}")
    measured = bpe.compression(tokenizer, (r.text for r in split.test))
    print(f"  {measured.summary()}")

    heading("5. Train the model, and the control it has to beat")
    space = features.label_space(split.train)
    encoded = {p: features.encode(split[p], tokenizer, space) for p in ("train", "dev", "test")}
    bayes = linear.train(encoded["train"], tokenizer.vocab_size)
    model, history = neural.train(
        encoded["train"],
        dev=encoded["dev"],
        vocab_size=tokenizer.vocab_size,
        padding_id=tokenizer.vocab[bpe.PADDING],
        tokenizer_digest=tokenizer.digest(),
    )
    print(f"  {history.summary()}")

    heading("6. Check the evaluation is a measurement, BEFORE reporting one")
    null = generate(PLANS["holdout"])
    calibration = contam.calibrate(split.train, null, drop_shared=True)
    contamination = contam.contamination_report(split.train, split.test, calibration=calibration)
    print(f"  {calibration.summary()}")
    print(f"  {contamination.summary()}")
    contam.enforce(contamination)
    print("  the gate passed, so the numbers below mean something")

    heading("7. Only now, the numbers")
    counts = encoded["test"].counts(tokenizer.vocab_size)
    predictions = {
        "naive-bayes": bayes.predict(counts),
        "neural": model.predict(encoded["test"].sequences, encoded["test"].lengths),
    }
    scored = {
        "naive-bayes": evaluate(
            "naive-bayes",
            predictions["naive-bayes"],
            bayes.predict_proba(counts),
            encoded["test"].targets,
            space,
            parameters=bayes.parameters,
        ),
        "neural": evaluate(
            "neural",
            predictions["neural"],
            model.predict_proba(encoded["test"].sequences, encoded["test"].lengths),
            encoded["test"].targets,
            space,
            parameters=model.parameters,
        ),
    }
    majority = max(split.test.label_counts().values()) / len(split.test)
    ceiling = label_noise_ceiling(split.test.records)
    print(f"  {'majority class':<14} {majority:>7.2%}")
    for name, metrics in sorted(scored.items(), key=lambda item: -item[1].accuracy):
        print(f"  {name:<14} {metrics.accuracy:>7.2%}   {metrics.parameters:>7,} parameters")
    if ceiling is not None:
        print(f"  {'ceiling':<14} {ceiling:>7.2%}   (4% of the labels are wrong on purpose)")

    heading("8. Is the difference between them real?")
    comparison = mcnemar(
        "naive-bayes",
        predictions["naive-bayes"],
        "neural",
        predictions["neural"],
        encoded["test"].targets,
    )
    print(f"  {comparison.summary()}")
    print()
    print("  The linear control matches the neural model with a fraction of the")
    print("  parameters. That is the finding, and publishing it is the point:")
    print("  a comparison nobody ran is not evidence that the complex thing won.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
