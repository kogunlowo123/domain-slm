"""The command line.

Nine commands, one pipeline::

    synth  ->  curate  ->  split  ->  train  ->  evaluate  ->  classify
    check          the committed corpus still matches its plan
    contamination  inspect the overlap without running an evaluation
    doctor         what this installation would do, end to end

Exit codes are the interface: **0** held, **1** the command line was used
wrongly, **2** a gate failed, **3** the tool could not produce a verdict. A
contaminated corpus and a broken installation are different facts and a pipeline
that cannot tell them apart gets retried until it goes green.

The report goes to **stdout** and everything else to **stderr**, so a report can
be piped without a log line landing in the middle of it.

Only this module knows about ``argparse``, ``sys.exit`` or where files live.
That is what makes every command testable in-process as well as end to end —
a subprocess is a different interpreter, invisible to coverage, and a CLI
exercised only end to end has error paths nothing ever executes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

from dslm import __version__
from dslm.config import Settings, load
from dslm.corpus import contamination as contam
from dslm.corpus.curate import (
    DEFAULT_THRESHOLD as DUPLICATE_THRESHOLD,
)
from dslm.corpus.curate import (
    apply_curation,
    data_card,
    find_duplicates,
    render_data_card,
)
from dslm.corpus.record import Corpus, Record, build_corpus, read_corpus
from dslm.corpus.split import PARTS, split_corpus
from dslm.corpus.synth import PLANS, generate
from dslm.errors import (
    EXIT_ERROR,
    EXIT_GATE_FAILED,
    EXIT_OK,
    EXIT_USAGE,
    ContaminationError,
    DslmError,
)
from dslm.evaluate import baseline as baselines
from dslm.evaluate import metrics as metrics_module
from dslm.evaluate.report import Evaluation, render_json, render_junit, render_markdown
from dslm.logging import configure
from dslm.model import features, io, linear, neural
from dslm.tokenizer import bpe

#: Filenames inside a split directory. Fixed, so that `--splits <dir>` is the
#: whole interface and three paths do not have to be threaded through every
#: command.
SPLIT_FILES = {part: f"{part}.jsonl.gz" for part in PARTS}
TOKENIZER_NAME = "tokenizer.json"


def _use_utf8() -> None:
    """Force UTF-8 on stdout and stderr.

    Windows consoles default to a legacy code page, and this project's reports
    contain en dashes and arrows. Without this they arrive as mojibake, which in
    a JSON report means output that will not parse.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _note(message: str) -> None:
    """A line of commentary, on stderr where it cannot corrupt a report."""
    print(message, file=sys.stderr)


def _read_split(directory: Path) -> dict[str, Corpus]:
    parts: dict[str, Corpus] = {}
    for part, name in SPLIT_FILES.items():
        parts[part] = read_corpus(directory / name)
    return parts


# -- commands ------------------------------------------------------------


def _command_synth(args: argparse.Namespace, _settings: Settings) -> int:
    plan = PLANS[args.plan]
    # `dataclasses.replace`, not `type(plan)(**plan.__dict__)`: Plan uses
    # slots, so it has no __dict__ at all and the obvious form raises
    # AttributeError the first time anyone passes --records.
    overrides: dict[str, Any] = {}
    if args.records is not None:
        overrides["records"] = args.records
    if args.seed is not None:
        overrides["seed"] = args.seed
    if overrides:
        plan = replace(plan, **overrides)

    corpus = generate(plan)
    corpus.write(args.out)
    _note(f"generated {len(corpus)} record(s): {plan.describe()}")
    print(
        json.dumps(
            {
                "plan": plan.name,
                "seed": plan.seed,
                "records": len(corpus),
                "labels": list(corpus.labels),
                "digest": corpus.digest(),
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return EXIT_OK


def _command_check(args: argparse.Namespace, _settings: Settings) -> int:
    """Does a committed corpus still match the plan that generated it?

    By digest, not by regenerating and diffing the file. The digest is over what
    a model learns from — the text and the label of each record — so reordering
    the file or adding a key to its metadata does not move it, and a check that
    fired on those is a check people learn to ignore.
    """
    corpus = read_corpus(args.corpus)
    expected = generate(PLANS[args.plan])
    if corpus.digest() != expected.digest():
        raise DslmError(
            f"{Path(args.corpus).name} no longer matches plan {args.plan!r}.",
            remedy=(
                f"The file holds {len(corpus)} record(s) with digest {corpus.digest()[:23]}... "
                f"and the plan generates {len(expected)} with {expected.digest()[:23]}.... "
                f"Regenerate with 'dslm synth --plan {args.plan} --out {args.corpus}', and "
                "commit the result in the same change as whatever altered the generator."
            ),
        )
    print(f"{Path(args.corpus).name} matches plan {args.plan!r}: {len(corpus)} record(s)")
    return EXIT_OK


def _command_curate(args: argparse.Namespace, _settings: Settings) -> int:
    corpus = read_corpus(args.corpus)
    curation = find_duplicates(corpus.records, threshold=args.threshold)
    _note(curation.summary())

    card = data_card(corpus, curation)
    if args.card:
        _write(Path(args.card), render_data_card(card))
        _note(f"data card written to {args.card}")

    if args.out:
        cleaned = apply_curation(corpus, curation)
        cleaned.write(args.out)
        _note(f"{len(cleaned)} record(s) kept, written to {args.out}")

    print(json.dumps({"duplicates": curation.as_dict(), "data_card": card}, indent=2))
    return EXIT_OK


def _command_split(args: argparse.Namespace, _settings: Settings) -> int:
    corpus = read_corpus(args.corpus)
    # Grouping is computed here rather than taken on trust. A caller who
    # deduplicated first still has *related* records, and those must not
    # straddle the split; see dslm.corpus.split.
    curation = find_duplicates(corpus.records, threshold=args.threshold)
    split = split_corpus(
        corpus,
        seed=args.seed,
        ratios=(args.train, args.dev, args.test),
        curation=curation,
        group_by=args.group_by,
    )
    for part, name in SPLIT_FILES.items():
        split[part].write(Path(args.out_dir) / name)
    _note(split.summary())
    print(json.dumps(split.as_dict(), indent=2))
    return EXIT_OK


def _command_tokenizer(args: argparse.Namespace, settings: Settings) -> int:
    corpus = read_corpus(args.corpus)
    size = args.vocab_size if args.vocab_size is not None else settings.train.vocab_size
    tokenizer, report = bpe.train((record.text for record in corpus), vocab_size=size)
    tokenizer.save(args.out)
    compression = bpe.compression(tokenizer, (record.text for record in corpus))
    _note(f"{report.summary()}; {compression.summary()}")
    print(
        json.dumps(
            {
                "training": report.as_dict(),
                "compression": compression.as_dict(),
                "digest": tokenizer.digest(),
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return EXIT_OK


def _command_train(args: argparse.Namespace, settings: Settings) -> int:
    """Train the tokenizer and both models from a split directory.

    Both models, always. The neural model's value is a *comparison*, and a
    command that trained only the neural one would make the comparison optional
    — which is how a linear control quietly stops being run.
    """
    parts = _read_split(Path(args.splits))
    out = Path(args.out_dir)
    vocab = args.vocab_size if args.vocab_size is not None else settings.train.vocab_size
    epochs = args.epochs if args.epochs is not None else settings.train.epochs
    seed = args.seed if args.seed is not None else settings.train.seed

    tokenizer, tokenizer_report = bpe.train(
        (record.text for record in parts["train"]), vocab_size=vocab
    )
    tokenizer.save(out / TOKENIZER_NAME)

    space = features.label_space(parts["train"])
    encoded = {
        part: features.encode(corpus, tokenizer, space, max_tokens=settings.train.max_tokens)
        for part, corpus in parts.items()
    }

    bayes = linear.train(encoded["train"], tokenizer.vocab_size)
    model, history = neural.train(
        encoded["train"],
        dev=encoded["dev"],
        epochs=epochs,
        seed=seed,
        vocab_size=tokenizer.vocab_size,
        padding_id=tokenizer.vocab[bpe.PADDING],
        tokenizer_digest=tokenizer.digest(),
    )
    io.save(model, out, extra={"history": history.as_dict(), "vocab_size": vocab})

    _note(f"tokenizer: {tokenizer_report.summary()}")
    _note(f"neural: {history.summary()}")
    print(
        json.dumps(
            {
                "tokenizer": {
                    "digest": tokenizer.digest(),
                    **tokenizer_report.as_dict(),
                },
                "naive_bayes": bayes.as_dict(),
                "neural": model.as_dict(),
                "history": history.as_dict(),
                "out": str(out),
            },
            indent=2,
        )
    )
    return EXIT_OK


def _score_models(
    parts: dict[str, Corpus], out: Path, settings: Settings, part: str
) -> tuple[dict[str, metrics_module.Metrics], dict[str, Any], Any]:
    """Train the control, load the neural model, and score both on *part*."""
    tokenizer = bpe.load(out / TOKENIZER_NAME)
    model, _ = io.load(out, tokenizer_digest=tokenizer.digest())
    space = model.label_space
    encoded = {
        name: features.encode(corpus, tokenizer, space, max_tokens=settings.train.max_tokens)
        for name, corpus in parts.items()
    }

    # The control is retrained here rather than stored. It is a closed-form fit
    # over counts and takes milliseconds, and storing it would add an artefact
    # that could go stale against the tokenizer beside it.
    bayes = linear.train(encoded["train"], tokenizer.vocab_size)

    target = encoded[part]
    counts = target.counts(tokenizer.vocab_size)
    predictions = {
        "naive-bayes": (bayes.predict(counts), bayes.predict_proba(counts), bayes.parameters),
        "neural": (
            model.predict(target.sequences, target.lengths),
            model.predict_proba(target.sequences, target.lengths),
            model.parameters,
        ),
    }
    scored = {
        name: metrics_module.evaluate(
            name, predicted, probabilities, target.targets, space, parameters=count
        )
        for name, (predicted, probabilities, count) in predictions.items()
    }
    return scored, {name: value[0] for name, value in predictions.items()}, (tokenizer, target)


def _command_evaluate(  # noqa: PLR0912, PLR0915 - the gate sequence is the command
    args: argparse.Namespace, settings: Settings
) -> int:
    parts = _read_split(Path(args.splits))
    out = Path(args.model)
    part = args.on

    # The contamination check runs *first*. Metrics computed on a contaminated
    # split are not reported at all, so they must not be computed and then
    # withheld — a number that exists is a number somebody quotes.
    calibration = None
    refused = ""
    if args.null:
        calibration = contam.calibrate(
            parts["train"],
            read_corpus(args.null),
            quantile=settings.gate.quantile,
            drop_shared=args.drop_shared,
        )
        _note(calibration.summary())
    report = contam.contamination_report(parts["train"], parts[part], calibration=calibration)
    _note(report.summary())

    try:
        contam.enforce(
            report,
            max_contaminated=settings.gate.max_contaminated,
            min_coverage=settings.gate.min_coverage,
            allow_uncalibrated=args.allow_uncalibrated,
        )
    except ContaminationError as exc:
        refused = f"{exc.message} {exc.remedy}".strip()

    scored: dict[str, metrics_module.Metrics] = {}
    comparisons: tuple[metrics_module.Comparison, ...] = ()
    confusions: tuple[tuple[int, int, int], ...] = ()
    ceiling = None
    tokenizer_digest = ""
    verdict = None

    if not refused:
        scored, predicted, (tokenizer, target) = _score_models(parts, out, settings, part)
        tokenizer_digest = tokenizer.digest()
        names = sorted(scored)
        comparisons = tuple(
            metrics_module.mcnemar(left, predicted[left], right, predicted[right], target.targets)
            for index, left in enumerate(names)
            for right in names[index + 1 :]
        )
        best = max(scored.values(), key=lambda score: score.accuracy)
        confusions = tuple(
            metrics_module.most_confused(predicted[best.name], target.targets, target.label_space)
        )
        ceiling = metrics_module.label_noise_ceiling(parts[part].records)
        for score in scored.values():
            _note(score.summary())
        for comparison in comparisons:
            _note(comparison.summary())

    split_digest = "sha256:" + "".join(
        corpus.digest().removeprefix("sha256:")[:8] for corpus in parts.values()
    )

    if not refused and args.baseline:
        path = Path(args.baseline)
        if args.update_baseline:
            record = baselines.from_metrics(
                scored,
                corpus_digest=parts["train"].digest(),
                split_digest=split_digest,
                tokenizer_digest=tokenizer_digest,
                contamination_rate=report.rate,
                contamination_calibrated=report.calibrated,
                recorded_at=datetime.now(UTC).isoformat(timespec="seconds"),
                note=args.note,
            )
            record.save(path)
            _note(f"baseline written to {path}")
        elif path.is_file():
            verdict = baselines.compare(
                baselines.load(path),
                scored,
                tolerance=settings.gate.tolerance,
                split_digest=split_digest,
            )
            _note(verdict.summary())
        else:
            _note(f"no baseline at {path}; nothing was compared")

    evaluation = Evaluation(
        metrics=scored,
        contamination=report,
        verdict=verdict,
        comparisons=comparisons,
        corpus_digest=parts["train"].digest(),
        split_digest=split_digest,
        tokenizer_digest=tokenizer_digest,
        refused=refused,
        ceiling=ceiling,
        confusions=confusions,
    )

    if args.json_out:
        _write(Path(args.json_out), render_json(evaluation))
    else:
        print(render_json(evaluation), end="")
    if args.junit_out:
        _write(Path(args.junit_out), render_junit(evaluation))
    if args.markdown_out:
        _write(Path(args.markdown_out), render_markdown(evaluation))

    return EXIT_OK if evaluation.passed else EXIT_GATE_FAILED


def _command_contamination(args: argparse.Namespace, settings: Settings) -> int:
    train = read_corpus(args.train)
    evaluate = read_corpus(args.evaluate)
    calibration = None
    if args.null:
        calibration = contam.calibrate(
            train,
            read_corpus(args.null),
            quantile=settings.gate.quantile,
            drop_shared=args.drop_shared,
        )
        _note(calibration.summary())
    report = contam.contamination_report(train, evaluate, calibration=calibration)
    _note(report.summary())
    print(json.dumps(report.as_dict(), indent=2))

    try:
        contam.enforce(
            report,
            max_contaminated=settings.gate.max_contaminated,
            min_coverage=settings.gate.min_coverage,
            allow_uncalibrated=args.allow_uncalibrated,
        )
    except ContaminationError as exc:
        _note(f"{exc.message}\n  {exc.remedy}")
        return EXIT_GATE_FAILED
    return EXIT_OK


def _command_classify(args: argparse.Namespace, settings: Settings) -> int:
    out = Path(args.model)
    tokenizer = bpe.load(out / TOKENIZER_NAME)
    model, _ = io.load(out, tokenizer_digest=tokenizer.digest())
    texts = list(args.text)
    if args.corpus:
        texts.extend(record.text for record in read_corpus(args.corpus))
    if not texts:
        raise DslmError(
            "there is nothing to classify.",
            remedy="Pass --text once or more, or --corpus <path>.",
        )

    records = [
        Record(
            record_id=f"input-{index:06d}",
            text=text,
            label=model.label_space.label_at(0),
            source="<stdin>",
            license="not-applicable",
        )
        for index, text in enumerate(texts)
    ]
    encoded = features.encode(
        build_corpus(records), tokenizer, model.label_space, max_tokens=settings.train.max_tokens
    )
    probabilities = model.predict_proba(encoded.sequences, encoded.lengths)
    predictions = probabilities.argmax(axis=1)

    results = [
        {
            "text": text,
            "label": model.label_space.label_at(int(index)),
            "confidence": round(float(probabilities[row, index]), 4),
        }
        for row, (text, index) in enumerate(zip(texts, predictions, strict=True))
    ]
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return EXIT_OK


def _command_doctor(args: argparse.Namespace, settings: Settings) -> int:
    """What this installation would do, end to end, with no files at all."""
    lines = [f"dslm {__version__}", ""]
    lines.append(f"plans           {', '.join(sorted(PLANS))}")
    lines.append("models          naive-bayes, neural")
    lines.append("")
    lines.append(f"tolerance       {settings.gate.tolerance:g}")
    lines.append(f"max contaminated {settings.gate.max_contaminated:g}")
    lines.append(f"quantile        {settings.gate.quantile:g}")
    lines.append(f"vocab / epochs  {settings.train.vocab_size} / {settings.train.epochs}")
    lines.append(f"log             {settings.log.level} as {settings.log.format}, to stderr")
    lines.append("")

    # The self-check: a whole pipeline on a small generated corpus. Proves the
    # installation can generate, curate, split, tokenise, train and score
    # without any file on disk or any network.
    plan = type(PLANS["main"])(name="doctor", records=600, seed=7)
    corpus = generate(plan)
    curation = find_duplicates(corpus.records, threshold=DUPLICATE_THRESHOLD)
    cleaned = apply_curation(corpus, curation)
    split = split_corpus(cleaned, seed=1, curation=find_duplicates(cleaned.records))
    tokenizer, _ = bpe.train((record.text for record in split.train), vocab_size=512)
    space = features.label_space(split.train)
    encoded = features.encode(split.train, tokenizer, space)
    scored = features.encode(split.test, tokenizer, space)
    bayes = linear.train(encoded, tokenizer.vocab_size)
    accuracy = neural.accuracy(bayes.predict(scored.counts(tokenizer.vocab_size)), scored.targets)
    lines.append(
        f"self-check      ok: {len(corpus)} generated, {len(cleaned)} after curation, "
        f"control scored {accuracy:.3f} on {len(split.test)} held-out record(s)"
    )

    if args.corpus:
        inspected = read_corpus(args.corpus)
        lines.extend(
            [
                "",
                f"corpus          {args.corpus}",
                f"records         {len(inspected)}",
                f"labels          {', '.join(str(label) for label in inspected.labels)}",
                f"licences        {', '.join(inspected.licenses)}",
                f"unreadable      {len(inspected.unreadable)} line(s)",
                f"digest          {inspected.digest()}",
            ]
        )

    print("\n".join(lines))
    return EXIT_OK


# -- wiring --------------------------------------------------------------


class _Parser(argparse.ArgumentParser):
    """An argument parser that exits 1 on a usage error, not 2.

    argparse's default is 2, which this tool documents as *a gate failed* —
    so a mistyped flag and a contaminated corpus would be indistinguishable
    to the thing most likely to be reading: a CI job. The exit codes are the
    interface, so argparse conforms to them rather than the reverse.
    """

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915 - nine subcommands
    """Build the argument parser."""
    parser = _Parser(
        prog="dslm",
        description="A small domain language model you can trust.",
    )
    parser.add_argument("--version", action="version", version=f"dslm {__version__}")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None)
    # parser_class so every subcommand inherits the exit-code behaviour too:
    # `dslm synth` with no --out is as much a usage error as `dslm nonsense`.
    sub = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)

    synth = sub.add_parser("synth", help="generate a deterministic domain corpus")
    synth.add_argument("--plan", choices=sorted(PLANS), default="main")
    synth.add_argument("--out", required=True, type=Path, help="where to write the corpus")
    synth.add_argument("--records", type=int, default=None, help="override the plan's size")
    synth.add_argument("--seed", type=int, default=None, help="override the plan's seed")
    synth.set_defaults(handler=_command_synth)

    check = sub.add_parser("check", help="does a committed corpus still match its plan?")
    check.add_argument("--plan", choices=sorted(PLANS), required=True)
    check.add_argument("--corpus", required=True, type=Path)
    check.set_defaults(handler=_command_check)

    curate = sub.add_parser("curate", help="find near-duplicates and write a data card")
    curate.add_argument("--corpus", required=True, type=Path)
    curate.add_argument("--out", type=Path, default=None, help="write the deduplicated corpus")
    curate.add_argument("--card", type=Path, default=None, help="write the data card here")
    curate.add_argument(
        "--threshold",
        type=float,
        default=DUPLICATE_THRESHOLD,
        help="Jaccard similarity at which two records are one (default: %(default)s)",
    )
    curate.set_defaults(handler=_command_curate)

    split = sub.add_parser("split", help="divide a corpus into train, dev and test")
    split.add_argument("--corpus", required=True, type=Path)
    split.add_argument("--out-dir", required=True, type=Path)
    split.add_argument("--seed", type=int, default=1)
    split.add_argument("--train", type=float, default=0.8)
    split.add_argument("--dev", type=float, default=0.1)
    split.add_argument("--test", type=float, default=0.1)
    split.add_argument(
        "--threshold",
        type=float,
        default=DUPLICATE_THRESHOLD,
        help="group records this similar onto the same side (default: %(default)s)",
    )
    split.add_argument(
        "--group-by",
        default="",
        help="also keep records sharing this metadata key on the same side",
    )
    split.set_defaults(handler=_command_split)

    tokenizer = sub.add_parser("tokenizer", help="train a byte-pair encoder on a corpus")
    tokenizer.add_argument("--corpus", required=True, type=Path)
    tokenizer.add_argument("--out", required=True, type=Path)
    tokenizer.add_argument("--vocab-size", type=int, default=None)
    tokenizer.set_defaults(handler=_command_tokenizer)

    train = sub.add_parser("train", help="train the tokenizer and both models")
    train.add_argument("--splits", required=True, type=Path, help="a directory from 'dslm split'")
    train.add_argument("--out-dir", required=True, type=Path)
    train.add_argument("--vocab-size", type=int, default=None)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--seed", type=int, default=None)
    train.set_defaults(handler=_command_train)

    evaluate = sub.add_parser("evaluate", help="score the models, with the gates")
    evaluate.add_argument("--splits", required=True, type=Path)
    evaluate.add_argument("--model", required=True, type=Path, help="a directory from 'dslm train'")
    evaluate.add_argument("--on", choices=sorted(PARTS), default="test")
    evaluate.add_argument(
        "--null",
        type=Path,
        default=None,
        help="a corpus sharing no record with training, used to calibrate the "
        "contamination threshold",
    )
    evaluate.add_argument(
        "--drop-shared",
        action="store_true",
        help="remove null records that also appear in training, and report how many",
    )
    evaluate.add_argument(
        "--allow-uncalibrated",
        action="store_true",
        help="apply the published contamination threshold without a null corpus",
    )
    evaluate.add_argument("--baseline", type=Path, default=None)
    evaluate.add_argument("--update-baseline", action="store_true")
    evaluate.add_argument("--note", default="", help="recorded in the baseline")
    evaluate.add_argument("--json-out", type=Path, default=None)
    evaluate.add_argument("--junit-out", type=Path, default=None)
    evaluate.add_argument("--markdown-out", type=Path, default=None)
    evaluate.set_defaults(handler=_command_evaluate)

    contamination = sub.add_parser("contamination", help="measure overlap between two corpora")
    contamination.add_argument("--train", required=True, type=Path)
    contamination.add_argument("--evaluate", required=True, type=Path)
    contamination.add_argument("--null", type=Path, default=None)
    contamination.add_argument("--drop-shared", action="store_true")
    contamination.add_argument("--allow-uncalibrated", action="store_true")
    contamination.set_defaults(handler=_command_contamination)

    classify = sub.add_parser("classify", help="classify text with a trained model")
    classify.add_argument("--model", required=True, type=Path)
    classify.add_argument("--text", action="append", default=[])
    classify.add_argument("--corpus", type=Path, default=None)
    classify.set_defaults(handler=_command_classify)

    doctor = sub.add_parser("doctor", help="what this installation would do")
    doctor.add_argument("--corpus", type=Path, default=None, help="also inspect this corpus")
    doctor.set_defaults(handler=_command_doctor)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return an exit code."""
    _use_utf8()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = load()
        configure(
            level=args.log_level or settings.log.level,
            json_output=settings.log.format == "json",
        )
        handler = args.handler
        return int(handler(args, settings))
    except DslmError as exc:
        print(f"{exc.message}", file=sys.stderr)
        if exc.remedy:
            print(f"  {exc.remedy}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:  # pragma: no cover - a person pressing ctrl-c
        print("interrupted", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover - exercised through __main__.py
    raise SystemExit(main())
