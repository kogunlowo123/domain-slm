#!/usr/bin/env python
"""What a domain tokenizer buys, measured against one trained on other text.

Documentation that executes. Run it:

    python examples/tokenizer_demo.py

The claim "a domain tokenizer is better" is worth exactly the number attached to
it, so this attaches one — and compares like with like: two tokenizers of the
*same vocabulary size*, differing only in what they were trained on.
"""

from __future__ import annotations

import sys
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dslm.corpus.synth import PLANS, generate
from dslm.tokenizer import bpe

#: Stand-in for general English.
#:
#: Not a real general corpus — that would be a licensing question and a download
#: — but genuinely out of domain, which is the property the comparison needs.
#:
#: It has to be *large and varied enough to fill the vocabulary*, and the first
#: version was not: six sentences of sixty-five distinct words left 569 of the
#: 1,024 merges unmade, so the "general" tokenizer was losing partly because it
#: had been handicapped. A comparison against a crippled baseline is not a
#: comparison. The demo prints the unfilled count for both so a reader can check
#: that neither side was starved.
WORDS = [
    "about",
    "above",
    "across",
    "after",
    "again",
    "against",
    "almost",
    "alone",
    "along",
    "already",
    "also",
    "although",
    "always",
    "among",
    "another",
    "answer",
    "appear",
    "around",
    "arrive",
    "because",
    "become",
    "before",
    "begin",
    "behind",
    "believe",
    "between",
    "beyond",
    "bring",
    "build",
    "carry",
    "catch",
    "centre",
    "certain",
    "change",
    "choose",
    "clear",
    "close",
    "common",
    "complete",
    "consider",
    "continue",
    "country",
    "course",
    "create",
    "decide",
    "describe",
    "develop",
    "different",
    "difficult",
    "direct",
    "discover",
    "during",
    "early",
    "effect",
    "enough",
    "evening",
    "example",
    "expect",
    "experience",
    "explain",
    "family",
    "father",
    "feeling",
    "figure",
    "finally",
    "follow",
    "forget",
    "friend",
    "garden",
    "general",
    "government",
    "happen",
    "happy",
    "history",
    "hundred",
    "important",
    "increase",
    "indeed",
    "interest",
    "kitchen",
    "knowledge",
    "language",
    "learn",
    "leave",
    "letter",
    "listen",
    "little",
    "market",
    "matter",
    "measure",
    "meeting",
    "member",
    "memory",
    "middle",
    "minute",
    "moment",
    "money",
    "morning",
    "mother",
    "mountain",
    "movement",
    "natural",
    "necessary",
    "neighbour",
    "never",
    "notice",
    "number",
    "object",
    "offer",
    "often",
    "order",
    "other",
    "paper",
    "parent",
    "particular",
    "perhaps",
    "period",
    "person",
    "picture",
    "place",
    "point",
    "police",
    "position",
    "possible",
    "power",
    "prepare",
    "present",
    "pretty",
    "probably",
    "problem",
    "produce",
    "provide",
    "public",
    "quarter",
    "question",
    "quick",
    "quiet",
    "ready",
    "reason",
    "receive",
    "record",
    "remember",
    "remove",
    "repeat",
    "report",
    "require",
    "result",
    "return",
    "school",
    "science",
    "season",
    "second",
    "serious",
    "service",
    "several",
    "should",
    "shoulder",
    "simple",
    "single",
    "sister",
    "situation",
    "society",
    "soldier",
    "sometimes",
    "special",
    "spring",
    "station",
    "still",
    "story",
    "street",
    "strong",
    "student",
    "subject",
    "success",
    "sudden",
    "suggest",
    "summer",
    "suppose",
    "surface",
    "surprise",
    "system",
    "table",
    "teacher",
    "television",
    "terrible",
    "thought",
    "thousand",
    "through",
    "together",
    "tomorrow",
    "tonight",
    "toward",
    "travel",
    "trouble",
    "understand",
    "village",
    "village",
    "visitor",
    "weather",
    "wedding",
    "welcome",
    "whether",
    "window",
    "winter",
    "wonder",
    "worry",
    "writer",
    "yesterday",
    "young",
]

FRAMES = (
    "the {a} of the {b} was {c} enough to {d} the {e} before the {f}",
    "she would {d} the {a} and {d} the {b} while the {c} {e} waited",
    "it is {c} that a {a} should {d} its {b} without a {e} of {f}",
    "they {d} the {a} at the {b} and found the {c} {e} already there",
    "no {a} can {d} a {b} that the {c} {e} has not first {d}",
    "after the {a} the {b} began to {d} and the {c} {e} followed",
)


def general_corpus(records: int = 4000, seed: int = 4) -> list[str]:
    """Synthesise out-of-domain English with enough vocabulary to be fair."""
    rng = Random(seed)  # noqa: S311 - a fixture, not cryptography
    return [
        rng.choice(FRAMES).format(**{key: rng.choice(WORDS) for key in "abcdef"})
        for _ in range(records)
    ]


VOCAB = 1024


def main() -> int:
    corpus = generate(PLANS["main"])
    texts = [record.text for record in corpus]

    print("Two tokenizers, same size, different training data")
    print("-" * 52)
    domain, domain_report = bpe.train(texts, vocab_size=VOCAB)
    general, general_report = bpe.train(general_corpus(), vocab_size=VOCAB)
    print(f"  domain  : {domain_report.summary()}")
    print(f"  general : {general_report.summary()}")
    if domain_report.unfilled or general_report.unfilled:
        print("  (unfilled merges above mean a side was starved of vocabulary;")
        print("   a comparison against a starved baseline is not a comparison)")

    print("\nMeasured on the same domain corpus")
    print("-" * 52)
    on_domain = bpe.compression(domain, texts)
    on_general = bpe.compression(general, texts)
    print(f"  {'':<10}{'tokens/record':>15}{'tokens/word':>14}{'chars/token':>14}")
    for name, measured in (("domain", on_domain), ("general", on_general)):
        print(
            f"  {name:<10}{measured.tokens_per_record:>15.2f}"
            f"{measured.tokens_per_word:>14.3f}{measured.characters_per_token:>14.2f}"
        )
    ratio = on_general.tokens_per_record / on_domain.tokens_per_record
    print(f"\n  The general tokenizer needs {ratio:.2f}x as many tokens per record.")
    print("  Every extra token is another position the model attends to, and")
    print("  another parameter it spends rediscovering that three pieces are one")
    print("  word.")

    print("\nWhat the domain tokenizer learned to keep whole")
    print("-" * 52)
    for word in ("actuator", "hydraulic", "transponder", "pressurisation", "nitrogen"):
        print(
            f"  {word:<16} domain {len(domain.encode(word)):>2} piece(s)"
            f"   general {len(general.encode(word)):>2} piece(s)"
        )

    print("\nNeither ever emits an unknown token")
    print("-" * 52)
    exotic = "圧力 low strut p/n 356-1194"
    for name, tokenizer in (("domain", domain), ("general", general)):
        ids = tokenizer.encode(exotic)
        unknown = sum(1 for index in ids if index == tokenizer.vocab[bpe.UNKNOWN])
        print(f"  {name:<10}{len(ids):>4} tokens, {unknown} unknown")
    print("  The byte fallback guarantees it. A tokenizer that can produce <unk>")
    print("  loses information silently, and the loss is blamed on the model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
