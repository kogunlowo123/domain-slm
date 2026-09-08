"""Byte-pair encoding, trained from scratch on the domain corpus.

## Why train one at all

A general-purpose tokenizer is trained on general text, and it spends its
vocabulary accordingly. Domain text is not general: ``actuator``, ``pitot``,
``bleed``, ``strut`` and ``amm 32-41-00`` are common here and rare everywhere
else, so a general tokenizer shatters them into pieces. Every extra piece is
another position the model must attend to and another parameter it must learn to
reassemble — the model spends capacity rediscovering that ``act`` ``uat`` ``or``
is one thing.

The measurable version of that claim is **tokens per record**, and it is
measured rather than asserted: :func:`compression` reports it for any tokenizer
against any corpus, and the shipped comparison is in ``docs/tokenizer.md``.

## What this implementation is and is not

Word-level BPE over a whitespace pre-tokenisation, with a byte-level fallback so
that nothing is ever unrepresentable. It is the classic algorithm — count
adjacent pair frequencies, merge the most frequent, repeat — implemented
directly, in about two hundred lines, because the point of this repository is
that every step is inspectable.

It is not a production tokenizer. There is no regex pre-tokeniser tuned over
years, no special-token machinery beyond what this project uses, and training on
a gigabyte would be slow. On this corpus it trains in a few seconds.

## Determinism

Exactly reproducible, everywhere, forever. Every operation is over integers and
strings — counting, sorting, merging — with no floating point anywhere, so the
one-ULP libm divergence that the *model* has to account for cannot arise here.
Ties in pair frequency are broken by the pair itself, sorted, so a dictionary's
iteration order can never decide a merge. That is what lets the trained
tokenizer be gated by digest rather than by tolerance. See ADR-005.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dslm.errors import TokenizerError

#: Marks the end of a word, so that ``valve`` at the end of a word and ``valve``
#: inside ``valves`` are different tokens. Without it the tokenizer cannot tell
#: a suffix from a whole word and merges across the boundary.
END_OF_WORD = "</w>"

#: The unknown token. Never produced by :meth:`Tokenizer.encode` — the byte
#: fallback guarantees every character is representable — but present so that a
#: vocabulary loaded from elsewhere has somewhere to put one.
UNKNOWN = "<unk>"

#: Padding, for batching records of different lengths.
PADDING = "<pad>"

#: Reserved ids, in this order. Fixed rather than derived, so that a model
#: trained against one vocabulary and run against another fails loudly on the
#: vocabulary digest rather than quietly on a shifted id.
SPECIAL_TOKENS = (PADDING, UNKNOWN)

#: Below this, a vocabulary cannot hold the byte alphabet plus anything useful.
MIN_VOCAB = 300
#: Above this, training on a corpus this size spends minutes to add merges that
#: appear once each.
MAX_VOCAB = 100_000

#: A pair is two symbols. Named because the bare 2 appears in this file
#: meaning both 'a pair' and 'often enough to be worth a vocabulary slot',
#: and those are not the same thing.
PAIR = 2
#: A merge is only made when its pair occurs at least this often. Below it,
#: the merge would spend a vocabulary slot on a single word.
MIN_PAIR_FREQUENCY = 2


def pre_tokenise(text: str) -> list[str]:
    """Split text into words for BPE training and encoding.

    Whitespace, after NFC normalisation and lowercasing. Deliberately simple and
    deliberately *not* stripping punctuation: ``32-41-00`` is a manual reference
    and ``p/n`` is a part number, and a pre-tokeniser that discards the hyphen
    and the slash destroys the very structure a domain tokenizer should learn.
    """
    return unicodedata.normalize("NFC", text).lower().split()


@dataclass(frozen=True, slots=True)
class TrainingReport:
    """What training the tokenizer did."""

    vocab_size: int
    merges: int
    #: Distinct words seen, and the total including repeats.
    words: int
    word_instances: int
    #: The alphabet before any merge: every character in the corpus.
    alphabet: int
    #: Merges that were requested but could not be made, because every remaining
    #: pair occurred once. Reported rather than silently ignored: asking for
    #: 8,000 tokens and getting 5,000 is a fact about the corpus worth knowing.
    unfilled: int

    def summary(self) -> str:
        """One line, for a terminal."""
        note = f", {self.unfilled} unfilled" if self.unfilled else ""
        return (
            f"{self.vocab_size} token(s) from {self.merges} merge(s) over "
            f"{self.words} distinct word(s) ({self.word_instances} instances){note}"
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialise for a report."""
        return {
            "vocab_size": self.vocab_size,
            "merges": self.merges,
            "distinct_words": self.words,
            "word_instances": self.word_instances,
            "alphabet": self.alphabet,
            "unfilled_merges": self.unfilled,
        }


@dataclass(frozen=True, slots=True)
class Compression:
    """How economically a tokenizer represents a corpus."""

    records: int
    tokens: int
    words: int
    characters: int
    #: Tokens whose id is the unknown token. Should be zero; a non-zero value
    #: means the byte fallback did not cover something, which is a bug.
    unknown: int

    @property
    def tokens_per_record(self) -> float:
        """The headline number. Lower is better."""
        return self.tokens / self.records if self.records else 0.0

    @property
    def tokens_per_word(self) -> float:
        """How many pieces the average word is broken into. 1.0 is perfect."""
        return self.tokens / self.words if self.words else 0.0

    @property
    def characters_per_token(self) -> float:
        """The usual measure of a tokenizer's efficiency. Higher is better."""
        return self.characters / self.tokens if self.tokens else 0.0

    def summary(self) -> str:
        """One line, for a terminal."""
        return (
            f"{self.tokens_per_record:.2f} tokens/record, "
            f"{self.tokens_per_word:.3f} tokens/word, "
            f"{self.characters_per_token:.2f} chars/token"
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialise for a report."""
        return {
            "records": self.records,
            "tokens": self.tokens,
            "words": self.words,
            "characters": self.characters,
            "unknown_tokens": self.unknown,
            "tokens_per_record": round(self.tokens_per_record, 4),
            "tokens_per_word": round(self.tokens_per_word, 4),
            "characters_per_token": round(self.characters_per_token, 4),
        }


@dataclass(frozen=True, slots=True)
class Tokenizer:
    """A trained byte-pair encoder."""

    #: Ordered merges. The order is the algorithm: applying them in a different
    #: order gives a different tokenisation of the same text.
    merges: tuple[tuple[str, str], ...]
    #: token -> id. Ids are stable: the special tokens first, then the alphabet,
    #: then one per merge in merge order.
    vocab: dict[str, int]

    @property
    def vocab_size(self) -> int:
        """How many distinct tokens this tokenizer can emit."""
        return len(self.vocab)

    def digest(self) -> str:
        """A content address over the merges and the vocabulary.

        Exact, not approximate. Everything in a tokenizer is integers and
        strings, so two machines that trained on the same corpus produce
        byte-identical output and a digest is the right check — unlike the model,
        whose float weights differ in the last bits across platforms.
        """
        material = "\n".join(
            [
                *(f"{left} {right}" for left, right in self.merges),
                *(f"{token}\t{index}" for token, index in sorted(self.vocab.items())),
            ]
        )
        return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _split_word(self, word: str) -> list[str]:
        """Apply the merges to one word, returning its pieces."""
        pieces = [*list(word), END_OF_WORD]
        if len(pieces) < PAIR:
            return pieces
        ranks = self._ranks
        while True:
            best: tuple[int, int] | None = None
            for index in range(len(pieces) - 1):
                rank = ranks.get((pieces[index], pieces[index + 1]))
                if rank is not None and (best is None or rank < best[0]):
                    best = (rank, index)
            if best is None:
                return pieces
            _, at = best
            pieces[at : at + 2] = [pieces[at] + pieces[at + 1]]

    def encode(self, text: str) -> list[int]:
        """Encode *text* to token ids.

        Never emits the unknown token: any piece that survived the merges but is
        not in the vocabulary is emitted as its UTF-8 bytes, each of which is in
        the vocabulary by construction. A tokenizer that can produce ``<unk>``
        loses information silently, and the loss shows up as an accuracy the
        model is blamed for.
        """
        ids: list[int] = []
        for word in pre_tokenise(text):
            for piece in self._split_word(word):
                index = self.vocab.get(piece)
                if index is not None:
                    ids.append(index)
                    continue
                for byte in piece.encode("utf-8"):
                    fallback = self.vocab.get(f"<0x{byte:02X}>")
                    ids.append(fallback if fallback is not None else self.vocab[UNKNOWN])
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Decode token ids back to text.

        Lossy by construction: the pre-tokeniser lowercased and collapsed
        whitespace, so this returns the normalised form rather than the original.
        Provided for inspection — reading what the model actually saw — not as a
        round trip.
        """
        inverse = self._inverse
        pieces = [inverse.get(index, UNKNOWN) for index in ids]
        joined = "".join(pieces)
        return joined.replace(END_OF_WORD, " ").strip()

    def as_dict(self) -> dict[str, Any]:
        """Serialise the whole tokenizer."""
        return {
            "dslm_tokenizer": 1,
            "vocab_size": self.vocab_size,
            "digest": self.digest(),
            "merges": [[left, right] for left, right in self.merges],
            "vocab": dict(sorted(self.vocab.items(), key=lambda item: item[1])),
        }

    def save(self, path: str | Path) -> Path:
        """Write the tokenizer as JSON and return the path."""
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return file

    # -- derived, cached on first use -------------------------------------

    @property
    def _ranks(self) -> dict[tuple[str, str], int]:
        cached = _RANK_CACHE.get(id(self))
        if cached is None:
            cached = {pair: rank for rank, pair in enumerate(self.merges)}
            _RANK_CACHE[id(self)] = cached
        return cached

    @property
    def _inverse(self) -> dict[int, str]:
        cached = _INVERSE_CACHE.get(id(self))
        if cached is None:
            cached = {index: token for token, index in self.vocab.items()}
            _INVERSE_CACHE[id(self)] = cached
        return cached


#: Derived lookups, keyed by object identity. The dataclass is frozen so it
#: cannot hold them, and rebuilding a 8,000-entry dictionary on every call to
#: ``encode`` made tokenising the corpus the slowest step in the pipeline.
_RANK_CACHE: dict[int, dict[tuple[str, str], int]] = {}
_INVERSE_CACHE: dict[int, dict[int, str]] = {}


def train(  # noqa: C901, PLR0912, PLR0915 - incremental counting, kept in one place
    texts: Iterable[str], *, vocab_size: int = 4096
) -> tuple[Tokenizer, TrainingReport]:
    """Train a byte-pair encoder on *texts*.

    Returns the tokenizer and a report of what training did — including how many
    requested merges could not be made, which is the difference between "you have
    a 8,000-token vocabulary" and "you asked for one".
    """
    if not MIN_VOCAB <= vocab_size <= MAX_VOCAB:
        raise TokenizerError(
            f"a vocabulary of {vocab_size} is outside {MIN_VOCAB}..{MAX_VOCAB}.",
            remedy=(
                f"Below {MIN_VOCAB} there is no room for the byte alphabet plus useful "
                "merges; above it, training spends minutes adding tokens that occur once."
            ),
        )

    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(pre_tokenise(text))
    if not counts:
        raise TokenizerError(
            "there are no words to train on.",
            remedy="An empty corpus produces a tokenizer with no merges, which is not one.",
        )

    # Every byte is in the vocabulary, so nothing is ever unrepresentable.
    alphabet = {f"<0x{byte:02X}>" for byte in range(256)}
    alphabet |= {character for word in counts for character in word}
    alphabet.add(END_OF_WORD)

    words: dict[tuple[str, ...], int] = {
        (*word, END_OF_WORD): frequency for word, frequency in counts.items()
    }

    merges: list[tuple[str, str]] = []
    budget = vocab_size - len(SPECIAL_TOKENS) - len(alphabet)
    unfilled = 0
    if budget < 0:
        raise TokenizerError(
            f"a vocabulary of {vocab_size} cannot hold the {len(alphabet)}-symbol alphabet.",
            remedy=f"Ask for at least {len(alphabet) + len(SPECIAL_TOKENS)} tokens.",
        )

    # Incremental pair counting.
    #
    # The straightforward implementation recounts every adjacent pair in every
    # word after each merge. That is O(merges x corpus) and took twenty seconds
    # for 1,745 merges over 6,000 records — the slowest step in the whole
    # pipeline by an order of magnitude, for a tokenizer that is meant to be
    # retrained casually.
    #
    # A merge only changes the words that contain the merged pair, so the counts
    # only need repairing there. `where` maps a pair to the words holding it;
    # applying a merge subtracts the affected words' old pairs, rewrites them,
    # and adds their new pairs back. Same merges, same order, same output —
    # asserted against the simple implementation in the test suite.
    table: list[list[str]] = []
    weights: list[int] = []
    for pieces, frequency in words.items():
        table.append(list(pieces))
        weights.append(frequency)

    pairs: Counter[tuple[str, str]] = Counter()
    where: dict[tuple[str, str], set[int]] = {}

    def add(index: int, sign: int) -> None:
        pieces = table[index]
        weight = weights[index] * sign
        for position in range(len(pieces) - 1):
            pair = (pieces[position], pieces[position + 1])
            pairs[pair] += weight
            if sign > 0:
                where.setdefault(pair, set()).add(index)

    for index in range(len(table)):
        add(index, 1)

    for _ in range(budget):
        # Ties broken by the pair itself, so a dictionary's iteration order can
        # never decide a merge and training is reproducible everywhere.
        best_pair: tuple[str, str] | None = None
        best_count = 0
        for pair, count in pairs.items():
            if count <= 0:
                continue
            tie = count == best_count and best_pair is not None and pair > best_pair
            if count > best_count or tie:
                best_pair, best_count = pair, count
        if best_pair is None or best_count < MIN_PAIR_FREQUENCY:
            unfilled = budget - len(merges)
            break

        merges.append(best_pair)
        merged = best_pair[0] + best_pair[1]
        affected = sorted(where.get(best_pair, set()))
        for index in affected:
            current = table[index]
            if len(current) < PAIR:
                continue
            add(index, -1)
            rebuilt: list[str] = []
            position = 0
            while position < len(current):
                if (
                    position < len(current) - 1
                    and current[position] == best_pair[0]
                    and current[position + 1] == best_pair[1]
                ):
                    rebuilt.append(merged)
                    position += 2
                else:
                    rebuilt.append(current[position])
                    position += 1
            table[index] = rebuilt
            add(index, 1)
        # The pair is gone from every word that had it.
        pairs.pop(best_pair, None)
        where.pop(best_pair, None)

    vocab: dict[str, int] = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for token in sorted(alphabet):
        vocab.setdefault(token, len(vocab))
    for left, right in merges:
        vocab.setdefault(left + right, len(vocab))

    tokenizer = Tokenizer(merges=tuple(merges), vocab=vocab)
    report = TrainingReport(
        vocab_size=len(vocab),
        merges=len(merges),
        words=len(counts),
        word_instances=sum(counts.values()),
        alphabet=len(alphabet),
        unfilled=unfilled,
    )
    return tokenizer, report


def _apply(words: dict[tuple[str, ...], int], pair: tuple[str, str]) -> dict[tuple[str, ...], int]:
    """Merge *pair* everywhere it occurs, returning the new word table.

    The straightforward form, kept because :func:`train_reference` uses it and
    the test suite asserts that the fast path produces identical merges. An
    optimisation nothing checks against a simple version is a rewrite nobody can
    trust.
    """
    merged = pair[0] + pair[1]
    out: dict[tuple[str, ...], int] = {}
    for pieces, frequency in words.items():
        if len(pieces) < PAIR:
            out[pieces] = out.get(pieces, 0) + frequency
            continue
        rebuilt: list[str] = []
        index = 0
        while index < len(pieces):
            if (
                index < len(pieces) - 1
                and pieces[index] == pair[0]
                and pieces[index + 1] == pair[1]
            ):
                rebuilt.append(merged)
                index += 2
            else:
                rebuilt.append(pieces[index])
                index += 1
        key = tuple(rebuilt)
        out[key] = out.get(key, 0) + frequency
    return out


def train_reference(
    texts: Iterable[str], *, vocab_size: int = 4096
) -> tuple[Tokenizer, TrainingReport]:
    """The obvious BPE implementation: recount every pair after every merge.

    Kept, exercised, and asserted to agree with :func:`train` merge for merge.
    :func:`train` is an optimisation — incremental pair counting — and an
    optimisation whose only evidence is that it runs faster is a rewrite nobody
    can trust. This is the thing it is trusted against.

    Too slow for the corpus (about twenty seconds where :func:`train` takes
    well under one), which is why it is not the default.
    """
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(pre_tokenise(text))
    if not counts:
        raise TokenizerError(
            "there are no words to train on.",
            remedy="An empty corpus produces a tokenizer with no merges, which is not one.",
        )

    alphabet = {f"<0x{byte:02X}>" for byte in range(256)}
    alphabet |= {character for word in counts for character in word}
    alphabet.add(END_OF_WORD)

    words: dict[tuple[str, ...], int] = {
        (*word, END_OF_WORD): frequency for word, frequency in counts.items()
    }
    merges: list[tuple[str, str]] = []
    budget = vocab_size - len(SPECIAL_TOKENS) - len(alphabet)
    unfilled = 0

    for _ in range(max(budget, 0)):
        pairs: Counter[tuple[str, str]] = Counter()
        for pieces, frequency in words.items():
            for index in range(len(pieces) - 1):
                pairs[pieces[index], pieces[index + 1]] += frequency
        if not pairs:
            unfilled = budget - len(merges)
            break
        pair, count = max(pairs.items(), key=lambda item: (item[1], item[0]))
        if count < MIN_PAIR_FREQUENCY:
            unfilled = budget - len(merges)
            break
        merges.append(pair)
        words = _apply(words, pair)

    vocab: dict[str, int] = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for token in sorted(alphabet):
        vocab.setdefault(token, len(vocab))
    for left, right in merges:
        vocab.setdefault(left + right, len(vocab))

    return Tokenizer(merges=tuple(merges), vocab=vocab), TrainingReport(
        vocab_size=len(vocab),
        merges=len(merges),
        words=len(counts),
        word_instances=sum(counts.values()),
        alphabet=len(alphabet),
        unfilled=unfilled,
    )


def load(path: str | Path) -> Tokenizer:
    """Read a tokenizer written by :meth:`Tokenizer.save`.

    The stored digest is recomputed and compared. A tokenizer file edited by hand
    — or truncated by a failed write — would otherwise produce a model that
    trains happily against ids that mean something else.
    """
    file = Path(path)
    if not file.is_file():
        raise TokenizerError(
            f"the tokenizer {str(file)!r} could not be opened.",
            remedy="Train one with 'dslm tokenizer --corpus <path> --out <path>'.",
        )
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TokenizerError(
            f"{file.name} is not valid JSON: {exc}.",
            remedy="Retrain it; a truncated tokenizer cannot be repaired by hand.",
        ) from exc

    version = payload.get("dslm_tokenizer")
    if not isinstance(version, int) or version > 1:
        raise TokenizerError(
            f"{file.name} was written by a newer version (schema {version!r}).",
            remedy="This build understands schema 1. Upgrade dslm.",
        )
    try:
        merges = tuple((left, right) for left, right in payload["merges"])
        vocab = {str(token): int(index) for token, index in payload["vocab"].items()}
    except (KeyError, TypeError, ValueError) as exc:
        raise TokenizerError(
            f"{file.name} is missing its merges or its vocabulary.",
            remedy="Retrain it.",
        ) from exc

    tokenizer = Tokenizer(merges=merges, vocab=vocab)
    stored = payload.get("digest")
    if stored and stored != tokenizer.digest():
        raise TokenizerError(
            f"{file.name} does not match its own digest.",
            remedy=(
                "The file has been edited or truncated. A model trained against a "
                "different vocabulary produces ids that mean something else, and nothing "
                "downstream would notice."
            ),
        )
    return tokenizer


def compression(tokenizer: Tokenizer, texts: Iterable[str]) -> Compression:
    """Measure how economically *tokenizer* represents *texts*.

    The claim "a domain tokenizer is better" is exactly this number, and nothing
    else. It is reported for the domain tokenizer and for the baselines side by
    side, because a compression figure with nothing to compare it to says only
    that text has a length.
    """
    records = tokens = words = characters = unknown = 0
    unknown_id = tokenizer.vocab.get(UNKNOWN)
    for text in texts:
        records += 1
        pieces = pre_tokenise(text)
        words += len(pieces)
        characters += sum(len(piece) for piece in pieces)
        ids = tokenizer.encode(text)
        tokens += len(ids)
        if unknown_id is not None:
            unknown += sum(1 for index in ids if index == unknown_id)
    if not records:
        raise TokenizerError(
            "there is nothing to measure compression over.",
            remedy="Pass a non-empty corpus.",
        )
    return Compression(
        records=records,
        tokens=tokens,
        words=words,
        characters=characters,
        unknown=unknown,
    )
