"""A record, a corpus, and the file they live in.

A **record** is one work order: a piece of domain text, the label it carries,
and — the part most corpora omit and then cannot reconstruct — where it came
from and under what licence.

A **corpus** is an ordered collection of records plus the lines that could not
be read. The format is JSON Lines: one header object, then one record per line,
gzipped when the path ends ``.gz``. That choice is inherited from the telemetry
tool in this series and is right for the same reason — a corpus is written by a
process that can be killed, and a truncated JSON array is unreadable in its
entirety while a truncated JSONL file has lost exactly its last line.

Provenance is not decoration. A corpus whose licence cannot be established is a
corpus that cannot be published, and by the time anyone asks, the person who
assembled it has usually left. So ``source`` and ``license`` are **required
fields on every record**, not a note in a README, and the data card is generated
from them rather than written by hand.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dslm.errors import CorpusError, RecordError

#: The corpus format version, written into every header and checked on read.
CORPUS_SCHEMA_VERSION = 1

#: A corpus is read into memory whole. Applied to the decompressed stream as
#: well as to the file on disk: checking only the file would make a small gzip
#: that expands without bound a perfectly acceptable input.
MAX_CORPUS_BYTES = 512 * 1024 * 1024
#: A line longer than this is not a record; it is a file that is not a corpus.
MAX_LINE_BYTES = 128 * 1024
#: Text longer than this is a document, not a work order. Refused rather than
#: truncated: silently keeping the first 4 KB of something would change what the
#: model was trained on without saying so.
MAX_TEXT_CHARS = 4096


class Record(BaseModel):
    """One labelled piece of domain text, with its provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: Annotated[str, Field(min_length=1, max_length=64)]
    text: Annotated[str, Field(min_length=1, max_length=MAX_TEXT_CHARS)]
    #: The ATA 100 chapter this work order belongs to. The classification
    #: target; see docs/domain.md for the chapters used.
    label: Annotated[int, Field(ge=0, le=99)]
    #: Where this record came from. Required — see the module docstring.
    source: Annotated[str, Field(min_length=1, max_length=128)]
    #: An SPDX identifier, or "proprietary". Required for the same reason.
    license: Annotated[str, Field(min_length=1, max_length=64)]
    #: Free-form, bounded. Aircraft type, shop, anything a split may group on.
    meta: dict[str, str] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _text_is_normalised(cls, value: str) -> str:
        """Refuse text that is not NFC, and text that is only whitespace.

        Two strings that look identical and differ in their Unicode composition
        are two vocabulary entries, two n-grams, and — worst — a contamination
        check that misses an exact duplicate. Normalising *here* rather than
        during training means every consumer sees the same string.
        """
        if not value.strip():
            raise ValueError("text is empty or only whitespace")
        if unicodedata.normalize("NFC", value) != value:
            raise ValueError(
                "text is not in Unicode NFC. Two strings that look identical and "
                "differ in composition are two vocabulary entries and a missed duplicate"
            )
        return value

    def fingerprint(self) -> str:
        """A content address over what a model actually learns from.

        The text and the label. Not the id, not the provenance, not the
        metadata: renaming a record or correcting its licence does not change
        what was trained, and a drift check that fires on those is a check
        people learn to ignore.
        """
        material = f"{self.label}\x1f{self.text}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]

    def as_dict(self) -> dict[str, Any]:
        """Serialise for a corpus file."""
        payload: dict[str, Any] = {
            "record_id": self.record_id,
            "text": self.text,
            "label": self.label,
            "source": self.source,
            "license": self.license,
        }
        if self.meta:
            payload["meta"] = dict(self.meta)
        return payload


@dataclass(frozen=True, slots=True)
class Corpus:
    """Records, and what could not be read."""

    records: tuple[Record, ...]
    #: (line number, reason) for every line the reader rejected. Reported,
    #: never silently dropped: a corpus that quietly lost 3% of itself trains a
    #: model that is quietly 3% worse for a reason nobody can find.
    unreadable: tuple[tuple[int, str], ...] = ()
    source: str = "<memory>"
    metadata: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[Record]:
        return iter(self.records)

    def __getitem__(self, index: int) -> Record:
        return self.records[index]

    @property
    def labels(self) -> tuple[int, ...]:
        """Every label present, sorted."""
        return tuple(sorted({record.label for record in self.records}))

    @property
    def licenses(self) -> tuple[str, ...]:
        """Every licence present, sorted."""
        return tuple(sorted({record.license for record in self.records}))

    def label_counts(self) -> dict[int, int]:
        """How many records carry each label."""
        counts: dict[int, int] = {}
        for record in self.records:
            counts[record.label] = counts.get(record.label, 0) + 1
        return dict(sorted(counts.items()))

    def digest(self) -> str:
        """A content address over the records, order-independent.

        Sorted before hashing so that shuffling a corpus does not change its
        identity — what a model learns from is the set, and a shuffle that
        moved the digest would make every legitimate reordering look like a
        data change.
        """
        material = "\n".join(sorted(record.fingerprint() for record in self.records))
        return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()

    def as_header(self) -> dict[str, Any]:
        """The first line of a corpus file."""
        return {
            "dslm_corpus": CORPUS_SCHEMA_VERSION,
            "records": len(self.records),
            "labels": list(self.labels),
            "licenses": list(self.licenses),
            "digest": self.digest(),
            "metadata": dict(self.metadata),
        }

    def write(self, path: str | Path) -> Path:
        """Write this corpus as JSON Lines, gzipped when the path ends ``.gz``."""
        payload = [self.as_header(), *(record.as_dict() for record in self.records)]
        return write_lines(path, payload)


def build_corpus(
    records: Iterable[Record],
    *,
    source: str = "<memory>",
    metadata: dict[str, str] | None = None,
    unreadable: Sequence[tuple[int, str]] = (),
) -> Corpus:
    """Assemble a corpus, refusing duplicate record ids.

    Duplicate ids are refused rather than de-duplicated. Two records with one id
    means something upstream is wrong, and quietly keeping one of them decides
    which by accident.
    """
    ordered = tuple(records)
    seen: set[str] = set()
    for record in ordered:
        if record.record_id in seen:
            raise CorpusError(
                f"record id {record.record_id!r} appears more than once.",
                remedy=(
                    "Ids identify records across splits and reports. Two records "
                    "sharing one means an upstream join went wrong; keeping either "
                    "silently would decide which by accident."
                ),
            )
        seen.add(record.record_id)
    return Corpus(
        records=ordered,
        unreadable=tuple(unreadable),
        source=source,
        metadata=dict(metadata or {}),
    )


def is_compressed(path: str | Path) -> bool:
    """Is this path gzipped? Decided by the suffix, never by sniffing.

    Sniffing the magic bytes is friendlier to a mislabelled file and worse for
    everything else: it teaches people to ship gzip named ``.jsonl``, and then
    something downstream that reads by name gets bytes it cannot parse.
    """
    return Path(path).suffix == ".gz"


def write_lines(path: str | Path, lines: Sequence[dict[str, Any]]) -> Path:
    """Write JSON Lines, gzipped by suffix, and return the path.

    The gzip member carries a zero timestamp and no filename, so writing the
    same corpus twice produces the same bytes. The default header stamps the
    current time, which would make every regenerated corpus differ from the
    committed one for a reason no reader could see — and turn the drift check
    into noise.
    """
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(item, sort_keys=True, ensure_ascii=False) for item in lines) + "\n"
    if is_compressed(file):
        with (
            file.open("wb") as sink,
            gzip.GzipFile(filename="", mode="wb", fileobj=sink, mtime=0) as raw,
        ):
            raw.write(body.encode("utf-8"))
    else:
        file.write_text(body, encoding="utf-8", newline="\n")
    return file


class _Bounded(io.RawIOBase):
    """A read-only stream that refuses to yield more than *limit* bytes."""

    def __init__(self, stream: gzip.GzipFile, limit: int) -> None:
        self._stream = stream
        self._limit = limit
        self._seen = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        chunk = self._stream.read(len(buffer))
        self._seen += len(chunk)
        if self._seen > self._limit:
            raise CorpusError(
                f"the corpus expands to more than {self._limit} bytes.",
                remedy=(
                    "A small file that decompresses without bound is a decompression "
                    "bomb, not a corpus. Split it, or raise MAX_CORPUS_BYTES deliberately."
                ),
            )
        buffer[: len(chunk)] = chunk
        return len(chunk)

    def close(self) -> None:
        self._stream.close()
        super().close()


def open_text(path: str | Path) -> IO[str]:
    """Open a corpus for reading as text, transparently decompressing.

    The stream refuses to yield more than ``MAX_CORPUS_BYTES``. For a plain file
    that duplicates the size check in :func:`read_corpus`; for a gzipped one it
    is the only thing between this tool and a small file that expands until the
    runner is killed, which reaches an operator as a flaky build.
    """
    file = Path(path)
    if not is_compressed(file):
        return file.open(encoding="utf-8")
    bounded = _Bounded(gzip.GzipFile(file, mode="rb"), MAX_CORPUS_BYTES)
    return io.TextIOWrapper(io.BufferedReader(bounded), encoding="utf-8")


def read_corpus(path: str | Path) -> Corpus:
    """Read a corpus file.

    A malformed record line is counted and reported, not fatal. A malformed
    *header* is fatal, because without it there is no record of what this file
    claims to be and every check downstream would be against a guess.
    """
    file = Path(path)
    if not file.is_file():
        raise CorpusError(
            f"the corpus {str(file)!r} could not be opened.",
            remedy="Check the path, or generate one with 'dslm synth --out <path>'.",
        )
    size = file.stat().st_size
    if size > MAX_CORPUS_BYTES:
        raise CorpusError(
            f"the corpus is {size} bytes; the limit is {MAX_CORPUS_BYTES}.",
            remedy="Split it, or raise MAX_CORPUS_BYTES deliberately.",
        )

    records: list[Record] = []
    unreadable: list[tuple[int, str]] = []
    header: dict[str, Any] | None = None

    with open_text(file) as handle:
        for number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if len(line.encode("utf-8")) > MAX_LINE_BYTES:
                unreadable.append((number, f"the line is longer than {MAX_LINE_BYTES} bytes"))
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                if header is None:
                    raise CorpusError(
                        f"{file.name} does not start with a corpus header: {exc}.",
                        remedy=(
                            "The first line must be the header object written by Corpus.write."
                        ),
                    ) from exc
                unreadable.append((number, f"not valid JSON: {exc.msg}"))
                continue
            if not isinstance(payload, dict):
                unreadable.append((number, "not a JSON object"))
                continue
            if header is None:
                header = _validated_header(payload, name=file.name)
                continue
            try:
                records.append(Record.model_validate(payload))
            except Exception as exc:  # noqa: BLE001 - reported per line, never fatal
                unreadable.append((number, _first_problem(exc)))

    if header is None:
        raise CorpusError(
            f"{file.name} is empty.",
            remedy="A corpus file has a header line even when it holds no records.",
        )

    return build_corpus(
        records,
        source=str(file),
        metadata={str(k): str(v) for k, v in (header.get("metadata") or {}).items()},
        unreadable=unreadable,
    )


def _validated_header(payload: dict[str, Any], *, name: str) -> dict[str, Any]:
    version = payload.get("dslm_corpus")
    if version is None:
        raise CorpusError(
            f"{name} has no corpus header.",
            remedy="The first line must carry 'dslm_corpus' and a record count.",
        )
    if not isinstance(version, int) or version > CORPUS_SCHEMA_VERSION:
        raise CorpusError(
            f"{name} was written by a newer version (schema {version}).",
            remedy=f"This build understands schema {CORPUS_SCHEMA_VERSION}. Upgrade dslm.",
        )
    return payload


def _first_problem(exc: Exception) -> str:
    """One line naming the first thing wrong, for the unreadable list."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        found = errors()
        if found:
            first = found[0]
            where = ".".join(str(part) for part in first.get("loc", ())) or "record"
            return f"{where}: {first.get('msg', 'invalid')}"
    return str(exc).splitlines()[0] if str(exc) else "invalid record"


def require_records(corpus: Corpus, *, what: str) -> None:
    """Refuse an empty corpus, naming what was about to be done with it.

    An empty corpus produces a tokenizer with no merges, a model that predicts
    the majority of nothing, and an accuracy of 1.0 over zero examples. Every
    one of those is a number, and none of them is a measurement.
    """
    if not corpus.records:
        raise RecordError(
            f"the corpus holds no records, so {what} would be meaningless.",
            remedy="Check the path, and check the reader's 'unreadable' count for rejected lines.",
        )
