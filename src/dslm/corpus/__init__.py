"""The corpus: generating it, reading it, curating it, and splitting it safely."""

from __future__ import annotations

from dslm.corpus.record import (
    CORPUS_SCHEMA_VERSION,
    MAX_CORPUS_BYTES,
    MAX_LINE_BYTES,
    MAX_TEXT_CHARS,
    Corpus,
    Record,
    build_corpus,
    is_compressed,
    open_text,
    read_corpus,
    require_records,
    write_lines,
)

__all__ = [
    "CORPUS_SCHEMA_VERSION",
    "MAX_CORPUS_BYTES",
    "MAX_LINE_BYTES",
    "MAX_TEXT_CHARS",
    "Corpus",
    "Record",
    "build_corpus",
    "is_compressed",
    "open_text",
    "read_corpus",
    "require_records",
    "write_lines",
]
