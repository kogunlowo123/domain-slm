"""The error type and the exit codes.

The exit codes are part of the interface, not an implementation detail. A
pipeline reads them, and "non-zero" is not enough information: a corpus that
failed its contamination check and a tool that could not read its corpus are
different facts, and a job that cannot tell them apart gets retried until it
goes green.

Every error carries a *remedy* as well as a message. A message says what is
wrong; a remedy says what to do, and the person reading it is usually not the
person who wrote the check.
"""

from __future__ import annotations

#: Everything that could be checked was checked, and it held.
EXIT_OK = 0
#: The command line was used wrongly.
EXIT_USAGE = 1
#: A gate failed: the corpus is contaminated, or a metric regressed past its
#: tolerance. The data or the model is at fault, not the tool.
EXIT_GATE_FAILED = 2
#: The tool could not produce a verdict at all. A missing corpus, an unreadable
#: model, a malformed configuration. Distinct from 2 on purpose: a broken
#: pipeline must never be mistaken for a clean bill of health.
EXIT_ERROR = 3


class DslmError(Exception):
    """Base class for every error this package raises deliberately.

    Anything that escapes as something else is a bug, and the command line
    renders these and lets everything else through as a traceback — which is
    the correct behaviour for an unexpected failure and the wrong one for an
    expected failure a user can fix.
    """

    def __init__(self, message: str, *, remedy: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.remedy = remedy

    def __str__(self) -> str:
        return self.message


class ConfigError(DslmError):
    """The settings are wrong. Raised before any work is done."""


class CorpusError(DslmError):
    """A corpus file could not be read, or is not a corpus."""


class RecordError(DslmError):
    """A record is malformed. Reported with its line number, never dropped."""


class SplitError(DslmError):
    """A split is impossible, or would leak between its parts."""


class ContaminationError(DslmError):
    """The evaluation set overlaps the training set beyond the threshold.

    Its own type because it is the one failure this project exists to catch:
    a reported accuracy computed over contaminated data is not a measurement,
    and reporting it anyway is worse than reporting nothing.
    """


class TokenizerError(DslmError):
    """A tokenizer could not be trained, loaded, or applied."""


class ModelError(DslmError):
    """A model could not be trained, loaded, or applied."""


class EvaluationError(DslmError):
    """An evaluation could not be completed, or its result cannot be trusted."""


class BaselineError(DslmError):
    """A committed baseline is missing, unreadable, or incomparable."""
