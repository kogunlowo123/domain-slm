"""Environment-driven defaults.

Small on purpose. Almost everything about a verdict belongs in the objectives
file, which is committed and reviewed; what lives here is the handful of
settings that differ between a laptop and a CI runner, and none of them change
what the gate decides.

``extra="forbid"`` catches a misspelt *field*. It does not catch a misspelt
*section* — pydantic never builds a ``logs`` key for it to reject — so
:func:`load` checks the section names itself. Both are translated into this
tool's own error type, because the command line renders those and lets anything
else escape as a traceback: a guard whose test does not exercise the boundary
that consumes it is a guard nobody has checked.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from dslm.errors import ConfigError


class Section(BaseModel):
    """Base for the settings sections."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GateSettings(Section):
    """Defaults for the gates.

    Every one of these can also be given on the command line, where it is
    visible in a diff; these are the values a machine falls back to.
    """

    #: How far a metric may fall against the committed baseline before the
    #: regression gate fails.
    tolerance: Annotated[float, Field(gt=0, le=1)] = 0.005
    #: The share of an evaluation set that may be contaminated.
    max_contaminated: Annotated[float, Field(ge=0, le=1)] = 0.02
    #: The share of an evaluation set that must be long enough to check at all.
    #: A check that could not be applied has not passed.
    min_coverage: Annotated[float, Field(gt=0, le=1)] = 0.5
    #: The quantile of the null distribution used as the contamination
    #: threshold. See dslm.corpus.contamination for why this is calibrated
    #: rather than fixed.
    quantile: Annotated[float, Field(gt=0, lt=1)] = 0.999


class LogSettings(Section):
    """Where diagnostics go and what they look like.

    Always stderr — that is not configurable, because stdout carries the report.
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "console"] = "json"


class TrainSettings(Section):
    """Defaults for training.

    Bounded, because an unbounded setting is a way to spend an afternoon
    discovering that ``epochs`` was read from the environment as a string.
    """

    #: Tokens the byte-pair encoder learns.
    vocab_size: Annotated[int, Field(ge=300, le=100_000)] = 1024
    #: Tokens kept per record.
    max_tokens: Annotated[int, Field(ge=1, le=4096)] = 64
    #: Passes over the training set.
    epochs: Annotated[int, Field(ge=1, le=1000)] = 20
    #: The seed. Recorded in every artefact this produces.
    seed: Annotated[int, Field(ge=0)] = 20260908


class Settings(BaseSettings):
    """The whole configuration.

    Environment variables are ``DSLM_<SECTION>__<FIELD>``, for example
    ``DSLM_GATE__TOLERANCE=0.01``.
    """

    model_config = SettingsConfigDict(
        env_prefix="DSLM_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        frozen=True,
    )

    gate: GateSettings = GateSettings()
    log: LogSettings = LogSettings()
    train: TrainSettings = TrainSettings()


def _reject_unknown_sections(names: Iterable[str]) -> None:
    """Fail on an ``DSLM_`` variable whose section is not a real one.

    Covers the process environment, which is where a CI runner sets things and
    where a mistake is invisible. A misspelt section in a ``.env`` file is not
    covered and does not need to be: that file is in the repository, next to the
    person editing it.
    """
    prefix = "DSLM_"
    known = set(Settings.model_fields)
    unknown = sorted(
        name
        for name in names
        if name.startswith(prefix) and name[len(prefix) :].split("__", 1)[0].lower() not in known
    )
    if unknown:
        raise ConfigError(
            f"unknown setting(s): {', '.join(unknown)}.",
            remedy=(
                f"Sections are {', '.join(sorted(known))}. "
                f"Variables are {prefix}<SECTION>__<FIELD>, for example "
                f"{prefix}GATE__MIN_SAMPLES=50."
            ),
        )


def load(environ: Mapping[str, str] | None = None) -> Settings:
    """Read settings from the environment and ``.env``."""
    _reject_unknown_sections(os.environ if environ is None else environ)
    try:
        return Settings()
    except ValidationError as exc:
        fields = ", ".join(".".join(str(part) for part in error["loc"]) for error in exc.errors())
        raise ConfigError(
            f"the settings are invalid: {fields or 'see below'}.",
            remedy=(
                "Variables are DSLM_<SECTION>__<FIELD>, for example "
                f"DSLM_GATE__TOLERANCE=0.01. Unknown names are refused rather "
                f"than ignored.\n  {exc}"
            ),
        ) from exc
