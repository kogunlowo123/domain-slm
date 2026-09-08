"""Settings.

Small module, but it decides how the gate behaves on a CI runner, and the
reason it refuses unknown keys is that the alternative — silently ignoring a
typo — leaves a job running at defaults with nobody the wiser.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from dslm.config import GateSettings, LogSettings, Settings, TrainSettings, load
from dslm.errors import ConfigError

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _neutral_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No inherited DSLM_* variables, and no `.env` in reach."""
    for name in tuple(os.environ):
        if name.startswith("DSLM_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


class TestDefaults:
    def test_a_machine_with_no_configuration_gets_working_defaults(self):
        settings = load()
        assert 0 < settings.gate.tolerance < 1
        assert settings.train.vocab_size >= 300
        assert settings.log.format == "json"

    def test_the_contamination_gate_is_on_by_default(self):
        # The gate this project exists for is not something you have to switch
        # on. A default that had to be enabled would be a default nobody enabled.
        assert 0 < load().gate.max_contaminated < 0.1

    def test_settings_are_frozen(self):
        with pytest.raises(ValidationError):
            load().gate.tolerance = 0.5


class TestEnvironment:
    def test_a_nested_variable_reaches_its_section(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DSLM_GATE__TOLERANCE", "0.05")
        monkeypatch.setenv("DSLM_LOG__FORMAT", "console")

        settings = load()

        assert settings.gate.tolerance == 0.05
        assert settings.log.format == "console"

    def test_a_dot_env_file_is_read(self, tmp_path: Path):
        (tmp_path / ".env").write_text("DSLM_TRAIN__VOCAB_SIZE=555\n", encoding="utf-8")
        assert load().train.vocab_size == 555

    def test_a_real_variable_beats_the_dot_env_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / ".env").write_text("DSLM_TRAIN__VOCAB_SIZE=555\n", encoding="utf-8")
        monkeypatch.setenv("DSLM_TRAIN__VOCAB_SIZE", "999")
        assert load().train.vocab_size == 999

    def test_a_misspelt_field_is_an_error_rather_than_a_shrug(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # Without this the typo leaves a CI job at the default and reporting
        # nothing. Reported as a ConfigError rather than pydantic's own error,
        # because the command line renders this tool's errors and lets anything
        # else escape as a traceback.
        monkeypatch.setenv("DSLM_GATE__TOLERANC", "0.05")
        with pytest.raises(ConfigError, match="toleranc"):
            load()

    def test_a_misspelt_section_is_an_error_too(self, monkeypatch: pytest.MonkeyPatch):
        # The worse of the two typos: pydantic never builds a `logs` key, so
        # extra="forbid" has nothing to reject and the variable is dropped.
        monkeypatch.setenv("DSLM_LOGS__LEVEL", "DEBUG")
        with pytest.raises(ConfigError, match="DSLM_LOGS__LEVEL"):
            load()

    def test_the_error_names_the_sections_that_do_exist(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DSLM_NETWORK__PROXY", "http://example.invalid")
        with pytest.raises(ConfigError) as caught:
            load()
        for section in ("gate", "log", "train"):
            assert section in caught.value.remedy

    def test_variables_belonging_to_other_tools_are_left_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("SOME_OTHER_TOOL", "value")
        monkeypatch.setenv("DSLMOMETHING", "unprefixed")
        assert load().train.vocab_size >= 300

    def test_an_explicit_environment_can_be_supplied(self):
        with pytest.raises(ConfigError):
            load({"DSLM_TYPO__FIELD": "1"})


class TestBounds:
    @pytest.mark.parametrize("value", [0.0, -1.0, 2.0])
    def test_the_tolerance_is_bounded(self, value: float):
        with pytest.raises(ValidationError):
            GateSettings(tolerance=value)

    @pytest.mark.parametrize("value", [-0.1, 1.5])
    def test_the_contamination_limit_is_bounded(self, value: float):
        with pytest.raises(ValidationError):
            GateSettings(max_contaminated=value)

    @pytest.mark.parametrize("value", [0.0, 1.0])
    def test_the_quantile_is_strictly_inside_zero_and_one(self, value: float):
        # 1.0 would set the threshold at the null's maximum, which no data
        # exceeds, so the gate could never fire.
        with pytest.raises(ValidationError):
            GateSettings(quantile=value)

    @pytest.mark.parametrize("value", [10, 500_000])
    def test_the_vocabulary_is_bounded(self, value: int):
        with pytest.raises(ValidationError):
            TrainSettings(vocab_size=value)

    @pytest.mark.parametrize("value", [0, 5000])
    def test_the_epoch_count_is_bounded(self, value: int):
        # Zero epochs produces an untrained model that still reports a number.
        with pytest.raises(ValidationError):
            TrainSettings(epochs=value)

    def test_the_log_level_is_a_closed_set(self):
        with pytest.raises(ValidationError):
            LogSettings(level="TRACE")  # type: ignore[arg-type]

    def test_the_log_format_is_a_closed_set(self):
        with pytest.raises(ValidationError):
            LogSettings(format="xml")  # type: ignore[arg-type]

    def test_settings_can_be_built_directly(self):
        settings = Settings(gate=GateSettings(tolerance=0.01), log=LogSettings(level="ERROR"))
        assert settings.gate.tolerance == 0.01
        assert settings.log.level == "ERROR"
