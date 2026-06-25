from __future__ import annotations

import pytest

from codebook_lab.experiments import (
    _coerce_bool,
    _normalize_optional_float,
    build_experiment_paths,
    expand_param_grid,
    normalize_chat_mode,
    normalize_reasoning,
)
from codebook_lab.types import ExperimentSpec


class TestCoerceBool:
    @pytest.mark.parametrize("value,expected", [
        (True, True),
        (False, False),
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("false", False),
        ("False", False),
        ("1", True),
        ("0", False),
        ("yes", True),
        ("no", False),
        (1, True),
        (0, False),
    ])
    def test_values(self, value, expected):
        assert _coerce_bool(value) is expected


class TestNormalizeOptionalFloat:
    def test_none_values(self):
        assert _normalize_optional_float(None) is None
        assert _normalize_optional_float("") is None
        assert _normalize_optional_float("None") is None

    def test_float_passthrough(self):
        assert _normalize_optional_float(0.5) == 0.5

    def test_string_to_float(self):
        assert _normalize_optional_float("0.7") == 0.7

    def test_int_to_float(self):
        assert _normalize_optional_float(1) == 1.0


class TestExpandParamGrid:
    def test_minimal_grid(self):
        specs = expand_param_grid({
            "tasks": ["policy-sentiment"],
            "models": ["gemma3:270m"],
        })
        assert len(specs) == 1
        assert specs[0].task == "policy-sentiment"
        assert specs[0].model == "gemma3:270m"
        assert specs[0].use_examples is True
        assert specs[0].temperature == 0.0
        assert specs[0].chat_mode == "per_text"

    def test_cartesian_product(self):
        specs = expand_param_grid({
            "tasks": ["t1"],
            "models": ["m1", "m2"],
            "use_examples": [False, True],
            "prompt_types": ["standard"],
        })
        assert len(specs) == 4  # 1 * 2 * 2 * 1

    def test_temperature_coercion(self):
        specs = expand_param_grid({
            "tasks": ["t"],
            "models": ["m"],
            "temperatures": ["0.5", "None"],
        })
        assert len(specs) == 2
        assert specs[0].temperature == 0.5
        assert specs[1].temperature is None

    def test_country_iso_code(self):
        specs = expand_param_grid({
            "tasks": ["t"],
            "models": ["m"],
            "country_iso_code": "DEU",
        })
        assert specs[0].country_iso_code == "DEU"

    def test_returns_experiment_specs(self):
        specs = expand_param_grid({"tasks": ["t"], "models": ["m"]})
        assert all(isinstance(s, ExperimentSpec) for s in specs)

    def test_chat_mode_and_reasoning_coercion(self):
        specs = expand_param_grid({
            "tasks": ["t"],
            "models": ["m"],
            "chat_modes": ["per-query", "continuous"],
            "reasoning": ["true", "false"],
        })
        assert len(specs) == 4
        assert {spec.chat_mode for spec in specs} == {"per_query", "continuous"}
        assert {spec.reasoning for spec in specs} == {True, False}


def test_normalize_chat_mode_aliases_and_rejects_unknown():
    assert normalize_chat_mode("fresh") == "per_query"
    assert normalize_chat_mode("row") == "per_text"
    assert normalize_chat_mode("single_chat") == "continuous"
    with pytest.raises(ValueError):
        normalize_chat_mode("foreverish")


def test_normalize_reasoning_values():
    assert normalize_reasoning("true") is True
    assert normalize_reasoning("off") is False
    assert normalize_reasoning("None") is None
    assert normalize_reasoning("low") == "low"


def test_build_experiment_paths_uses_run_id_directory(tmp_path):
    paths = build_experiment_paths(
        task="task",
        model="model:1",
        use_examples=True,
        prompt_type="standard",
        temperature=0.0,
        top_p=None,
        process_textbox=False,
        output_root=tmp_path,
        timestamp="2026-06-25_10-00-00",
        run_id="run_test",
    )
    assert paths["run_id"] == "run_test"
    assert paths["experiment_directory"] == tmp_path / "task" / "run_test"
    assert paths["output_csv"] == tmp_path / "task" / "run_test" / "output.csv"
