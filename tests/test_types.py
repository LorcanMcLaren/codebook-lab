from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from codebook_lab.types import (
    AnnotationRunResult,
    ExperimentRunResult,
    ExperimentSpec,
    HumanGroundTruthResult,
    HumanReliabilityResult,
    MetricsRunResult,
)


class TestExperimentSpec:
    def test_defaults(self):
        spec = ExperimentSpec(task="t", model="m")
        assert spec.use_examples is False
        assert spec.prompt_type == "standard"
        assert spec.temperature is None
        assert spec.top_p is None
        assert spec.process_textbox is False
        assert spec.country_iso_code == "USA"

    def test_frozen(self):
        spec = ExperimentSpec(task="t", model="m")
        with pytest.raises(FrozenInstanceError):
            spec.task = "other"

    def test_equality(self):
        a = ExperimentSpec(task="t", model="m", temperature=0.5)
        b = ExperimentSpec(task="t", model="m", temperature=0.5)
        assert a == b

    def test_custom_values(self):
        spec = ExperimentSpec(
            task="my-task",
            model="llama3:8b",
            use_examples=True,
            prompt_type="persona",
            temperature=0.7,
            top_p=0.9,
            process_textbox=True,
            country_iso_code="IRL",
        )
        assert spec.task == "my-task"
        assert spec.temperature == 0.7
        assert spec.country_iso_code == "IRL"


class TestAnnotationRunResult:
    def test_construction(self):
        result = AnnotationRunResult(
            model="m",
            output_path=Path("out.csv"),
            experiment_directory=Path("exp"),
            config={"key": "val"},
            char_counts={"input_chars": 100},
            timing_data={"total_inference_time": 1.0},
            emissions=0.001,
            dataframe=None,
        )
        assert result.model == "m"
        assert result.emissions == 0.001


class TestMetricsRunResult:
    def test_construction(self):
        result = MetricsRunResult(
            output_csv=Path("metrics.csv"),
            report_file=Path("report.txt"),
            columns_to_compare=["col_a"],
            metrics_by_column={"col_a": {"accuracy": 0.9}},
            reports={"col_a": "report text"},
            total_inference_time=1.2,
            avg_inference_time=0.6,
            input_chars=100,
            output_chars=50,
            energy_consumed=0.01,
            emissions=0.001,
            cpu_model="cpu",
            gpu_model=None,
            summary_text="Run Summary\n\nPerformance\n- col_a: accuracy=0.900",
        )
        assert result.columns_to_compare == ["col_a"]
        assert result.metrics_by_column["col_a"]["accuracy"] == 0.9
        assert "Run Summary" in result.summary_text
        assert result.output_chars == 50


class TestHumanReliabilityResult:
    def test_construction(self):
        result = HumanReliabilityResult(
            validation_issues=None,
            pairwise_icr=None,
            multirater_icr=None,
            disagreements=None,
            summary_text="summary",
            output_dir=Path("out"),
        )
        assert result.summary_text == "summary"
        assert result.output_dir == Path("out")


class TestHumanGroundTruthResult:
    def test_construction(self):
        result = HumanGroundTruthResult(
            ground_truth=None,
            adjudication_queue=None,
            validation_issues=None,
            output_dir=None,
        )
        assert result.output_dir is None
