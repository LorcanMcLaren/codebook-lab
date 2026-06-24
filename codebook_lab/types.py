from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class AnnotationRunResult:
    """Result returned by ``run_annotation``.

    Attributes:
        model: Ollama model name used for the run, such as ``"gemma3:270m"``.
        output_path: Filesystem path to the generated annotation CSV.
        experiment_directory: Directory containing run metadata and outputs.
        config: Serializable experiment configuration written to ``config.json``.
        char_counts: Prompt and response character counts collected during the run.
        timing_data: Inference timing summary, including total and average seconds.
        emissions: CodeCarbon estimate in kilograms of CO2 equivalent, or ``None``.
        dataframe: Pandas DataFrame containing the annotated rows written to disk.
    """

    model: str
    output_path: Path
    experiment_directory: Path
    config: dict[str, Any]
    char_counts: dict[str, Any]
    timing_data: dict[str, Any]
    emissions: float | None
    dataframe: pd.DataFrame


@dataclass
class MetricsRunResult:
    """Result returned by ``run_metrics``.

    Attributes:
        output_csv: Filesystem path to the aggregate metrics CSV that was updated.
        report_file: Filesystem path to the per-column classification report text file.
        columns_to_compare: Annotation columns included in the evaluation.
        metrics_by_column: Nested dictionary of computed metrics keyed by column name.
        reports: Human-readable report text keyed by annotation column name.
        total_inference_time: Total model inference time in seconds, if available.
        avg_inference_time: Mean inference time per annotation request in seconds, if available.
        input_chars: Total prompt characters sent to the model, if available.
        output_chars: Total response characters returned by the model, if available.
        energy_consumed: Energy consumption in kilowatt-hours, if available.
        emissions: Emissions estimate in kilograms of CO2 equivalent, if available.
        cpu_model: CPU metadata recorded by CodeCarbon, if available.
        gpu_model: GPU metadata recorded by CodeCarbon, if available.
        summary_text: Plain-text summary of the main evaluation metrics.
        run_id: Identifier linking this run's rows across the runs and metrics tables.
        runs_csv: Filesystem path to the per-run efficiency/config table, if written.
        n_queries: Number of model inference calls in the annotation run, if available.
    """

    output_csv: Path
    report_file: Path
    columns_to_compare: list[str]
    metrics_by_column: dict[str, dict[str, Any]]
    reports: dict[str, str]
    total_inference_time: float | None
    avg_inference_time: float | None
    input_chars: int | None
    output_chars: int | None
    energy_consumed: float | None
    emissions: float | None
    cpu_model: str | None
    gpu_model: str | None
    summary_text: str
    run_id: str | None = None
    runs_csv: Path | None = None
    n_queries: int | None = None


@dataclass
class HumanReliabilityResult:
    """Result returned by ``calculate_human_reliability``.

    Attributes:
        validation_issues: DataFrame of coder-file and label validation issues.
        pairwise_icr: Pairwise inter-coder reliability metrics.
        multirater_icr: Multi-rater reliability metrics.
        disagreements: Item/field rows where coders supplied different labels.
        summary_text: Markdown summary of validation, reliability, and disagreements.
        output_dir: Directory where outputs were written, or ``None`` if no output
            directory was requested.
    """

    validation_issues: pd.DataFrame
    pairwise_icr: pd.DataFrame
    multirater_icr: pd.DataFrame
    disagreements: pd.DataFrame
    summary_text: str
    output_dir: Path | None


@dataclass
class HumanGroundTruthResult:
    """Result returned by ``build_human_ground_truth``.

    Attributes:
        ground_truth: Final or provisional human consensus labels.
        adjudication_queue: Rows that still need adjudication.
        validation_issues: DataFrame of coder-file and label validation issues.
        output_dir: Directory where outputs were written, or ``None`` if no output
            directory was requested.
    """

    ground_truth: pd.DataFrame
    adjudication_queue: pd.DataFrame
    validation_issues: pd.DataFrame
    output_dir: Path | None


@dataclass(frozen=True)
class ExperimentSpec:
    """Declarative specification for one experiment run in a sweep.

    Attributes:
        task: Task folder name under ``tasks/``, for example ``"policy-sentiment"``.
        model: Ollama model identifier, such as ``"gemma3:270m"``.
        use_examples: Whether to include worked examples from the codebook.
        prompt_type: Registered prompt wrapper name, for example ``"standard"``.
        temperature: Optional sampling temperature as ``None``, string, or float.
        top_p: Optional nucleus-sampling value as ``None``, string, or float.
        process_textbox: Whether textbox annotations should be generated and scored.
        process_span: Whether span annotations should be generated and scored.
        country_iso_code: Three-letter ISO 3166-1 alpha-3 code for CodeCarbon.
    """

    task: str
    model: str
    use_examples: bool = False
    prompt_type: str = "standard"
    temperature: float | None = None
    top_p: float | None = None
    process_textbox: bool = False
    process_span: bool = False
    country_iso_code: str = "USA"


@dataclass
class ExperimentRunResult:
    """Combined result returned by ``run_experiment``.

    Attributes:
        spec: The experiment specification that was executed.
        experiment_directory: Directory containing this run's outputs.
        model_id: Stable model/config identifier used in the metrics log.
        label: Task label written to the metrics CSV.
        annotation: Result object returned by ``run_annotation``.
        metrics: Result object returned by ``run_metrics``.
    """

    spec: ExperimentSpec
    experiment_directory: Path
    model_id: str
    label: str
    annotation: AnnotationRunResult
    metrics: MetricsRunResult
