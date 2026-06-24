from __future__ import annotations

from itertools import product
from pathlib import Path
import time

from .examples import get_example_task_dir
from .ollama import ensure_ollama_model
from .types import ExperimentRunResult, ExperimentSpec


def _normalize_optional_float(value) -> float | None:
    """Convert empty or ``"None"``-style sweep values to ``None``, others to ``float``."""
    if value in (None, "", "None"):
        return None
    return float(value)


def _coerce_bool(value) -> bool:
    """Convert common string and numeric truthy values to ``bool``."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _coerce_sweep(values, default):
    """Return a list of sweep values, falling back to a single default value."""
    if values in (None, []):
        return [default]
    return list(values)


def expand_param_grid(param_grid: dict) -> list[ExperimentSpec]:
    """Expand a parameter grid into concrete :class:`ExperimentSpec` runs.

    Args:
        param_grid: In-memory grid dictionary describing the Cartesian-product
            sweep. Expected keys include ``tasks``, ``models``, ``use_examples``,
            ``prompt_types``, ``temperatures``, ``top_ps``, ``process_textboxes``,
            and ``country_iso_code``.

    Returns:
        A list of experiment specifications, one per Cartesian-product combination.
    """
    tasks = list(param_grid.get("tasks") or ["policy-sentiment"])
    models = list(param_grid.get("models") or ["gemma3:270m"])
    use_examples_values = _coerce_sweep(param_grid.get("use_examples"), False)
    prompt_types = _coerce_sweep(param_grid.get("prompt_types"), "standard")
    temperatures = _coerce_sweep(param_grid.get("temperatures"), None)
    top_ps = _coerce_sweep(param_grid.get("top_ps"), None)
    process_textboxes = _coerce_sweep(param_grid.get("process_textboxes"), False)
    process_spans = _coerce_sweep(param_grid.get("process_spans"), False)
    country_iso_code = str(param_grid.get("country_iso_code") or "USA")

    specs = []
    for task, model, use_examples, prompt_type, temperature, top_p, process_textbox, process_span in product(
        tasks,
        models,
        use_examples_values,
        prompt_types,
        temperatures,
        top_ps,
        process_textboxes,
        process_spans,
    ):
        specs.append(
            ExperimentSpec(
                task=str(task),
                model=str(model),
                use_examples=_coerce_bool(use_examples),
                prompt_type=str(prompt_type),
                temperature=_normalize_optional_float(temperature),
                top_p=_normalize_optional_float(top_p),
                process_textbox=_coerce_bool(process_textbox),
                process_span=_coerce_bool(process_span),
                country_iso_code=country_iso_code,
            )
        )
    return specs


def resolve_task_dir(task: str, task_root: str | Path | None = None) -> Path:
    """Resolve a task directory from either a user path or bundled examples.

    Args:
        task: Task directory name such as ``"policy-sentiment"``.
        task_root: Optional directory containing user-defined task folders.

    Returns:
        Filesystem path to the resolved task directory.
    """
    if task_root is not None:
        candidate = Path(task_root) / task
        if candidate.exists():
            return candidate

    try:
        return get_example_task_dir(task)
    except FileNotFoundError as exc:
        searched = f"{Path(task_root) / task}" if task_root is not None else "no external task_root provided"
        raise FileNotFoundError(
            f"Task '{task}' was not found. Searched external task path {searched} "
            "and bundled example tasks."
        ) from exc


def build_experiment_paths(
    *,
    task: str,
    model: str,
    use_examples: bool,
    prompt_type: str,
    temperature,
    top_p,
    process_textbox: bool,
    process_span: bool = False,
    output_root: str | Path = "outputs",
    timestamp: str | None = None,
) -> dict[str, Path | str]:
    """Build deterministic output paths for one experiment configuration.

    Args:
        task: Task folder name under ``tasks/``.
        model: Ollama model identifier.
        use_examples: Whether codebook examples are included in prompts.
        prompt_type: Prompt wrapper name used for the run.
        temperature: Optional temperature value or ``None``.
        top_p: Optional top-p value or ``None``.
        process_textbox: Whether textbox annotations are included.
        output_root: Root directory where experiment outputs should be written.
        timestamp: Optional timestamp string in ``YYYY-MM-DD_HH-MM-SS`` format.

    Returns:
        Dictionary containing the experiment directory and standard output file paths.
    """
    timestamp = timestamp or time.strftime("%Y-%m-%d_%H-%M-%S")
    model_safe = model.replace(":", "-")
    experiment_dir = Path(output_root) / task / f"{model_safe}_examples{str(use_examples).lower()}_{prompt_type}"
    model_id = f"{model}_examples{str(use_examples).lower()}_{prompt_type}"

    if temperature is not None:
        experiment_dir = Path(f"{experiment_dir}_temp{temperature}")
        model_id = f"{model_id}_temp{temperature}"
    if top_p is not None:
        experiment_dir = Path(f"{experiment_dir}_topp{top_p}")
        model_id = f"{model_id}_topp{top_p}"
    if process_textbox:
        experiment_dir = Path(f"{experiment_dir}_textbox")
        model_id = f"{model_id}_textbox"
    if process_span:
        experiment_dir = Path(f"{experiment_dir}_span")
        model_id = f"{model_id}_span"

    experiment_dir = Path(f"{experiment_dir}_{timestamp}")

    return {
        "timestamp": timestamp,
        "model_id": model_id,
        "experiment_directory": experiment_dir,
        "output_csv": experiment_dir / "output.csv",
        "report_file": experiment_dir / "classification_reports.txt",
        "emissions_file": experiment_dir / "emissions.csv",
        "timing_file": experiment_dir / "timing_data.json",
        "char_counts_file": experiment_dir / "char_counts.json",
    }


def run_experiment(
    spec: ExperimentSpec,
    *,
    task_root: str | Path | None = None,
    output_root: str | Path = "outputs",
    metrics_output_root: str | Path | None = None,
    timestamp: str | None = None,
    pull_model: bool = True,
    start_ollama_if_needed: bool = True,
    quantization_type: str | None = None,
):
    """Run one end-to-end experiment and evaluate it against ground truth.

    Args:
        spec: Concrete experiment specification to execute.
        task_root: Optional directory containing user-defined task folders with
            ``codebook.json`` and ``ground-truth.csv``. If omitted, bundled example
            tasks are used when the task name matches one shipped with the package.
        output_root: Root directory where per-run outputs should be created.
        metrics_output_root: Directory for aggregate metrics CSV files. Defaults to ``output_root / "metrics"``.
        timestamp: Optional timestamp string to control output folder naming.
        pull_model: If ``True``, run ``ollama pull`` for ``spec.model`` before execution.
            Defaults to ``True`` so experiment runs ensure the requested model is available.
        start_ollama_if_needed: If ``True``, try to auto-start a local Ollama
            server when the default local host is not already reachable.
            Defaults to ``True`` so experiment runs can bring up local Ollama
            automatically when needed.
        quantization_type: Optional metadata field written to the metrics CSV.

    Returns:
        :class:`ExperimentRunResult` containing both annotation and metrics results.
    """
    from .annotate import run_annotation
    from .metrics import run_metrics

    output_root = Path(output_root)
    metrics_output_root = Path(metrics_output_root) if metrics_output_root is not None else output_root / "metrics"

    task_dir = resolve_task_dir(spec.task, task_root)
    ground_truth_csv = task_dir / "ground-truth.csv"
    codebook_path = task_dir / "codebook.json"

    if not codebook_path.exists():
        raise FileNotFoundError(f"Codebook not found for task '{spec.task}': {codebook_path}")
    if not ground_truth_csv.exists():
        raise FileNotFoundError(f"Ground truth file not found for task '{spec.task}': {ground_truth_csv}")

    if pull_model:
        ensure_ollama_model(spec.model)

    paths = build_experiment_paths(
        task=spec.task,
        model=spec.model,
        use_examples=spec.use_examples,
        prompt_type=spec.prompt_type,
        temperature=spec.temperature,
        top_p=spec.top_p,
        process_textbox=spec.process_textbox,
        process_span=spec.process_span,
        output_root=output_root,
        timestamp=timestamp,
    )

    label = spec.task
    metrics_output_csv = metrics_output_root / f"{spec.task}_metrics_log.csv"

    annotation_result = run_annotation(
        model=spec.model,
        csv_path=ground_truth_csv,
        codebook_path=codebook_path,
        output_path=paths["output_csv"],
        experiment_directory=paths["experiment_directory"],
        prompt_type=spec.prompt_type,
        use_examples=spec.use_examples,
        temperature=spec.temperature,
        top_p=spec.top_p,
        process_textbox=spec.process_textbox,
        process_span=spec.process_span,
        country_iso_code=spec.country_iso_code,
        start_ollama_if_needed=start_ollama_if_needed,
    )

    metrics_result = run_metrics(
        ground_truth_csv=ground_truth_csv,
        llm_output_csv=paths["output_csv"],
        label=label,
        output_csv=metrics_output_csv,
        model_id=paths["model_id"],
        codebook_path=codebook_path,
        report_file=paths["report_file"],
        quantization_type=quantization_type,
        temperature=spec.temperature,
        top_p=spec.top_p,
        prompt_type=spec.prompt_type,
        use_examples=spec.use_examples,
        process_textbox=spec.process_textbox,
        process_span=spec.process_span,
        emissions_file=paths["emissions_file"],
        experiment_directory=paths["experiment_directory"],
        timestamp=paths["timestamp"],
        timing_file=paths["timing_file"],
        char_counts_file=paths["char_counts_file"],
    )

    print()
    print(f"Experiment complete: {paths['experiment_directory']}")
    print(metrics_result.summary_text)

    return ExperimentRunResult(
        spec=spec,
        experiment_directory=paths["experiment_directory"],
        model_id=paths["model_id"],
        label=label,
        annotation=annotation_result,
        metrics=metrics_result,
    )


def run_experiment_grid(
    *,
    specs: list[ExperimentSpec] | None = None,
    param_grid: dict | None = None,
    task_root: str | Path | None = None,
    output_root: str | Path = "outputs",
    metrics_output_root: str | Path | None = None,
    pull_models: bool = True,
    start_ollama_if_needed: bool = True,
) -> list[ExperimentRunResult]:
    """Run a whole parameter sweep from a grid or a prebuilt spec list.

    Args:
        specs: Optional pre-expanded list of :class:`ExperimentSpec` objects.
        param_grid: Optional in-memory parameter grid dictionary.
        task_root: Optional directory containing user-defined task folders.
        output_root: Root directory for per-run outputs.
        metrics_output_root: Directory for aggregate metrics CSV files.
        pull_models: If ``True``, pull each Ollama model before its first run.
            Defaults to ``True`` so sweep runs ensure requested models are available.
        start_ollama_if_needed: If ``True``, try to auto-start a local Ollama
            server before each run when needed. Defaults to ``True``.

    Returns:
        List of :class:`ExperimentRunResult` objects, one per completed run.
    """
    if specs is None:
        if param_grid is None:
            raise ValueError("Provide either specs or an in-memory param_grid dictionary.")
        specs = expand_param_grid(param_grid=param_grid)

    results = []
    for spec in specs:
        results.append(
            run_experiment(
                spec,
                task_root=task_root,
                output_root=output_root,
                metrics_output_root=metrics_output_root,
                pull_model=pull_models,
                start_ollama_if_needed=start_ollama_if_needed,
            )
        )
    return results
