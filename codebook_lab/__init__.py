from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from .examples import copy_example_task, get_example_task_dir, get_example_task_files, list_example_tasks
from .ollama import ensure_ollama_available, ensure_ollama_model, get_ollama_base_url
from .prompts import (
    PromptContext,
    get_prompt_wrapper,
    list_prompt_wrappers,
    register_prompt_wrapper,
)
from .types import AnnotationRunResult, ExperimentRunResult, ExperimentSpec, MetricsRunResult

if TYPE_CHECKING:
    from .annotate import run_annotation
    from .experiments import expand_param_grid, resolve_task_dir, run_experiment, run_experiment_grid
    from .metrics import run_metrics

try:
    __version__ = version("codebook-lab")
except PackageNotFoundError:
    __version__ = "0.0.0"

_LAZY_EXPORTS = {
    "expand_param_grid": (".experiments", "expand_param_grid"),
    "resolve_task_dir": (".experiments", "resolve_task_dir"),
    "run_annotation": (".annotate", "run_annotation"),
    "run_experiment": (".experiments", "run_experiment"),
    "run_experiment_grid": (".experiments", "run_experiment_grid"),
    "run_metrics": (".metrics", "run_metrics"),
}

__all__ = [
    "PromptContext",
    "AnnotationRunResult",
    "ExperimentRunResult",
    "ExperimentSpec",
    "MetricsRunResult",
    "copy_example_task",
    "ensure_ollama_available",
    "ensure_ollama_model",
    "expand_param_grid",
    "get_example_task_dir",
    "get_example_task_files",
    "get_ollama_base_url",
    "get_prompt_wrapper",
    "list_example_tasks",
    "list_prompt_wrappers",
    "register_prompt_wrapper",
    "resolve_task_dir",
    "run_annotation",
    "run_experiment",
    "run_experiment_grid",
    "run_metrics",
]


def __getattr__(name: str) -> Any:
    """Load heavier runtime modules only when their exports are requested."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
