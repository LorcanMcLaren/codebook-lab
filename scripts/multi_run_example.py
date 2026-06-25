"""Run a small multi-experiment sweep with CodeBook Lab.

This script is intentionally small so users can test the package quickly.
Edit the grid below to explore more combinations once the basic workflow is
working in your environment. The package will try to start a local Ollama
server if needed and will pull any missing models automatically.
"""

from pathlib import Path

from codebook_lab import run_experiment_grid


OUTPUT_ROOT = Path("outputs")

PARAM_GRID = {
    "country_iso_code": "IRL",
    "tasks": ["policy-sentiment"],
    "models": ["gemma3:270m"],
    "use_examples": [True],
    "prompt_types": ["standard"],
    "temperatures": [0],
    "top_ps": [None],
    "chat_modes": ["per_text"],
    "reasoning": [None],
    "process_textboxes": [True],
}


def main() -> None:
    """Run a small sweep and print a short summary of the completed runs."""
    results = run_experiment_grid(
        param_grid=PARAM_GRID,
        output_root=OUTPUT_ROOT,
    )

    print(f"Completed {len(results)} experiment runs.")
    for result in results:
        print(f"- {result.model_id}: {result.experiment_directory}")


if __name__ == "__main__":
    main()
