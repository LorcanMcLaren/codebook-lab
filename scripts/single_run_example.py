"""Run one bundled-example experiment with CodeBook Lab.

Edit the constants below if you want to change the model, task, or output
location. This script assumes:

1. CodeBook Lab has been installed in the current environment, for example
   with ``python -m pip install codebook-lab``.
2. Ollama is installed and available on PATH.

The package will try to start a local Ollama server if needed and will pull the
requested model automatically before running the experiment.
"""

from pathlib import Path

from codebook_lab import ExperimentSpec, run_experiment


TASK = "policy-sentiment"
MODEL = "gemma3:270m"
COUNTRY_ISO_CODE = "IRL"
OUTPUT_ROOT = Path("outputs")


def main() -> None:
    """Run a single experiment and print the key output locations."""
    result = run_experiment(
        ExperimentSpec(
            task=TASK,
            model=MODEL,
            chat_mode="per_text",
            reasoning=None,
            process_textbox=True,
            country_iso_code=COUNTRY_ISO_CODE,
        ),
        output_root=OUTPUT_ROOT,
    )

    print("Completed single experiment run.")
    print(f"Experiment directory: {result.experiment_directory}")
    print(f"Metrics CSV: {result.metrics.output_csv}")
    print(f"Classification report: {result.metrics.report_file}")


if __name__ == "__main__":
    main()
