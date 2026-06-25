# CodeBook Lab

[![DOI](https://zenodo.org/badge/1186234207.svg)](https://doi.org/10.5281/zenodo.19185921) [![PyPI](https://img.shields.io/pypi/v/codebook-lab)](https://pypi.org/project/codebook-lab/) [![Python](https://img.shields.io/pypi/pyversions/codebook-lab)](https://pypi.org/project/codebook-lab/) [![License](https://img.shields.io/pypi/l/codebook-lab)](https://pypi.org/project/codebook-lab/)

CodeBook Lab is an LLM annotation experiment pipeline for computational social science. It takes a codebook and labelled dataset from [CodeBook Studio](https://codebook.streamlit.app/) ([source](https://github.com/LorcanMcLaren/codebook-studio)) and runs structured experiments across the dimensions that matter for text-as-data research: model choice, model size, prompt style, zero-shot versus few-shot learning, and sampling hyperparameters — all benchmarked against human labels.

Experiments are controlled through Python objects rather than by editing pipeline code. Because the codebook and labelled data stay constant across runs, each dimension can be isolated and compared against the same human labels.

For a step-by-step walkthrough covering both tools, see the [CodeBook Studio & Lab Tutorial](https://lorcanmclaren.com/codebook-tutorial.html).

## Contents

- [How It Fits With CodeBook Studio](#how-it-fits-with-codebook-studio)
- [Package Overview](#package-overview)
- [Quickstart](#quickstart)
- [Experiment Configuration](#experiment-configuration)
- [Create Your Own Task](#create-your-own-task)
- [Advanced Customization](#advanced-customization)
- [License](#license)
- [Citation](#citation)

## How It Fits With CodeBook Studio

[CodeBook Studio](https://codebook.streamlit.app/) defines the task. CodeBook Lab runs and evaluates the experiment.

<table>
  <tr>
    <td align="center"><strong>CodeBook Studio</strong></td>
    <td align="center"></td>
    <td align="center"><strong>CodeBook Lab</strong></td>
  </tr>
  <tr>
    <td valign="top">
      Define the annotation task<br>
      Annotate texts with humans<br>
      Export <code>codebook.json</code><br>
      Save labeled data as <code>ground-truth.csv</code>
    </td>
    <td align="center" valign="middle">→</td>
    <td valign="top">
      Strip label columns automatically<br>
      Run LLM annotation experiments<br>
      Sweep over models, prompts, and hyperparameters<br>
      Evaluate outputs against human labels
    </td>
  </tr>
</table>

## Package Overview

The package is organized around a small set of importable modules:

- `codebook_lab.experiments`: high-level functions for single experiments and multi-run comparisons
- `codebook_lab.annotate`: lower-level annotation functions
- `codebook_lab.metrics`: evaluation and metrics functions
- `codebook_lab.human_reliability`: human coder validation, ICR, disagreement, and ground-truth helpers
- `codebook_lab.prompts`: prompt wrapper registry for built-in and custom prompt styles
- `codebook_lab.examples`: helpers for bundled example tasks
- `codebook_lab.types`: dataclasses for experiment specifications and result objects

The package also ships with a bundled example task, `policy-sentiment`, so you can start experimenting immediately after installation.

## Quickstart

### 1. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install codebook-lab
```

This installs CodeBook Lab from a package index so you can import it in your own scripts, notebooks, or analysis workflows.

If you plan to generate or score `textbox` annotations, install the optional textbox dependencies as well:

```bash
python -m pip install "codebook-lab[textbox]"
```

### 2. Install and start Ollama

Install Ollama on your machine, then make sure the local server is running:

```bash
ollama serve
```

If the default local Ollama server is not already running, CodeBook Lab will try to start it automatically when you run an experiment. It will also pull the requested Ollama model automatically if it is not already available locally.

### 3. Choose a model and task

The package ships with a bundled example task called `policy-sentiment`. Any Ollama model available on your machine can be used.

```python
task = "policy-sentiment"
model = "gemma3:270m"
```

You can inspect or copy bundled example tasks from Python:

```python
from codebook_lab import copy_example_task, list_example_tasks

print(list_example_tasks())
copy_example_task("policy-sentiment", "./my_tasks", overwrite=True)
```

Set `country_iso_code` to the country where the compute is physically running. This is used by CodeCarbon to convert energy use into emissions factors and should be a 3-letter ISO 3166-1 alpha-3 code such as `USA`, `IRL`, or `DEU`.

### 4. Run experiments from Python

Single experiment:

```python
from codebook_lab import ExperimentSpec, run_experiment

result = run_experiment(
    ExperimentSpec(
        task="policy-sentiment",
        model="gemma3:270m",
        use_examples=False,
        prompt_type="standard",
        temperature=None,
        top_p=None,
        process_textbox=True,
        country_iso_code="IRL",
    ),
    output_root="outputs",
)

print(result.experiment_directory)
print(result.metrics.summary_text)
```

If `process_textbox=True`, CodeBook Lab will calculate textbox similarity metrics such as ROUGE, cosine similarity, and BERTScore when the optional textbox dependencies are installed. Without them, the run still completes, but textbox metrics that rely on those packages will be reported as unavailable and the warning will tell you how to install them.

Parameter sweep:

```python
from codebook_lab import run_experiment_grid

results = run_experiment_grid(
    param_grid={
        "country_iso_code": "IRL",
        "tasks": ["policy-sentiment"],
        "models": ["gemma3:270m", "llama3.2:3b"],
        "use_examples": ["false", "true"],
        "prompt_types": ["standard", "persona"],
        "temperatures": ["None", "0.2"],
        "top_ps": ["None"],
        "process_textboxes": ["true"],
        "process_spans": ["false"],
    },
    output_root="outputs",
)

print(f"Completed {len(results)} runs")
```

Custom prompt wrapper:

```python
from codebook_lab import ExperimentSpec, PromptContext, register_prompt_wrapper, run_experiment

def concise_wrapper(context: PromptContext) -> str:
    return (
        "Annotate the text as carefully as possible.\n\n"
        f"{context.core_prompt}\n\n"
        f'Text:\n"{context.text}"\n\n'
        "Response:\n"
    )

register_prompt_wrapper("concise", concise_wrapper)

result = run_experiment(
    ExperimentSpec(
        task="policy-sentiment",
        model="gemma3:270m",
        prompt_type="concise",
        country_iso_code="IRL",
    )
)
```

### 5. Inspect the outputs

Each run creates a timestamped experiment directory under `outputs/<task>/` containing:

- `output.csv`: row-level model annotations
- `config.json`: the run configuration
- `classification_reports.txt`: per-label evaluation summaries
- `emissions.csv`: CodeCarbon output
- `timing_data.json`: inference timing summary
- `char_counts.json`: prompt and response character counts

Aggregate metrics are written to `outputs/metrics/<task>_metrics_log.csv`.

That metrics log stores both annotation-quality metrics and run metadata. Depending on the annotation type, it can include:

- classification metrics such as accuracy, precision, recall, F1, and percentage agreement
- inter-rater style agreement metrics such as Cohen's kappa and Krippendorff's alpha
- ordinal metrics for Likert labels such as Spearman correlation and quadratic weighted kappa
- textbox metrics such as normalized Levenshtein similarity, BLEU, ROUGE, cosine similarity, and BERTScore
- resource and run metadata such as CPU model, GPU model, total inference time, average inference time, total input characters, total output characters, energy consumed in kWh, and emissions in kg CO2eq

This makes it easy to compare not just which model is most accurate, but also which setup is fastest, cheapest to run, and most energy intensive.

Textbox note: normalized Levenshtein and BLEU work with the base install, but ROUGE, embedding-based cosine similarity, and BERTScore require the optional textbox extras. Install them with `python -m pip install "codebook-lab[textbox]"`.

## Experiment Configuration

Most multi-run setup happens through the parameter grid dictionary you pass into `run_experiment_grid(...)`.

- `tasks`: which task folders to run
- `models`: which Ollama models to evaluate (e.g. `gemma3:270m`, `llama3.2:3b`, `qwen3.5:latest`)
- `use_examples`: whether to include worked examples from the codebook in the LLM prompt (zero-shot vs. few-shot)
- `prompt_types`: which prompt wrapper to use (`standard`, `persona`, or `CoT`)
- `temperatures`: sampling temperature values (leave empty for model default)
- `top_ps`: nucleus sampling values (leave empty for model default)
- `process_textboxes`: whether textbox-style annotations should be generated and scored
- `process_spans`: whether span annotations should be generated and scored

When `process_textboxes` is enabled, install the optional textbox extras first if you want the full textbox metric suite:

```bash
python -m pip install "codebook-lab[textbox]"
```

Add multiple values to any field and the package sweeps them automatically. For a single quick run, keep one value in each field.

## Create Your Own Task

1. Create a local folder such as `my_tasks/my-task/`.
2. Annotate your data in [CodeBook Studio](https://codebook.streamlit.app/) and save the labeled file as `my_tasks/my-task/ground-truth.csv`.
3. Download the codebook JSON from Studio and save it as `my_tasks/my-task/codebook.json`.
4. Pass `task_root="my_tasks"` and `task="my-task"` into `ExperimentSpec(...)` when you run experiments.

If you are still designing a task and do not yet have human-coded labels, you can run annotation with `codebook_lab.run_annotation(...)` on an unlabeled CSV and add `ground-truth.csv` later when you want to score model performance with `codebook_lab.run_metrics(...)`.

## Human Reliability And Adjudication

When multiple human coders annotate the same items, CodeBook Lab can validate the coder CSVs, calculate inter-coder reliability, find disagreements, and build a consensus `ground-truth.csv`.

```python
from codebook_lab import build_human_ground_truth, calculate_human_reliability

coder_csvs = {
    "coder1": "annotations/coder1.csv",
    "coder2": "annotations/coder2.csv",
    "coder3": "annotations/coder3.csv",
}

reliability = calculate_human_reliability(
    codebook_path="codebook.json",
    coder_csvs=coder_csvs,
    output_dir="outputs/human_reliability",
)

ground_truth = build_human_ground_truth(
    codebook_path="codebook.json",
    coder_csvs=coder_csvs,
    output_dir="outputs/ground_truth",
)
```

Each coder CSV must contain a stable item identifier column. The default is `sample_id`; pass `id_column="..."` to use a different column. By default, coder assignments are inferred from the submitted files. To validate expected coverage, pass an optional assignment CSV in either long format (`sample_id,coder_id`) or wide format (`sample_id,ra_1,ra_2,...`).

Reliability outputs include `validation_issues.csv`, `pairwise_icr.csv`, `multirater_icr.csv`, `disagreements.csv`, and `summary.md`. Ground-truth outputs include `ground-truth.csv`, `adjudication_queue.csv`, and `validation_issues.csv`.

Rows without a strict majority are written to `adjudication_queue.csv`. Open that queue in CodeBook Studio's adjudication mode, fill the unresolved blanks, export the completed queue, then rebuild:

```python
resolved = build_human_ground_truth(
    codebook_path="codebook.json",
    coder_csvs=coder_csvs,
    adjudications_csv="adjudication_queue.csv",
    output_dir="outputs/ground_truth_resolved",
)
```

## Advanced Customization

If you want to go beyond the default wrappers and hyperparameters, `codebook_lab/annotate.py` and `codebook_lab/prompts.py` are the main extension points.

- To add new prompt wrappers beyond `standard`, `persona`, and `CoT`, register them from Python with `register_prompt_wrapper(...)` or extend the built-in registry in `codebook_lab/prompts.py`.
- To expose additional model hyperparameters such as `top_k`, add them to `setup_model()`, thread them through `run_annotation(...)` and `run_experiment(...)`, and add the corresponding field to the grid you pass into `run_experiment_grid(...)`.

## License

This project is licensed under the [GNU Affero General Public License v3.0](https://github.com/LorcanMcLaren/codebook-lab/blob/main/LICENSE).

## Citation

If you use CodeBook Lab in research, please cite both:

- this software package
- the associated arXiv preprint

Citation metadata is also available in the project's [`CITATION.cff`](https://github.com/LorcanMcLaren/codebook-lab/blob/main/CITATION.cff).

### Software Citation

APSR style:

McLaren, Lorcan. 2026. *CodeBook Lab* (Version v1.0.0) [Computer software]. Zenodo. [https://doi.org/10.5281/zenodo.19185921](https://doi.org/10.5281/zenodo.19185921).

BibTeX:

```bibtex
@software{mclaren_codebook_lab_2026,
  author = {McLaren, Lorcan},
  title = {CodeBook Lab},
  year = {2026},
  version = {v1.0.0},
  doi = {10.5281/zenodo.19185921},
  url = {https://doi.org/10.5281/zenodo.19185921}
}
```

### Preprint Citation

APSR style:

McLaren, Lorcan, James P. Cross, Zuzanna Krakowska, Robin Rauner, and Martijn Schoonvelde. 2026. *Magic Words or Methodical Work? Challenging Conventional Wisdom in LLM-Based Political Text Annotation*. arXiv preprint arXiv:2603.26898. [https://arxiv.org/abs/2603.26898](https://arxiv.org/abs/2603.26898).

BibTeX:

```bibtex
@misc{mclaren_magic_words_2026,
  author = {McLaren, Lorcan and Cross, James P. and Krakowska, Zuzanna and Rauner, Robin and Schoonvelde, Martijn},
  title = {Magic Words or Methodical Work? Challenging Conventional Wisdom in LLM-Based Political Text Annotation},
  year = {2026},
  eprint = {2603.26898},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  doi = {10.48550/arXiv.2603.26898},
  url = {https://arxiv.org/abs/2603.26898}
}
```
