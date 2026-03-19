# CodeBook Lab

This repository is a public-facing tutorial version of an LLM annotation pipeline designed for computational social scientists and political scientists. It is meant to make text annotation workflows much more accessible across a wide range of technical backgrounds.

The central idea is that researchers can define their own annotation tasks with CodeBook Studio, then use this pipeline to run and compare LLM-based annotation experiments without writing custom annotation code from scratch each time. Human annotations serve as the validation benchmark against which model performance is measured, making it possible to compare LLM outputs to a researcher-defined gold standard. Both CodeBook Studio and the LLM workflow support the main annotation types researchers commonly need: binary labels, categorical labels, Likert-scale ordinal labels, and open-ended text responses. That makes it easier to focus on substantive social science questions while still comparing the tradeoffs of different implementation choices.

In practice, the workflow looks like this:

- define a task and codebook in CodeBook Studio
- bring the exported `codebook.json` and your CSV data into this repository
- control the experiment through `param_grid.yaml`
- run the pipeline and compare performance, timing, and energy tradeoffs

The key idea is that experiments are controlled almost entirely through `param_grid.yaml`. In most cases, you do not need to edit the pipeline code to try a new task, model, or prompt setup. You update the codebook and data for a task, adjust `param_grid.yaml`, and rerun the experiment.

The tutorial keeps the workflow simple:
1. Pick a task with a `sample.csv`, `ground-truth.csv`, and `codebook.json`.
2. Choose one or more local Ollama models in `param_grid.yaml`.
3. Run the annotation pipeline.
4. Inspect the generated outputs and metrics.

## Contents

- [Why This Exists](#why-this-exists)
- [Repository Layout](#repository-layout)
- [Quickstart](#quickstart)
- [Tutorial Walkthrough](#tutorial-walkthrough)
- [Create Your Own Task](#create-your-own-task)
- [Running on HPC](#running-on-hpc)
- [Advanced Customization](#advanced-customization)
- [Citation](#citation)

## Why This Exists

This pipeline is paired with the CodeBook Studio web app, both developed by Lorcan McLaren, so that researchers can:

- define their own annotation tasks instead of relying only on prebuilt examples
- run classification and other structured text annotation tasks with local LLMs
- work with binary, categorical, Likert-scale, and open-ended text annotations in the same workflow
- compare the effects of different prompt and decoding choices
- validate model outputs against human annotations used as a benchmark
- evaluate accuracy, agreement, speed, and energy tradeoffs in a consistent way
- spend less time building bespoke annotation infrastructure for each new project

This combination is especially useful for research teams with varied technical proficiencies, because the codebook and experiment configuration are separated from the lower-level pipeline code.

## Repository Layout

- `pipeline/annotate.py`: runs annotation over a CSV using a codebook-driven prompt.
- `pipeline/metrics.py`: compares model outputs against ground truth and logs metrics.
- `scripts/run_local.sh`: local experiment runner using Ollama.
- `scripts/run_hpc_slurm.sh`: simple SLURM template that calls the same runner.
- `tasks/policy-sentiment/`: synthetic starter task showing checkbox, dropdown, Likert, and textbox annotations in one workflow.
- `tasks/sentiment/`: an additional sentiment example task.
- `tasks/approval/`: an additional Likert-style approval example task.
- `param_grid.yaml`: the main experiment control file. It determines which tasks, models, and settings are swept over in a run.

## Quickstart

### 1. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Install and start Ollama

Install Ollama on your machine, then make sure the local server is running:

```bash
ollama serve
```

You can also let `scripts/run_local.sh` start it automatically if it is not already running.

### 3. Choose a model and task

The default tutorial config in `param_grid.yaml` runs the `policy-sentiment` task with `gemma3:270m`.

```yaml
# Environment setting
country_iso_code: "USA"

# Experiment sweep settings
tasks: ["policy-sentiment"]
models: ["gemma3:270m"]
```

Any model available through Ollama can be used in this pipeline. In `param_grid.yaml`, use Ollama model naming conventions such as `gemma3:270m`, `llama3.2:3b`, or `qwen3.5:latest`.

Set `country_iso_code` to the country where the compute is physically running. This is used by CodeCarbon to convert energy use into emissions factors, so it should be a 3-letter ISO 3166-1 alpha-3 code such as `USA`, `IRL`, or `DEU`.

For local runs, this usually means your own country. For HPC runs, it should be the country where the cluster or data center is located, not necessarily the country you are connecting from.

You can switch to `approval`, add more models, or vary prompt settings by editing the YAML file. In normal use, this file is the main control surface for the whole experiment.

### 4. Run the tutorial experiment

```bash
bash scripts/run_local.sh
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

Because emissions are estimated from energy use using country-specific factors, you should make sure `country_iso_code` in `param_grid.yaml` matches the location of the machine actually doing the computation. On HPC, that means the cluster's country.

This makes it easy to compare not just which model is most accurate, but also which setup is fastest, cheapest to run, and most energy intensive.

## Tutorial Walkthrough

### Step 1: Understand the task files

Each task directory contains:

- `sample.csv`: the input rows to annotate
- `ground-truth.csv`: the reference labels used for evaluation
- `codebook.json`: the annotation instructions and output format

The codebook controls which text column is read, how prompts are worded, and what valid label formats look like.

### Step 2: Start with `policy-sentiment`

`tasks/policy-sentiment/` is the best first example because it is synthetic, easy to share publicly, and demonstrates all four annotation types supported by CodeBook Studio and the pipeline:

- checkbox / binary annotation
- dropdown / categorical annotation
- Likert / ordinal annotation
- textbox / open-ended annotation

It is designed as a lightweight public tutorial task for computational social science and political science workflows.

### Step 3: Try a second task

`tasks/approval/` and `tasks/sentiment/` show how the same pipeline can also be used on more specialized task setups.

To run it, change:

```yaml
tasks: ["approval"]
```

### Step 4: Sweep over settings

You can run small experiments just by editing `param_grid.yaml`. The runner evaluates every combination implied by the fields in that file.

The main fields are:

- `country_iso_code`: a one-time environment setting for CodeCarbon emissions calculations, usually changed only when you move between countries or clusters in different countries

The remaining fields are the ones you typically sweep over:

- `tasks`: which task folders to run, for example `sentiment` or `approval`
- `models`: which Ollama models to evaluate, using Ollama's own model-name format such as `gemma3:270m` or `qwen3.5:latest`
- `use_examples`: whether to include the worked examples from the codebook in the LLM prompt
- `prompt_types`: which prompt template to use; the tutorial pipeline supports `standard`, `persona`, and `CoT`
- `temperatures`: generation temperature values; leave this empty to use the model default
- `top_ps`: nucleus sampling values; leave this empty to use the model default
- `process_textboxes`: whether textbox-style annotations should be generated and scored

For example:

- adding more items to `models` compares several local models on the same task
- changing `tasks` runs the same setup on different annotation problems
- toggling `use_examples` compares zero-shot and example-augmented prompting
- varying `prompt_types`, `temperatures`, and `top_ps` lets you test prompt style and hyperparameter choices

If you only want a single quick run, keep one value in each field. If you want a broader comparison, add multiple values and let the runner sweep them automatically.

## Create Your Own Task

You are not limited to the example tasks in this repository. You can define your own annotation task by preparing a dataset and creating a JSON codebook with CodeBook Studio.

CodeBook Studio is a codebook-driven text annotation app for computational social science. It lets you:

- choose the header and text columns from your CSV
- define sections and annotation questions
- use checkbox, dropdown, Likert, and textbox response types
- add instructions and examples for annotators and LLM prompts
- preview the prompts that will be sent to the LLM
- download the codebook as JSON for direct use in this pipeline

The public app is available at [codebook.streamlit.app](https://codebook.streamlit.app/).

To add your own task to this tutorial repo:

1. Create a new folder such as `tasks/my-task/`.
2. Add your input data as `sample.csv`.
3. Create your codebook in CodeBook Studio and download the JSON file.
4. Save that JSON file as `tasks/my-task/codebook.json`.
5. If you want evaluation metrics, add a labeled `tasks/my-task/ground-truth.csv`.
6. Update `param_grid.yaml` to include your task name.

The JSON codebook used by CodeBook Studio matches the structure expected by this pipeline. In practice, that means your codebook should define:

- `header_column`: the column shown as the title or identifier during annotation
- `text_column`: the text field the model should annotate
- one or more `section_*` blocks
- within each section, one or more `annotations`

Each annotation can include:

- `name`
- `type`
- `tooltip`
- `example`
- `options` for dropdown fields
- `min_value` and `max_value` for Likert fields

If you are still designing a task and do not yet have gold labels, you can still run annotation with `pipeline/annotate.py` and add `ground-truth.csv` later when you want to score model performance with `pipeline/metrics.py`.

## Running on HPC

Use `scripts/run_hpc_slurm.sh` as a starting point. It is intentionally minimal so you can adapt the module loads, conda environment, and resource requests to your own cluster.

## Advanced Customization

If you want to go beyond the default wrappers and hyperparameters, `pipeline/annotate.py` is the place to extend the tutorial pipeline.

- To add new prompt wrappers beyond `standard`, `persona`, and `CoT`, extend the prompt-formatting logic in `pipeline/annotate.py` and then expose the new wrapper name through the `prompt_type` argument and `param_grid.yaml`.
- To expose additional model hyperparameters such as `top_k`, add them to `setup_model()`, add a command-line argument in `pipeline/annotate.py`, pass them through from `scripts/run_local.sh`, and add the corresponding field to `param_grid.yaml`.

That way, you can keep the same configuration-driven workflow while expanding what the experiment runner is able to sweep over.

## Citation

If you use this repository in research, please cite both:

- this software repository
- the associated preprint

The repository includes a [`CITATION.cff`](/Users/lorcanmclaren/Python/codebook-lab/CITATION.cff) file to make the software citable.

The associated preprint is:

McLaren, L., Cross, J. P., Krakowska, Z., Rauner, R., and Schoonvelde, M. (2026). *Magic Words or Methodical Work? Challenging Conventional Wisdom in LLM-Based Political Text Annotation*.
