# CodeBook Lab

CodeBook Lab is an LLM annotation experiment pipeline for computational social science. It takes a codebook and labelled dataset from [CodeBook Studio](https://codebook.streamlit.app/) ([source](https://github.com/LorcanMcLaren/codebook-studio)) and runs structured experiments across the dimensions that matter for text-as-data research: model choice, model size, prompt style, zero-shot versus few-shot learning, and sampling hyperparameters — all benchmarked against human labels.

Experiments are controlled through a single `param_grid.yaml` file rather than by editing pipeline code. Because the codebook and labelled data stay constant across runs, each dimension can be isolated and compared against the same human labels.

For a step-by-step walkthrough covering both tools, see the [CodeBook Studio & Lab Tutorial](https://lorcanmclaren.com/codebook-tutorial.html).

## Contents

- [How It Fits With CodeBook Studio](#how-it-fits-with-codebook-studio)
- [Repository Layout](#repository-layout)
- [Quickstart](#quickstart)
- [Experiment Configuration](#experiment-configuration)
- [Create Your Own Task](#create-your-own-task)
- [Running on HPC](#running-on-hpc)
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

## Repository Layout

- `pipeline/annotate.py`: runs annotation over a CSV using a codebook-driven prompt.
- `pipeline/metrics.py`: compares model outputs against ground truth and logs metrics.
- `scripts/run_local.sh`: local experiment runner using Ollama.
- `scripts/run_hpc_slurm.sh`: simple SLURM template that calls the same runner.
- `tasks/policy-sentiment/`: a ready-to-use example task covering all four annotation types (binary, categorical, Likert, and open-ended text).
- `param_grid.yaml`: the main experiment control file.

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

The default config in `param_grid.yaml` runs the `policy-sentiment` task with `gemma3:270m`. Any model available through Ollama can be used.

```yaml
# Environment setting
country_iso_code: "USA"

# Experiment sweep settings
tasks: ["policy-sentiment"]
models: ["gemma3:270m"]
```

Set `country_iso_code` to the country where the compute is physically running. This is used by CodeCarbon to convert energy use into emissions factors and should be a 3-letter ISO 3166-1 alpha-3 code such as `USA`, `IRL`, or `DEU`.

### 4. Run the experiment

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

This makes it easy to compare not just which model is most accurate, but also which setup is fastest, cheapest to run, and most energy intensive.

## Experiment Configuration

Most experiment setup happens through `param_grid.yaml`. The runner evaluates every combination implied by the fields in that file.

- `tasks`: which task folders to run
- `models`: which Ollama models to evaluate (e.g. `gemma3:270m`, `llama3.2:3b`, `qwen3.5:latest`)
- `use_examples`: whether to include worked examples from the codebook in the LLM prompt (zero-shot vs. few-shot)
- `prompt_types`: which prompt wrapper to use (`standard`, `persona`, or `CoT`)
- `temperatures`: sampling temperature values (leave empty for model default)
- `top_ps`: nucleus sampling values (leave empty for model default)
- `process_textboxes`: whether textbox-style annotations should be generated and scored

Add multiple values to any field and the runner sweeps them automatically. For a single quick run, keep one value in each field.

## Create Your Own Task

1. Create a new folder such as `tasks/my-task/`.
2. Annotate your data in [CodeBook Studio](https://codebook.streamlit.app/) and save the labeled file as `tasks/my-task/ground-truth.csv`.
3. Download the codebook JSON from Studio and save it as `tasks/my-task/codebook.json`.
4. Update `param_grid.yaml` to include your task name.

If you are still designing a task and do not yet have human-coded labels, you can still run annotation with `pipeline/annotate.py` on an unlabeled CSV and add `ground-truth.csv` later when you want to score model performance with `pipeline/metrics.py`.

## Running on HPC

Use `scripts/run_hpc_slurm.sh` as a starting point. It is intentionally minimal so you can adapt the module loads, conda environment, and resource requests to your own cluster.

## Advanced Customization

If you want to go beyond the default wrappers and hyperparameters, `pipeline/annotate.py` is the place to extend the pipeline.

- To add new prompt wrappers beyond `standard`, `persona`, and `CoT`, extend the prompt-formatting logic in `pipeline/annotate.py` and then expose the new wrapper name through the `prompt_type` argument and `param_grid.yaml`.
- To expose additional model hyperparameters such as `top_k`, add them to `setup_model()`, add a command-line argument in `pipeline/annotate.py`, pass them through from `scripts/run_local.sh`, and add the corresponding field to `param_grid.yaml`.

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).

## Citation

If you use this repository in research, please cite both:

- this software repository
- the associated preprint

The repository includes a [`CITATION.cff`](CITATION.cff) file for the software citation used by GitHub's citation interface.

### Software Citation

APSR style:

McLaren, Lorcan. 2026. *CodeBook Lab* (Version 0.1.0) [Computer software]. [https://github.com/LorcanMcLaren/codebook-lab](https://github.com/LorcanMcLaren/codebook-lab).

BibTeX:

```bibtex
@software{mclaren_codebook_lab_2026,
  author = {McLaren, Lorcan},
  title = {CodeBook Lab},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/LorcanMcLaren/codebook-lab}
}
```

### Preprint Citation

APSR style:

McLaren, Lorcan, James P. Cross, Zuzanna Krakowska, Robin Rauner, and Martijn Schoonvelde. 2026. *Magic Words or Methodical Work? Challenging Conventional Wisdom in LLM-Based Political Text Annotation*. Preprint.

BibTeX:

```bibtex
@misc{mclaren_magic_words_2026,
  author = {McLaren, Lorcan and Cross, James P. and Krakowska, Zuzanna and Rauner, Robin and Schoonvelde, Martijn},
  title = {Magic Words or Methodical Work? Challenging Conventional Wisdom in LLM-Based Political Text Annotation},
  year = {2026},
  note = {Preprint}
}
```
