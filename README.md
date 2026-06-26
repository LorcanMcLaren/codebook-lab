# CodeBook Lab

[![DOI](https://zenodo.org/badge/1186234207.svg)](https://doi.org/10.5281/zenodo.19185921) [![PyPI](https://img.shields.io/pypi/v/codebook-lab)](https://pypi.org/project/codebook-lab/) [![Python](https://img.shields.io/pypi/pyversions/codebook-lab)](https://pypi.org/project/codebook-lab/) [![License](https://img.shields.io/pypi/l/codebook-lab)](https://pypi.org/project/codebook-lab/)

CodeBook Lab is a validation-first experiment pipeline for LLM-based text annotation in computational social science. It takes a `codebook.json` and human-labelled `ground-truth.csv`, then runs controlled annotation experiments across model choice, prompt style, few-shot examples, chat mode, reasoning mode, and sampling settings.

The package is designed to work with [CodeBook Studio](https://codebook.streamlit.app/), which defines annotation tasks and collects human labels. Studio creates the task materials; Lab runs, scores, and compares LLM annotation experiments against those human labels.

## Documentation

Most user-facing documentation lives on the CodeBook docs site:

- [CodeBook Lab documentation](https://lorcanmclaren.com/codebook-lab/)
- [Installation guide](https://lorcanmclaren.com/codebook-lab/install.html)
- [Examples](https://lorcanmclaren.com/codebook-lab/examples.html)
- [CodeBook Studio guide](https://lorcanmclaren.com/codebook-lab/studio.html)
- [API reference](https://lorcanmclaren.com/codebook-lab/reference/index.html)
- [Citation information](https://lorcanmclaren.com/codebook-lab/citation.html)

## How Studio And Lab Fit Together

1. Define the annotation scheme in [CodeBook Studio](https://codebook.streamlit.app/).
2. Annotate texts with human coders.
3. Export `codebook.json` and save labelled data as `ground-truth.csv`.
4. Use CodeBook Lab to run LLM annotation experiments and compare outputs against the human labels.

For adjudicating human disagreements, Studio can complete the `adjudication_queue.csv` produced by Lab, then Lab can rebuild the final consensus `ground-truth.csv`.

## Quickstart

Install CodeBook Lab:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install codebook-lab
```

Install optional textbox metrics when you need ROUGE, cosine similarity, or BERTScore:

```bash
python -m pip install "codebook-lab[textbox]"
```

CodeBook Lab uses local Ollama models for LLM annotation. Install Ollama and make sure the server is running:

```bash
ollama serve
```

Run a bundled example task:

```python
from codebook_lab import ExperimentSpec, run_experiment

result = run_experiment(
    ExperimentSpec(
        task="policy-sentiment",
        model="gemma3:270m",
        country_iso_code="IRL",
    ),
    output_root="outputs",
)

print(result.experiment_directory)
print(result.metrics.summary_text)
```

For parameter sweeps, custom tasks, human reliability, adjudication, and output formats, see the [full documentation](https://lorcanmclaren.com/codebook-lab/).

## Repository Layout

- `codebook_lab/`: package source
- `codebook_lab/tasks/`: bundled example annotation tasks
- `tests/`: test suite
- `scripts/`: release and maintenance helpers
- `pyproject.toml`: package metadata and dependencies

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE).

## Citation

If you use CodeBook Lab in research, please cite the software package and, where relevant, the associated preprint. Citation metadata is available in [`CITATION.cff`](CITATION.cff) and on the [citation page](https://lorcanmclaren.com/codebook-lab/citation.html).

McLaren, Lorcan. 2026. *CodeBook Lab* (Version v1.4.0) [Computer software]. Zenodo. <https://doi.org/10.5281/zenodo.19185921>.
