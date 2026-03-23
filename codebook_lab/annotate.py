import json
import logging
from pathlib import Path
import sys
import time

import pandas as pd
import regex
from codecarbon import OfflineEmissionsTracker
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama
from pydantic import BaseModel

from .conditions import (
    get_annotation_column_name,
    get_annotation_entries,
    is_annotation_applicable,
    normalize_annotation_response_value,
)
from .ollama import ensure_ollama_available


class AnnotationResponse(BaseModel):
    """Schema used by ChatOllama structured output to guarantee valid JSON."""
    response: str


_PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""{question}""")
from .prompts import PromptContext, get_prompt_type_name, render_prompt
from .types import AnnotationRunResult

logger = logging.getLogger(__name__)


class _AnnotationProgressBar:
    """Render a compact terminal progress bar for annotation runs."""

    def __init__(self, total_steps: int, enabled: bool | None = None) -> None:
        self.total_steps = max(total_steps, 0)
        self.completed_steps = 0
        self.enabled = sys.stderr.isatty() if enabled is None else enabled
        self._last_message = ""

    def update(self, row_num: int, total_rows: int, annotation_name: str) -> None:
        """Advance the bar by one annotation and redraw it."""
        if self.total_steps == 0:
            return

        self.completed_steps += 1
        if not self.enabled:
            return

        width = 28
        progress = self.completed_steps / self.total_steps
        filled = int(width * progress)
        bar = "#" * filled + "-" * (width - filled)
        message = (
            f"\rAnnotating [{bar}] {self.completed_steps}/{self.total_steps} "
            f"({progress:.0%})  row {row_num}/{total_rows}  {annotation_name}"
        )
        message = message[:140]
        padding = max(0, len(self._last_message) - len(message))
        sys.stderr.write(message + (" " * padding))
        sys.stderr.flush()
        self._last_message = message

    def finish(self) -> None:
        """Terminate the in-place progress bar cleanly."""
        if self.enabled and self.total_steps > 0:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def skip(self, count: int = 1) -> None:
        """Reduce the remaining work estimate when prompts are skipped."""
        if count <= 0:
            return
        self.total_steps = max(self.completed_steps, self.total_steps - count)


def _count_annotations(codebook, process_textbox=False):
    """Count the maximum number of annotation prompts that could be issued for one row."""
    count = 0
    for _, _, _, annotation in get_annotation_entries(codebook):
        if annotation.get("type") == "textbox" and not process_textbox:
            continue
        count += 1
    return count

def load_codebook(codebook_path):
    """Load a CodeBook Studio/CodeBook Lab codebook JSON file.

    Args:
        codebook_path: Path to a ``codebook.json`` file.

    Returns:
        Parsed codebook dictionary.
    """
    with open(codebook_path, 'r') as file:
        codebook = json.load(file)
    return codebook

def get_annotation_column_names(codebook):
    """Return the annotation column names implied by a codebook structure.

    Args:
        codebook: Parsed codebook dictionary.

    Returns:
        List of column names in ``<section_name>_<annotation_name>`` format.
    """
    return [
        get_annotation_column_name(section_content, annotation)
        for _, section_content, _, annotation in get_annotation_entries(codebook)
    ]

def load_input_dataframe(csv_path, codebook):
    """Load the input CSV and remove any existing annotation label columns.

    Args:
        csv_path: Path to the input CSV containing the source text column.
        codebook: Parsed codebook dictionary describing annotation columns.

    Returns:
        Pandas DataFrame ready for annotation.
    """
    df = pd.read_csv(csv_path)
    annotation_columns = get_annotation_column_names(codebook)
    columns_to_drop = [column for column in annotation_columns if column in df.columns]

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        dropped_columns = ", ".join(columns_to_drop)
        logger.info(
            "Dropping annotation label columns from input before LLM annotation: %s",
            dropped_columns,
        )

    text_column = codebook["text_column"]
    if text_column not in df.columns:
        raise ValueError(
            f"Text column '{text_column}' was not found in {csv_path} after preparing the input data."
        )

    return df

def normalize_country_iso_code(country_iso_code):
    """Validate and normalize an ISO 3166-1 alpha-3 country code.

    Args:
        country_iso_code: Three-letter country code such as ``"USA"`` or ``"IRL"``.

    Returns:
        Uppercase three-letter country code.
    """
    normalized = country_iso_code.strip().upper()
    if len(normalized) != 3 or not normalized.isalpha():
        raise ValueError(
            "country_iso_code must be a 3-letter ISO 3166-1 alpha-3 country code, "
            "for example USA, IRL, or DEU."
        )
    return normalized


def setup_model(model_name, temperature=None, top_p=None):
    """Create the LangChain-Ollama pipeline used for annotation.

    Args:
        model_name: Ollama model identifier such as ``"gemma3:270m"``.
        temperature: Optional sampling temperature.
        top_p: Optional nucleus-sampling value.

    Returns:
        ``ChatOllama`` instance.  The caller builds structured-output chains
        from this model as needed.
    """
    model_kwargs = {}
    if temperature is not None:
        model_kwargs['temperature'] = float(temperature)
    if top_p is not None:
        model_kwargs['top_p'] = float(top_p)

    llm = ChatOllama(model=model_name, **model_kwargs)
    return llm

def generate_response(chain, prompt, char_counts, timing_data, row_num=None, annotation_name=None):
    """Run one prompt through the model and update timing/count statistics.

    Args:
        chain: ``ChatOllama`` instance returned by :func:`setup_model`.
        prompt: Fully rendered prompt string.
        char_counts: Mutable dict with ``input_chars`` and ``output_chars`` integers.
        timing_data: Mutable dict with inference timing counters.
        row_num: Optional 1-based row number for progress logging.
        annotation_name: Optional annotation label for progress logging.

    Returns:
        Raw model response string, or ``""`` if inference failed.
    """
    try:
        # Track input characters
        char_counts['input_chars'] += len(prompt)

        if row_num and annotation_name:
            logger.info("[Row %s] Sending request for: %s...", row_num, annotation_name)

        structured_chain = (
            _PROMPT_TEMPLATE
            | chain.with_structured_output(AnnotationResponse, include_raw=True)
        )

        start_time = time.time()
        result = structured_chain.invoke({"question": prompt})
        end_time = time.time()
        inference_time = end_time - start_time
        timing_data['total_inference_time'] += inference_time
        timing_data['inference_count'] += 1

        if result.get("parsed") is not None:
            response = result["parsed"].model_dump_json()
        else:
            raw = result.get("raw")
            response = raw.content if raw else ""
            logger.debug("Structured parsing failed, using raw response for %s", annotation_name)

        char_counts['output_chars'] += len(response)

        if row_num and annotation_name:
            logger.info("[Row %s] %s done (%.1fs)", row_num, annotation_name, inference_time)

        return response
    except Exception as e:
        logger.warning("Error generating response: %s", e)
        return ""

def extract_json_response(response, annotation_type, min_value=None, max_value=None, options=None):
    """
    Extract and validate JSON response based on annotation type
    
    Args:
        response: Raw model response text that should contain a JSON object.
        annotation_type: Annotation type string such as ``"dropdown"`` or ``"likert"``.
        min_value: Optional integer lower bound for Likert annotations.
        max_value: Optional integer upper bound for Likert annotations.
        options: Optional dropdown option list used to normalize categorical labels.

    Returns:
        Parsed response value coerced into the expected annotation format.
    """
    pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
    json_strings = pattern.findall(response)

    def normalize_dropdown_value(value):
        return normalize_annotation_response_value(
            {
                "type": "dropdown",
                "options": options or [],
            },
            value,
        )
    
    for json_string in json_strings:
        try:
            parsed_json = json.loads(json_string)
            response_value = parsed_json.get("response", "")
            
            # Validate and format based on annotation type
            if annotation_type == "dropdown":
                return normalize_dropdown_value(response_value)
            elif annotation_type == "checkbox":
                # Convert to 1 or 0
                if isinstance(response_value, bool):
                    return 1 if response_value else 0
                elif isinstance(response_value, int) and (response_value == 0 or response_value == 1):
                    return response_value
                elif isinstance(response_value, str):
                    if response_value.lower() in ["yes", "true", "1"]:
                        return 1
                    elif response_value.lower() in ["no", "false", "0"]:
                        return 0
                # Default to 0 if invalid
                return 0
            elif annotation_type == "textbox":
                # Return as string
                return str(response_value).strip()
            elif annotation_type == "likert":
                # Validate is within range and convert to int
                try:
                    value = int(float(response_value))
                    if min_value is not None and max_value is not None:
                        return max(min_value, min(max_value, value))  # Clamp to range
                    return value
                except (ValueError, TypeError):
                    # If not a valid number, return the middle of the scale if available
                    if min_value is not None and max_value is not None:
                        return (min_value + max_value) // 2
                    return response_value
            
            # Fallback
            return str(response_value).strip() if isinstance(response_value, str) else response_value
        except json.JSONDecodeError as e:
            logger.debug("Error parsing JSON: %s", e)
    
    # If no valid JSON, try to extract direct response
    stripped_response = response.strip()

    if annotation_type == "dropdown":
        return normalize_dropdown_value(stripped_response)
    elif annotation_type == "checkbox":
        if "yes" in response.lower() or "true" in response.lower():
            return 1
        elif "no" in response.lower() or "false" in response.lower():
            return 0
        return 0
    elif annotation_type == "likert" and min_value is not None and max_value is not None:
        # Try to find a number in the response
        numbers = regex.findall(r'\d+', response)
        for num in numbers:
            try:
                value = int(num)
                if min_value <= value <= max_value:
                    return value
            except ValueError:
                continue
        return (min_value + max_value) // 2  # Default to middle value
    elif annotation_type == "textbox":
        return stripped_response
    
    return None

def format_prompt(section_name, section_instruction, name, tooltip, annotation_type, 
               options=None, min_value=None, max_value=None, example=None, 
               text=None, prompt_type="standard", use_examples=False):
    """
    Format the prompt based on annotation type and specified prompt type
    
    Args:
        section_name: Codebook section name.
        section_instruction: Optional section-level instructions.
        name: Annotation name within the section.
        tooltip: Optional guidance text for the annotation.
        annotation_type: One of ``"dropdown"``, ``"checkbox"``, ``"likert"``, or ``"textbox"``.
        options: Dropdown option list when applicable.
        min_value: Minimum Likert value when applicable.
        max_value: Maximum Likert value when applicable.
        example: Optional example block from the codebook.
        text: Raw source text being annotated.
        prompt_type: Registered prompt wrapper name or callable wrapper.
        use_examples: Whether examples should be included in the prompt.

    Returns:
        Full prompt string ready to send to the model.
    """
    # Get response instructions based on annotation type
    response_instructions = _get_response_instructions(
        annotation_type, options, min_value, max_value
    )
    
    # Build the core prompt that's common to all prompt types
    core_prompt = _build_core_prompt(
        section_name, section_instruction, name, tooltip, 
        response_instructions, example, use_examples
    )
    
    context = PromptContext(
        section_name=section_name,
        section_instruction=section_instruction,
        annotation_name=name,
        tooltip=tooltip,
        annotation_type=annotation_type,
        options=options,
        min_value=min_value,
        max_value=max_value,
        example=example or "",
        text=text or "",
        use_examples=use_examples,
        response_instructions=response_instructions,
        core_prompt=core_prompt,
    )
    return render_prompt(prompt_type, context)


def _get_response_instructions(annotation_type, options=None, min_value=None, max_value=None):
    """Generate type-specific response instructions for a prompt.

    Args:
        annotation_type: Annotation type string.
        options: Dropdown options when ``annotation_type`` is ``"dropdown"``.
        min_value: Likert minimum when applicable.
        max_value: Likert maximum when applicable.

    Returns:
        Instruction string describing the expected response format.
    """
    if annotation_type == "dropdown" and options:
        options_str = ', or '.join(f'"{option}"' for option in options)
        return f"Respond only with one of the following options: {options_str}."
    elif annotation_type == "checkbox":
        return "Respond with 1 if \"Yes\" or 0 if \"No\"."
    elif annotation_type == "likert" and min_value is not None and max_value is not None:
        return f"Respond with a whole number from {min_value} to {max_value} (inclusive), where {min_value} means lowest and {max_value} means highest."
    elif annotation_type == "textbox":
        return "Respond with a brief text explanation."
    return ""


def _build_core_prompt(section_name, section_instruction, name, tooltip, 
                     response_instructions, example, use_examples):
    """Build the wrapper-agnostic prompt body for a single annotation.

    Args:
        section_name: Codebook section name.
        section_instruction: Optional section-level instructions.
        name: Annotation name within the section.
        tooltip: Optional annotation guidance text.
        response_instructions: String describing the expected response format.
        example: Optional example block from the codebook.
        use_examples: Whether example blocks should be included.

    Returns:
        Core prompt string before a prompt wrapper is applied.
    """
    core = f"{section_name}"
    
    if section_instruction:
        core += f"\n{section_instruction}"
        
    core += f"\n\n{name}"
    
    if tooltip:
        core += f"\n{tooltip}"
        
    if response_instructions:
        core += f"\n\n{response_instructions}"
    
    core += "\n\nReturn your response in JSON format, with the key \"response\"."
    
    if use_examples and example:
        core += f"\n\n{example}"
    elif not use_examples and example:
        # Check if example contains instruction text that might be needed
        if "Text:" not in example:
            core += f"\n\n{example}"
    
    return core


def _extract_task_name(csv_path):
    """Extract the task folder name from a CSV path when possible.

    Args:
        csv_path: Input CSV path, usually under ``tasks/<task_name>/``.

    Returns:
        Task name string if it can be inferred, otherwise ``None``.
    """
    task_name = None
    try:
        parts = str(csv_path).split('/')
        if 'tasks' in parts:
            task_idx = parts.index('tasks') + 1
            if task_idx < len(parts):
                task_name = parts[task_idx]
    except Exception:
        pass
    return task_name


def _normalize_optional_parameter(value):
    """Normalize blank or ``"None"`` CLI-style values to ``None``."""
    if value in (None, "", "None"):
        return None
    return value

def classify_text(chain, text, codebook, prompt_type="standard", use_examples=False,
                 char_counts=None, timing_data=None, process_textbox=False, row_num=None,
                 progress_bar=None, total_rows=None):
    """Annotate one text row across all sections in a codebook.

    Args:
        chain: Runnable returned by :func:`setup_model`.
        text: Raw source text to annotate.
        codebook: Parsed codebook dictionary.
        prompt_type: Registered prompt wrapper name or callable wrapper.
        use_examples: Whether codebook examples should be included in prompts.
        char_counts: Optional mutable counter dict for prompt/response characters.
        timing_data: Optional mutable timing dict for inference statistics.
        process_textbox: Whether textbox annotations should be generated.
        row_num: Optional 1-based row number for progress logging.
        progress_bar: Optional progress-bar helper updated after each annotation.
        total_rows: Optional total row count for progress rendering.

    Returns:
        Tuple of ``(responses, char_counts, timing_data)``.
    """
    responses = {}
    
    # Initialize character counts if not provided
    if char_counts is None:
        char_counts = {'input_chars': 0, 'output_chars': 0}
    
    # Initialize timing data if not provided
    if timing_data is None:
        timing_data = {'total_inference_time': 0, 'inference_count': 0}
    
    for section_key, section, annotation_key, annotation in get_annotation_entries(codebook):
        section_name = section['section_name']
        section_instruction = section.get('section_instruction', '')
        name = annotation['name']
        annotation_type = annotation['type']
        annotation_full_name = f"{section_name}_{name}"
        column_name = get_annotation_column_name(section, annotation)

        if annotation_type == "textbox" and not process_textbox:
            if progress_bar is not None:
                progress_bar.skip()
            continue

        if not is_annotation_applicable(codebook, section_key, annotation_key, responses):
            responses[column_name] = None
            if progress_bar is not None:
                progress_bar.skip()
            continue

        tooltip = annotation.get('tooltip', '')
        example = annotation.get('example', '')

        options = None
        min_value = None
        max_value = None

        if annotation_type == "dropdown":
            options = annotation.get('options', [])
        elif annotation_type == "likert":
            min_value = annotation.get('min_value')
            max_value = annotation.get('max_value')

        prompt = format_prompt(
            section_name,
            section_instruction,
            name,
            tooltip,
            annotation_type,
            options,
            min_value,
            max_value,
            example,
            text,
            prompt_type=prompt_type,
            use_examples=use_examples
        )

        response_text = generate_response(
            chain,
            prompt,
            char_counts,
            timing_data,
            row_num=row_num,
            annotation_name=annotation_full_name
        )
        response_value = extract_json_response(
            response_text,
            annotation_type,
            min_value,
            max_value,
            options=options,
        )

        responses[column_name] = response_value if response_value is not None else None

        if progress_bar is not None and row_num is not None and total_rows is not None:
            progress_bar.update(row_num, total_rows, annotation_full_name)

    return responses, char_counts, timing_data

def apply_classification_to_csv(csv_path, output_path, codebook, chain, prompt_type="standard", 
                              use_examples=False, process_textbox=False):
    """Run annotation over every row in an input CSV and write incremental results.

    Args:
        csv_path: Path to the input CSV file.
        output_path: Path where the annotated CSV should be written.
        codebook: Parsed codebook dictionary.
        chain: Runnable returned by :func:`setup_model`.
        prompt_type: Registered prompt wrapper name or callable wrapper.
        use_examples: Whether codebook examples should be included in prompts.
        process_textbox: Whether textbox annotations should be generated.

    Returns:
        Tuple of ``(classified_df, char_counts, timing_data)``.
    """
    df = load_input_dataframe(csv_path, codebook)
    
    logger.info("Starting classification of %d rows", len(df))

    annotations_per_row = _count_annotations(codebook, process_textbox)
    total_steps = len(df) * annotations_per_row
    progress_bar = _AnnotationProgressBar(total_steps)
    
    # Create a list to store all results
    results = []
    
    # Initialize character counts dictionary
    char_counts = {'input_chars': 0, 'output_chars': 0}
    
    # Initialize timing data dictionary
    timing_data = {'total_inference_time': 0, 'inference_count': 0}
    
    # Process each row individually
    try:
        for idx, row in df.iterrows():
            row_num = idx + 1
            text = row[codebook['text_column']]

            logger.info("[Row %d/%d] Starting annotations...", row_num, len(df))

            annotations, char_counts, timing_data = classify_text(
                chain,
                text,
                codebook,
                prompt_type,
                use_examples,
                char_counts,
                timing_data,
                process_textbox,
                row_num=row_num,
                progress_bar=progress_bar,
                total_rows=len(df),
            )

            # Add annotations to row data
            row_data = row.to_dict()
            row_data.update(annotations)
            results.append(row_data)

            # Save progress after each row
            pd.DataFrame(results).to_csv(output_path, index=False)

            avg_time = timing_data['total_inference_time'] / timing_data['inference_count'] if timing_data['inference_count'] > 0 else 0
            logger.info("[Row %d/%d] Complete! (avg: %.1fs per annotation)", row_num, len(df), avg_time)
    finally:
        progress_bar.finish()
    
    # Create final DataFrame
    classified_df = pd.DataFrame(results)
    classified_df.to_csv(output_path, index=False)
    
    # Calculate average inference time
    avg_inference_time = 0
    if timing_data['inference_count'] > 0:
        avg_inference_time = timing_data['total_inference_time'] / timing_data['inference_count']
    timing_data['avg_inference_time'] = avg_inference_time
    
    # Return character counts and timing data
    return classified_df, char_counts, timing_data

def run_annotation(
    *,
    model,
    csv_path,
    codebook_path,
    output_path,
    experiment_directory,
    prompt_type="standard",
    use_examples=False,
    temperature=None,
    top_p=None,
    process_textbox=False,
    country_iso_code="USA",
    start_ollama_if_needed=True,
):
    """Run one annotation job and persist its outputs to disk.

    Args:
        model: Ollama model identifier such as ``"gemma3:270m"``.
        csv_path: Path to the input CSV file to annotate.
        codebook_path: Path to the matching ``codebook.json`` file.
        output_path: Path where the annotated CSV should be written.
        experiment_directory: Directory for metadata and sidecar output files.
        prompt_type: Registered prompt wrapper name or callable wrapper.
        use_examples: Whether codebook examples should be included in prompts.
        temperature: Optional sampling temperature.
        top_p: Optional nucleus-sampling value.
        process_textbox: Whether textbox annotations should be generated.
        country_iso_code: Three-letter ISO 3166-1 alpha-3 country code for CodeCarbon.
        start_ollama_if_needed: If ``True``, try to start a local ``ollama serve``
            process when the default local server is not already reachable.
            Defaults to ``True`` so annotation runs can bring up the local Ollama
            server automatically when needed.

    Returns:
        :class:`codebook_lab.types.AnnotationRunResult` describing the completed run.
    """
    country_iso_code = normalize_country_iso_code(country_iso_code)
    temperature = _normalize_optional_parameter(temperature)
    top_p = _normalize_optional_parameter(top_p)
    ollama_base_url = ensure_ollama_available(start_if_needed=start_ollama_if_needed)

    experiment_directory = Path(experiment_directory)
    output_path = Path(output_path)
    experiment_directory.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task_name = _extract_task_name(csv_path)
    prompt_type_name = get_prompt_type_name(prompt_type)

    config = {
        "model": model,
        "prompt_type": prompt_type_name,
        "use_examples": bool(use_examples),
        "process_textbox": bool(process_textbox),
        "country_iso_code": country_iso_code,
        "task_name": task_name,
    }
    if temperature is not None:
        config["temperature"] = temperature
    if top_p is not None:
        config["top_p"] = top_p

    with open(experiment_directory / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    codebook = load_codebook(codebook_path)

    project_name = f"{model}_{prompt_type_name}_examples{str(bool(use_examples)).lower()}"
    if temperature is not None:
        project_name += f"_temp{temperature}"
    if top_p is not None:
        project_name += f"_topp{top_p}"

    tracker = OfflineEmissionsTracker(
        country_iso_code=country_iso_code,
        output_dir=str(experiment_directory),
        project_name=project_name,
        allow_multiple_runs=True,
        log_level='error'
    )
    tracker.start()

    try:
        chain = setup_model(model, temperature, top_p)
        classified_df, char_counts, timing_data = apply_classification_to_csv(
            str(csv_path),
            str(output_path),
            codebook,
            chain,
            prompt_type,
            bool(use_examples),
            bool(process_textbox),
        )
    finally:
        emissions = tracker.stop()

    with open(experiment_directory / "char_counts.json", "w") as f:
        json.dump(char_counts, f, indent=2)

    with open(experiment_directory / "timing_data.json", "w") as f:
        json.dump(timing_data, f, indent=2)

    logger.info("Classification complete. Results saved to %s", output_path)
    logger.info("Configuration: %s", config)
    logger.info("Country for emissions factors: %s", country_iso_code)
    logger.info("Ollama server: %s", ollama_base_url)
    logger.info("Estimated emissions: %s kg CO2eq", emissions)
    logger.info("Total input characters: %s", char_counts['input_chars'])
    logger.info("Total output characters: %s", char_counts['output_chars'])
    logger.info("Total inference time: %.2f seconds", timing_data['total_inference_time'])
    logger.info("Average inference time: %.2f seconds per call", timing_data['avg_inference_time'])

    return AnnotationRunResult(
        model=model,
        output_path=output_path,
        experiment_directory=experiment_directory,
        config=config,
        char_counts=char_counts,
        timing_data=timing_data,
        emissions=emissions,
        dataframe=classified_df,
    )
