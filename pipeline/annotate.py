import pandas as pd
import regex
import json
import argparse
import os
import sys
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from codecarbon import OfflineEmissionsTracker
import time

def load_codebook(codebook_path):
    with open(codebook_path, 'r') as file:
        codebook = json.load(file)
    return codebook

def normalize_country_iso_code(country_iso_code):
    normalized = country_iso_code.strip().upper()
    if len(normalized) != 3 or not normalized.isalpha():
        raise ValueError(
            "country_iso_code must be a 3-letter ISO 3166-1 alpha-3 country code, "
            "for example USA, IRL, or DEU."
        )
    return normalized

def setup_model(model_name, temperature=None, top_p=None):
    """
    Set up the model with optional temperature and top_p parameters
    """
    # Handle string 'None' values
    if temperature == 'None':
        temperature = None
    if top_p == 'None':
        top_p = None
    
    # Only include parameters if they're provided
    model_kwargs = {}
    if temperature is not None:
        model_kwargs['temperature'] = float(temperature)
    if top_p is not None:
        model_kwargs['top_p'] = float(top_p)
    
    llm = OllamaLLM(model=model_name, **model_kwargs)
    prompt_template = ChatPromptTemplate.from_template("""{question}""")
    chain = prompt_template | llm
    return chain

def generate_response(chain, prompt, char_counts, timing_data, row_num=None, annotation_name=None):
    try:
        # Track input characters
        char_counts['input_chars'] += len(prompt)
        
        # Print BEFORE inference
        if row_num and annotation_name:
            print(f"[Row {row_num}] Sending request for: {annotation_name}...", flush=True, end='')
            sys.stdout.flush()
        
        # Start timing
        start_time = time.time()
        
        # Generate response
        response = chain.invoke({"question": prompt})
        
        # End timing and add to timing data
        end_time = time.time()
        inference_time = end_time - start_time
        timing_data['total_inference_time'] += inference_time
        timing_data['inference_count'] += 1
        
        # Track output characters
        char_counts['output_chars'] += len(response)
        
        # Print AFTER inference
        if row_num and annotation_name:
            print(f" done ({inference_time:.1f}s)", flush=True)
            sys.stdout.flush()
        
        return response
    except Exception as e:
        print(f"\nError generating response: {e}", flush=True)
        sys.stdout.flush()
        return ""

def extract_json_response(response, annotation_type, min_value=None, max_value=None):
    """
    Extract and validate JSON response based on annotation type
    
    Args:
        response: LLM response text
        annotation_type: Type of annotation (dropdown, checkbox, textbox, likert)
        min_value: Minimum value for likert scale (optional)
        max_value: Maximum value for likert scale (optional)
    """
    pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
    json_strings = pattern.findall(response)
    
    for json_string in json_strings:
        try:
            parsed_json = json.loads(json_string)
            response_value = parsed_json.get("response", "")
            
            # Validate and format based on annotation type
            if annotation_type == "dropdown":
                return response_value
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
                return str(response_value)
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
            return response_value
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}", flush=True)
    
    # If no valid JSON, try to extract direct response
    if annotation_type == "checkbox":
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
    
    return response  # Return raw response as fallback

def format_prompt(section_name, section_instruction, name, tooltip, annotation_type, 
               options=None, min_value=None, max_value=None, example=None, 
               text=None, prompt_type="standard", use_examples=False):
    """
    Format the prompt based on annotation type and specified prompt type
    
    Args:
        section_name: Name of the annotation section
        section_instruction: Instruction for the section from the codebook
        name: Name of the annotation
        tooltip: Tooltip text explaining the annotation
        annotation_type: Type of annotation (dropdown, checkbox, textbox, likert)
        options: List of possible options (for dropdown)
        min_value: Minimum value for likert scale
        max_value: Maximum value for likert scale
        example: Example text from the codebook
        text: The text to classify
        prompt_type: Type of prompt formatting ("standard", "persona", "CoT")
        use_examples: Whether to include examples from the codebook
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
    
    # Customize the prompt based on prompt type
    if prompt_type == "persona":
        return _add_persona_wrapper(core_prompt, text)
    elif prompt_type == "CoT":
        return _add_cot_wrapper(core_prompt, text)
    else:  # Default "standard"
        return _add_standard_wrapper(core_prompt, text)


def _get_response_instructions(annotation_type, options=None, min_value=None, max_value=None):
    """Helper function to generate response instructions based on annotation type"""
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
    """Build the core prompt with consistent field ordering"""
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


def _add_standard_wrapper(core_prompt, text):
    """Add standard wrapper to the core prompt"""
    return f"{core_prompt}\n\n---\n\nText: \n\"{text}\"\n\nResponse: \n"


def _add_persona_wrapper(core_prompt, text):
    """Add persona-based wrapper to the core prompt"""
    prefix = "You are an expert political scientist and data annotator with extensive experience in analyzing political discourse and parliamentary debates.\n\nTask: Annotate the following text using the criteria below. Your annotation should be precise, consistent, and based solely on the text content.\n\n"
    
    suffix = f"\n\n---\n\nText: \n\"{text}\"\n\nResponse: \n"
    
    return f"{prefix}{core_prompt}{suffix}"


def _add_cot_wrapper(core_prompt, text):
    """Add Chain-of-Thought wrapper to the core prompt"""
    suffix = "\n\nI'll think through this step by step:\n\n"
    suffix += "1. First, I'll identify key parts of the text relevant to this dimension\n"
    suffix += "2. Next, I'll analyze how these elements relate to the annotation criteria\n"
    suffix += "3. Then, I'll consider my assessment carefully\n"
    suffix += "4. Finally, I'll make my selection based on this analysis\n"
    
    suffix += f"\n\n---\n\nText: \n\"{text}\"\n\n"
    suffix += "Step-by-step analysis:\n\n[Think through your reasoning here]\n\n"
    suffix += "Response: \n"
    
    return f"{core_prompt}{suffix}"

def classify_text(chain, text, codebook, prompt_type="standard", use_examples=False, 
                 char_counts=None, timing_data=None, process_textbox=False, row_num=None):
    """
    Classify text according to codebook with all annotation types
    """
    responses = {}
    
    # Initialize character counts if not provided
    if char_counts is None:
        char_counts = {'input_chars': 0, 'output_chars': 0}
    
    # Initialize timing data if not provided
    if timing_data is None:
        timing_data = {'total_inference_time': 0, 'inference_count': 0}
    
    for key, section in codebook.items():
        if key.startswith('section_'):
            section_name = section['section_name']
            section_instruction = section.get('section_instruction', '')
            annotations = section['annotations']
            
            for annotation_key, annotation in annotations.items():
                name = annotation['name']
                annotation_type = annotation['type']
                
                # Skip textbox type annotations if process_textbox is False
                if annotation_type == "textbox" and not process_textbox:
                    continue
                
                tooltip = annotation.get('tooltip', '')
                example = annotation.get('example', '')
                
                # Get type-specific parameters
                options = None
                min_value = None
                max_value = None
                
                if annotation_type == "dropdown":
                    options = annotation.get('options', [])
                elif annotation_type == "likert":
                    min_value = annotation.get('min_value')
                    max_value = annotation.get('max_value')

                # Format prompt based on specified type and annotation type
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
                
                annotation_full_name = f"{section_name}_{name}"
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
                    max_value
                )
                
                if response_value is not None:
                    # Store the response with a meaningful column name
                    column_name = f"{section_name}_{name}"
                    responses[column_name] = response_value

    return responses, char_counts, timing_data

def apply_classification_to_csv(csv_path, output_path, codebook, chain, prompt_type="standard", 
                              use_examples=False, process_textbox=False):
    df = pd.read_csv(csv_path)
    
    print(f"Starting classification of {len(df)} rows", flush=True)
    sys.stdout.flush()
    
    # Create a list to store all results
    results = []
    
    # Initialize character counts dictionary
    char_counts = {'input_chars': 0, 'output_chars': 0}
    
    # Initialize timing data dictionary
    timing_data = {'total_inference_time': 0, 'inference_count': 0}
    
    # Process each row individually
    for idx, row in df.iterrows():
        row_num = idx + 1
        text = row[codebook['text_column']]
        
        print(f"\n[Row {row_num}/{len(df)}] Starting annotations...", flush=True)
        sys.stdout.flush()
        
        annotations, char_counts, timing_data = classify_text(
            chain, 
            text, 
            codebook, 
            prompt_type, 
            use_examples, 
            char_counts,
            timing_data,
            process_textbox,
            row_num=row_num
        )
        
        # Add annotations to row data
        row_data = row.to_dict()
        row_data.update(annotations)
        results.append(row_data)
        
        # Save progress after each row
        pd.DataFrame(results).to_csv(output_path, index=False)
        
        # Calculate and print progress stats
        avg_time = timing_data['total_inference_time'] / timing_data['inference_count'] if timing_data['inference_count'] > 0 else 0
        print(f"[Row {row_num}/{len(df)}] Complete! (avg: {avg_time:.1f}s per annotation)", flush=True)
        sys.stdout.flush()
    
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

def main():
    parser = argparse.ArgumentParser(description="Run text classification using LangChain with Ollama.")
    
    # Existing arguments
    parser.add_argument("model", type=str, help="The Ollama model to use.")
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file.")
    parser.add_argument("codebook_path", type=str, help="Path to the codebook JSON file.")
    parser.add_argument("output_path", type=str, help="Path to save the output CSV file.")
    parser.add_argument("experiment_directory", type=str, help="Path to experiment directory.")
    
    # Hyperparameter arguments
    parser.add_argument("--prompt_type", type=str, default="standard", 
                       choices=["standard", "persona", "CoT"],
                       help="Type of prompt formatting to use.")
    parser.add_argument("--use_examples", type=str, default="false", 
                       choices=["true", "false"],
                       help="Whether to include examples from the codebook in the prompt.")
    parser.add_argument("--temperature", type=str, default=None, 
                       help="Temperature for model generation. If not provided, use model default.")
    parser.add_argument("--top_p", type=str, default=None, 
                       help="Top-p (nucleus sampling) parameter. If not provided, use model default.")
    parser.add_argument("--process_textbox", type=str, default="false", 
                       choices=["true", "false"],
                       help="Whether to process textbox type annotations.")
    parser.add_argument("--country_iso_code", type=str, default="USA",
                       help="3-letter ISO 3166-1 alpha-3 country code used by CodeCarbon for emissions calculations.")
    
    args = parser.parse_args()
    
    # Convert string bool to actual bool
    use_examples = args.use_examples.lower() == "true"
    process_textbox = args.process_textbox.lower() == "true"
    country_iso_code = normalize_country_iso_code(args.country_iso_code)
    
    # Ensure experiment directory exists
    os.makedirs(args.experiment_directory, exist_ok=True)
    
    # Extract task name from csv_path
    task_name = None
    try:
        parts = args.csv_path.split('/')
        if 'tasks' in parts:
            task_idx = parts.index('tasks') + 1
            if task_idx < len(parts):
                task_name = parts[task_idx]
    except:
        pass
    
    # Save experiment configuration
    config = {
        "model": args.model,
        "prompt_type": args.prompt_type,
        "use_examples": use_examples,
        "process_textbox": process_textbox,
        "country_iso_code": country_iso_code,
        "task_name": task_name,
    }
    
    # Only include parameters in config if explicitly provided
    if args.temperature is not None and args.temperature != "None":
        config["temperature"] = args.temperature
    if args.top_p is not None and args.top_p != "None":
        config["top_p"] = args.top_p
    
    with open(os.path.join(args.experiment_directory, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Load the codebook
    codebook = load_codebook(args.codebook_path)

    # Start emissions tracking with a suitable project name
    project_name = f"{args.model}_{args.prompt_type}_examples{args.use_examples}"
    if args.temperature is not None and args.temperature != "None":
        project_name += f"_temp{args.temperature}"
    if args.top_p is not None and args.top_p != "None":
        project_name += f"_topp{args.top_p}"
    
    tracker = OfflineEmissionsTracker(
        country_iso_code=country_iso_code,
        output_dir=args.experiment_directory, 
        project_name=project_name,
        allow_multiple_runs=True,
        log_level='error'  # Reduce CodeCarbon logging verbosity
    )
    tracker.start()
    
    # Setup the model with temperature and top_p
    chain = setup_model(args.model, args.temperature, args.top_p)

    # Apply classification to CSV and track metrics
    classified_df, char_counts, timing_data = apply_classification_to_csv(
        args.csv_path, 
        args.output_path, 
        codebook, 
        chain, 
        args.prompt_type,
        use_examples,
        process_textbox
    )

    # Stop emissions tracking
    emissions = tracker.stop()
    
    # Save character counts to a file in the experiment directory
    with open(os.path.join(args.experiment_directory, "char_counts.json"), "w") as f:
        json.dump(char_counts, f, indent=2)
    
    # Save timing data to a file in the experiment directory
    with open(os.path.join(args.experiment_directory, "timing_data.json"), "w") as f:
        json.dump(timing_data, f, indent=2)
    
    # Print completion information
    print(f"\nClassification complete. Results saved to {args.output_path}", flush=True)
    print(f"Configuration: {config}", flush=True)
    print(f"Country for emissions factors: {country_iso_code}", flush=True)
    print(f"Estimated emissions: {emissions} kg CO2eq", flush=True)
    print(f"Total input characters: {char_counts['input_chars']}", flush=True)
    print(f"Total output characters: {char_counts['output_chars']}", flush=True)
    print(f"Total inference time: {timing_data['total_inference_time']:.2f} seconds", flush=True)
    print(f"Average inference time: {timing_data['avg_inference_time']:.2f} seconds per call", flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
