import contextlib
import importlib.util
import io
import logging
from functools import lru_cache
import warnings

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, cohen_kappa_score
import numpy as np
import os
import uuid
from pathlib import Path
import json
from datetime import datetime
import krippendorff
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix

from .conditions import (
    get_annotation_column_name,
    get_annotation_condition,
    get_annotation_entries,
    get_annotation_lookup,
    normalize_annotation_response_value,
)
from .span_metrics import compute_span_metrics
from .types import MetricsRunResult

logger = logging.getLogger(__name__)


TEXTBOX_DEPENDENCY_GUIDANCE = (
    "Install the optional textbox extras to enable the full textbox metric suite: "
    "python -m pip install \"codebook-lab[textbox]\"."
)

BERT_SCORE_NUMPY_WARNING = (
    "The given NumPy array is not writable, and PyTorch does not support non-writable tensors."
)


@contextlib.contextmanager
def _quiet_optional_nlp_output():
    """Suppress noisy but non-actionable output from optional NLP metric libraries."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=BERT_SCORE_NUMPY_WARNING, category=UserWarning)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                from transformers.utils import logging as transformers_logging
            except ImportError:
                transformers_logging = None

            if transformers_logging is None:
                yield
                return

            previous_verbosity = transformers_logging.get_verbosity()
            transformers_logging.set_verbosity_error()
            try:
                yield
            finally:
                transformers_logging.set_verbosity(previous_verbosity)


@lru_cache(maxsize=1)
def _load_sentence_transformer_model():
    from sentence_transformers import SentenceTransformer

    with _quiet_optional_nlp_output():
        return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def _load_bert_score_module():
    import bert_score

    return bert_score


def extract_column_info_from_codebook(codebook_path):
    """
    Extract column names and their annotation types from the codebook file.

    Args:
        codebook_path: Path to a ``codebook.json`` file.

    Returns:
        Dictionary mapping ``<section_name>_<annotation_name>`` column names to
        annotation metadata such as type, options, and Likert bounds.
    """
    with open(codebook_path, 'r') as file:
        codebook = json.load(file)

    lookup = get_annotation_lookup(codebook)
    column_info = {}

    for section_key, section, annotation_key, annotation in get_annotation_entries(codebook):
        column_name = get_annotation_column_name(section, annotation)
        annotation_type = annotation.get('type', 'dropdown')

        properties = {
            'type': annotation_type,
            'section_key': section_key,
            'annotation_key': annotation_key,
        }

        if annotation_type == 'dropdown':
            properties['options'] = annotation.get('options', [])
        elif annotation_type == 'likert':
            properties['min_value'] = annotation.get('min_value', 0)
            properties['max_value'] = annotation.get('max_value', 5)
        elif annotation_type == 'span':
            label_options = annotation.get('label_options') or []
            properties['label_options'] = list(label_options)

        condition = get_annotation_condition(annotation)
        if condition:
            source_entry = lookup.get((condition['section_key'], condition['annotation_key']))
            if source_entry:
                source_section, source_annotation = source_entry
                properties['condition'] = {
                    'source_column': get_annotation_column_name(source_section, source_annotation),
                    'source_type': source_annotation.get('type', 'dropdown'),
                    'value': normalize_annotation_response_value(source_annotation, condition.get('value')),
                }

        column_info[column_name] = properties
    
    logger.debug("Extracted column info from codebook: %s", column_info)
    return column_info


def _is_row_applicable_for_column(merged_row, column, column_info, side="gt", visited=None):
    """Return whether a conditional annotation is applicable for one merged row."""
    info = column_info.get(column, {})
    condition = info.get("condition")
    if not condition:
        return True

    source_column = condition.get("source_column")
    if not source_column:
        return True

    visited = visited or set()
    if column in visited:
        return True

    if source_column in column_info and not _is_row_applicable_for_column(
        merged_row,
        source_column,
        column_info,
        side=side,
        visited=visited | {column},
    ):
        return False

    source_value = merged_row.get(f"{source_column}_{side}")
    source_annotation = {"type": condition.get("source_type", "dropdown")}
    actual_value = normalize_annotation_response_value(source_annotation, source_value)
    expected_value = normalize_annotation_response_value(source_annotation, condition.get("value"))

    if actual_value is None:
        return False
    if condition.get("source_type") == "textbox" and actual_value == "":
        return False

    return actual_value == expected_value


def _get_applicable_row_mask(merged_df, column, column_info, side="gt"):
    """Return a boolean mask for rows where an annotation is applicable."""
    if "condition" not in column_info.get(column, {}):
        return pd.Series(True, index=merged_df.index)

    return merged_df.apply(
        lambda row: _is_row_applicable_for_column(row, column, column_info, side=side),
        axis=1,
    )

def load_data(ground_truth_path, llm_output_path, columns_to_compare):
    """Load and align ground-truth and model-output CSV files for evaluation.

    Args:
        ground_truth_path: Path to the human-labeled reference CSV.
        llm_output_path: Path to the model-generated annotation CSV.
        columns_to_compare: List of annotation column names to evaluate.

    Returns:
        Combined DataFrame containing suffixed ``_gt`` and ``_llm`` columns.
    """
    ground_truth_df = pd.read_csv(ground_truth_path)
    llm_output_df = pd.read_csv(llm_output_path)
    
    logger.debug("Ground truth dataframe shape: %s", ground_truth_df.shape)
    logger.debug("LLM output dataframe shape: %s", llm_output_df.shape)
    
    # Verify the row counts match
    if len(ground_truth_df) != len(llm_output_df):
        logger.warning("Dataframes have different numbers of rows!")
        logger.warning("Ground truth rows: %s, LLM output rows: %s", len(ground_truth_df), len(llm_output_df))
        logger.warning("Proceeding with merge based on row order, but results may be incorrect.")
    else:
        logger.info("Both dataframes have %s rows. Proceeding with row-based merge.", len(ground_truth_df))
    
    logger.debug("Before merge:")
    for column in columns_to_compare:
        logger.debug("Column: %s", column)
        if column in ground_truth_df.columns and column in llm_output_df.columns:
            gt_values = ground_truth_df[column].fillna('').astype(str)
            llm_values = llm_output_df[column].fillna('').astype(str)
            gt_unique = sorted([x for x in gt_values.unique() if x != ''])
            llm_unique = sorted([x for x in llm_values.unique() if x != ''])
            logger.debug("Ground truth unique values: %s", gt_unique)
            logger.debug("LLM output unique values: %s", llm_unique)
        else:
            if column not in ground_truth_df.columns:
                logger.warning("Column '%s' not found in ground truth dataframe", column)
            if column not in llm_output_df.columns:
                logger.warning("Column '%s' not found in LLM output dataframe", column)

    # Create new dataframe by adding suffix to column names
    gt_columns = {col: f"{col}_gt" for col in ground_truth_df.columns if col in columns_to_compare}
    llm_columns = {col: f"{col}_llm" for col in llm_output_df.columns if col in columns_to_compare}
    
    # Create copies with renamed columns for the columns we want to compare
    gt_renamed = ground_truth_df.rename(columns=gt_columns)
    llm_renamed = llm_output_df.rename(columns=llm_columns)
    
    # Concatenate horizontally (cbind/hstack) based on row position
    merged_df = pd.concat([gt_renamed, llm_renamed], axis=1)
    
    logger.debug("Merged dataframe has %s rows and %s columns", len(merged_df), merged_df.shape[1])
    
    logger.debug("After merge:")
    for column in columns_to_compare:
        column_gt = f'{column}_gt'
        column_llm = f'{column}_llm'

        if column_gt in merged_df.columns and column_llm in merged_df.columns:
            logger.debug("Column: %s", column)
            gt_values = merged_df[column_gt].fillna('').astype(str)
            llm_values = merged_df[column_llm].fillna('').astype(str)
            gt_unique = sorted([x for x in gt_values.unique() if x != ''])
            llm_unique = sorted([x for x in llm_values.unique() if x != ''])
            logger.debug("Ground truth unique values: %s", gt_unique)
            logger.debug("LLM output unique values: %s", llm_unique)

            logger.debug("Value counts in ground truth:\n%s", merged_df[column_gt].fillna('').astype(str).value_counts().to_string())
            logger.debug("Value counts in LLM output:\n%s", merged_df[column_llm].fillna('').astype(str).value_counts().to_string())

            gt_col = merged_df[column_gt].fillna('').astype(str)
            llm_col = merged_df[column_llm].fillna('').astype(str)
            mismatches = merged_df[gt_col != llm_col]
            if len(mismatches) > 0:
                logger.debug("Found %s mismatches for %s. First few examples:", len(mismatches), column)
                for idx, row in mismatches.head().iterrows():
                    logger.debug("Row index: %s", idx)
                    logger.debug("Ground truth: '%s'", row[column_gt])
                    logger.debug("LLM output: '%s'", row[column_llm])
                    logger.debug("---")
        else:
            if column_gt not in merged_df.columns:
                logger.warning("Column '%s' not found in merged dataframe", column_gt)
            if column_llm not in merged_df.columns:
                logger.warning("Column '%s' not found in merged dataframe", column_llm)

    # Convert all columns to string type before returning
    for column in columns_to_compare:
        column_gt = f'{column}_gt'
        column_llm = f'{column}_llm'
        if column_gt in merged_df.columns:
            merged_df[column_gt] = merged_df[column_gt].fillna('').astype(str)
        if column_llm in merged_df.columns:
            merged_df[column_llm] = merged_df[column_llm].fillna('').astype(str)

    return merged_df

def fill_specific_missing_values(df, columns, fill_value):
    """Fill missing values in specific DataFrame columns.

    Args:
        df: Pandas DataFrame to mutate in place.
        columns: Iterable of column names to fill if present.
        fill_value: Replacement value for missing entries.
    """
    for column in columns:
        if column in df.columns:
            df[column] = df[column].fillna(fill_value)
        else:
            logger.warning("Column '%s' not found in DataFrame.", column)

def fill_missing_values(df, columns_to_compare, fill_value="unknown"):
    """Fill missing values in a list of DataFrame columns if they exist.

    Args:
        df: Pandas DataFrame to mutate in place.
        columns_to_compare: Iterable of column names to fill if present.
        fill_value: Replacement value for missing entries.
    """
    for column in columns_to_compare:
        if column in df.columns:
            df[column] = df[column].fillna(fill_value)
        else:
            logger.warning("Column '%s' not found in DataFrame.", column)

def calculate_percentage_agreement(y_true, y_pred):
    """Compute exact-match agreement between two aligned series."""
    return sum(y_true == y_pred) / len(y_true)

def read_emissions_data(emissions_file):
    """Read emissions and energy metadata from a CodeCarbon CSV file.

    Args:
        emissions_file: Path to ``emissions.csv`` produced during annotation.

    Returns:
        Tuple of ``(emissions, energy_consumed, cpu_model, gpu_model)``.
    """
    try:
        emissions_df = pd.read_csv(emissions_file)
        if len(emissions_df) > 0:
            latest_row = emissions_df.iloc[-1]
            return (latest_row['emissions'], latest_row['energy_consumed'], 
                   latest_row['cpu_model'], latest_row['gpu_model'])
        return None, None, None, None
    except Exception as e:
        logger.warning("Error reading emissions file: %s", e)
        return None, None, None, None
    
def read_timing_data(timing_file):
    """Read inference timing statistics from a JSON sidecar file.

    Args:
        timing_file: Path to ``timing_data.json``.

    Returns:
        Tuple of ``(total_inference_time, avg_inference_time, n_queries)``, where
        ``n_queries`` is the number of model inference calls.
    """
    try:
        with open(timing_file, 'r') as file:
            timing_data = json.load(file)
            return (timing_data.get('total_inference_time', None),
                   timing_data.get('avg_inference_time', None),
                   timing_data.get('inference_count', None))
    except Exception as e:
        logger.warning("Error reading timing file: %s", e)
        return None, None, None
    
def read_char_counts(char_counts_file):
    """Read prompt and response character counts from a JSON sidecar file.

    Args:
        char_counts_file: Path to ``char_counts.json``.

    Returns:
        Tuple of ``(input_chars, output_chars)``.
    """
    try:
        with open(char_counts_file, 'r') as file:
            char_counts = json.load(file)
            return (char_counts.get('input_chars', None), 
                   char_counts.get('output_chars', None))
    except Exception as e:
        logger.warning("Error reading character counts file: %s", e)
        return None, None

def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculate the quadratic weighted kappa.
    
    Args:
        y_true: Iterable of gold ordinal labels.
        y_pred: Iterable of predicted ordinal labels.

    Returns:
        Quadratic weighted kappa as a float, or ``nan`` when it cannot be computed.
    """
    # Convert to numeric if not already
    y_true = pd.to_numeric(y_true, errors='coerce')
    y_pred = pd.to_numeric(y_pred, errors='coerce')
    
    # Drop missing values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return float('nan')
    
    # Get the unique classes
    labels = sorted(set(np.concatenate([y_true, y_pred])))
    n_labels = len(labels)
    
    if n_labels <= 1:
        return float('nan')  # Can't calculate with only one class
    
    # Create a label mapping
    label_map = {l: i for i, l in enumerate(labels)}
    
    # Map the values to integers
    y_true_mapped = np.array([label_map[v] for v in y_true])
    y_pred_mapped = np.array([label_map[v] for v in y_pred])
    
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=range(n_labels))
    
    # Calculate the weights matrix (quadratic weighting)
    weights = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        for j in range(n_labels):
            weights[i, j] = ((i - j) ** 2) / ((n_labels - 1) ** 2)
    
    # Calculate expected matrix
    row_sum = np.sum(cm, axis=1)
    col_sum = np.sum(cm, axis=0)
    expected = np.outer(row_sum, col_sum) / np.sum(cm)
    
    # Calculate weighted kappa
    k = 1.0 - np.sum(weights * cm) / np.sum(weights * expected)
    
    return k

def evaluate_performance(
    merged_df,
    columns_to_compare,
    column_info,
    process_textbox=False,
    process_span=False,
    texts=None,
):
    """Compute evaluation metrics for each requested annotation column.

    Args:
        merged_df: DataFrame returned by :func:`load_data`.
        columns_to_compare: List of annotation columns to evaluate.
        column_info: Metadata mapping produced by :func:`extract_column_info_from_codebook`.
        process_textbox: Whether textbox similarity metrics should be calculated.
        process_span: Whether span metrics should be calculated.
        texts: Optional iterable of source texts (one per row in ``merged_df``)
            used to evaluate span annotations. Required when ``process_span``
            is True and any span columns are present.

    Returns:
        Tuple of metric dictionaries and report text keyed by annotation column name.
    """
    # Initialize all metric dictionaries with default NaN values for all columns
    accuracy_scores = {col: float('nan') for col in columns_to_compare}
    precision_scores = {col: float('nan') for col in columns_to_compare}
    recall_scores = {col: float('nan') for col in columns_to_compare}
    f1_scores = {col: float('nan') for col in columns_to_compare}
    cohen_kappa_scores = {col: float('nan') for col in columns_to_compare}
    krippendorff_alpha_scores = {col: float('nan') for col in columns_to_compare}
    percentage_agreement_scores = {col: float('nan') for col in columns_to_compare}
    reports = {col: "Not processed" for col in columns_to_compare}
    
    # Ordinal metrics for likert scales
    spearman_corr_scores = {col: float('nan') for col in columns_to_compare}
    quadratic_kappa_scores = {col: float('nan') for col in columns_to_compare}
    
    # Textbox metrics
    norm_levenshtein_scores = {col: float('nan') for col in columns_to_compare}
    bleu_scores = {col: float('nan') for col in columns_to_compare}
    rouge1_f_scores = {col: float('nan') for col in columns_to_compare}
    rouge2_f_scores = {col: float('nan') for col in columns_to_compare}
    rougeL_f_scores = {col: float('nan') for col in columns_to_compare}
    cosine_scores = {col: float('nan') for col in columns_to_compare}
    bertscore_p_scores = {col: float('nan') for col in columns_to_compare}
    bertscore_r_scores = {col: float('nan') for col in columns_to_compare}
    bertscore_f1_scores = {col: float('nan') for col in columns_to_compare}

    # Span metrics
    token_f1_scores = {col: float('nan') for col in columns_to_compare}
    exact_match_f1_scores = {col: float('nan') for col in columns_to_compare}
    char_iou_scores = {col: float('nan') for col in columns_to_compare}

    for column in columns_to_compare:
        column_gt = f'{column}_gt'
        column_llm = f'{column}_llm'
        
        # Check if both columns exist in the dataframe
        if column_gt not in merged_df.columns or column_llm not in merged_df.columns:
            logger.warning("Skipping column '%s' as one or both corresponding columns are missing in the merged dataframe", column)
            reports[column] = "Column missing in dataframe."
            continue
        
        # Get annotation type
        annotation_type = column_info.get(column, {}).get('type', 'dropdown')
        logger.debug("Processing column: %s (Type: %s)", column, annotation_type)
        
        # Skip textbox annotations if process_textbox is False
        if annotation_type == 'textbox' and not process_textbox:
            logger.info("Skipping textbox column '%s' as process_textbox is False", column)
            reports[column] = "Textbox processing skipped."
            continue

        # Skip span annotations if process_span is False
        if annotation_type == 'span' and not process_span:
            logger.info("Skipping span column '%s' as process_span is False", column)
            reports[column] = "Span processing skipped."
            continue

        # Span annotations get a dedicated metric pipeline and don't use the
        # downstream classification-style code path.
        if annotation_type == 'span':
            applicable_mask = _get_applicable_row_mask(merged_df, column, column_info, side="gt")
            gt_values = merged_df.loc[applicable_mask, column_gt]
            pred_values = merged_df.loc[applicable_mask, column_llm]
            if texts is None:
                logger.warning(
                    "Cannot compute span metrics for column '%s' without source texts.",
                    column,
                )
                reports[column] = "Span metrics skipped: source texts unavailable."
                continue
            texts_series = pd.Series(list(texts), index=merged_df.index)
            row_texts = texts_series.loc[applicable_mask]
            label_options = column_info.get(column, {}).get('label_options') or None
            try:
                span_metrics = compute_span_metrics(
                    row_texts.fillna('').astype(str).tolist(),
                    gt_values.tolist(),
                    pred_values.tolist(),
                    label_options=label_options,
                )
                token_f1_scores[column] = span_metrics['token_f1']
                exact_match_f1_scores[column] = span_metrics['exact_match_f1']
                char_iou_scores[column] = span_metrics['char_iou']
                reports[column] = (
                    "Span field, specialized metrics calculated "
                    f"(token_f1={span_metrics['token_f1']:.3f}, "
                    f"exact_match_f1={span_metrics['exact_match_f1']:.3f}, "
                    f"char_iou={span_metrics['char_iou']:.3f})."
                )
            except Exception as exc:
                logger.warning("Error calculating span metrics for column '%s': %s", column, exc)
                reports[column] = f"Error in span metrics: {exc}"
            continue

        applicable_mask = _get_applicable_row_mask(merged_df, column, column_info, side="gt")
        y_true = merged_df.loc[applicable_mask, column_gt]
        y_pred = merged_df.loc[applicable_mask, column_llm]

        # Handle values based on annotation type
        if annotation_type == 'checkbox':
            logger.debug("Handling checkbox column - converting to binary values")
            y_true = y_true.fillna('0')
            y_pred = y_pred.fillna('0')
            # Ensure values are '0' or '1'
            y_true = y_true.map(lambda x: '1' if str(x).lower() in ['1', 'true', 'yes', 'checked'] else '0')
            y_pred = y_pred.map(lambda x: '1' if str(x).lower() in ['1', 'true', 'yes', 'checked'] else '0')
        elif annotation_type == 'likert':
            logger.debug("Handling likert column - calculating both ordinal and classification metrics")
            # Get min and max values from codebook
            min_value = column_info[column].get('min_value', 0)
            max_value = column_info[column].get('max_value', 5)
            
            # Convert to numeric values for ordinal metrics
            y_true_numeric = pd.to_numeric(y_true, errors='coerce')
            y_pred_numeric = pd.to_numeric(y_pred, errors='coerce')
            
            # Calculate ordinal metrics
            valid_indices = ~(y_true_numeric.isna() | y_pred_numeric.isna())
            
            if valid_indices.any():
                # Calculate Spearman's correlation
                corr, _ = spearmanr(y_true_numeric[valid_indices], y_pred_numeric[valid_indices])
                spearman_corr_scores[column] = corr
                
                # Calculate quadratic weighted kappa
                quadratic_kappa_scores[column] = quadratic_weighted_kappa(
                    y_true_numeric[valid_indices], y_pred_numeric[valid_indices])
            
            # For classification metrics, convert to integers and then to string
            # This ensures '1.0' and '1' are treated as the same value
            # First cast to float, then to int, then to string to standardize
            y_true = y_true_numeric.fillna(-9999).astype(int).astype(str)
            y_pred = y_pred_numeric.fillna(-9999).astype(int).astype(str)
            
            # Replace placeholder for NaN
            y_true = y_true.replace('-9999', '')
            y_pred = y_pred.replace('-9999', '')
        elif annotation_type == 'dropdown':
                valid_indices = y_true.notna() | y_pred.notna()
                y_true = y_true[valid_indices]
                y_pred = y_pred[valid_indices]
        elif annotation_type == 'textbox':
            logger.debug("Handling textbox column - calculating text similarity metrics")
            # Process textbox with specialized metrics
            try:
                textbox_metrics = evaluate_textbox_performance(y_true, y_pred)
                missing_dependencies = sorted(set(textbox_metrics.pop('_missing_dependencies', [])))
                
                # Store results in corresponding dictionaries
                norm_levenshtein_scores[column] = textbox_metrics['norm_levenshtein']
                bleu_scores[column] = textbox_metrics['bleu']
                rouge1_f_scores[column] = textbox_metrics['rouge1_f']
                rouge2_f_scores[column] = textbox_metrics['rouge2_f']
                rougeL_f_scores[column] = textbox_metrics['rougeL_f']
                cosine_scores[column] = textbox_metrics['cosine_similarity']
                bertscore_p_scores[column] = textbox_metrics['bertscore_precision']
                bertscore_r_scores[column] = textbox_metrics['bertscore_recall']
                bertscore_f1_scores[column] = textbox_metrics['bertscore_f1']
                
                # For textbox, we only keep percentage agreement for the traditional metrics
                y_true = y_true.fillna('')
                y_pred = y_pred.fillna('')
                if missing_dependencies:
                    reports[column] = (
                        "Textbox field, available metrics calculated. "
                        f"Missing packages for some metrics: {', '.join(missing_dependencies)}. "
                        f"{TEXTBOX_DEPENDENCY_GUIDANCE}"
                    )
                else:
                    reports[column] = "Textbox field, specialized metrics calculated."
            except Exception as e:
                logger.warning("Error calculating textbox metrics for column '%s': %s", column, e)
                reports[column] = f"Error in textbox metrics: {str(e)}"

        logger.debug("Initial data shape - y_true: %s, y_pred: %s", y_true.shape, y_pred.shape)

        true_unique = sorted([x for x in y_true.unique() if pd.notna(x)])
        pred_unique = sorted([x for x in y_pred.unique() if pd.notna(x)])
        logger.debug("Initial unique values in y_true: %s", true_unique)
        logger.debug("Initial unique values in y_pred: %s", pred_unique)

        logger.debug("Detailed value counts:")
        logger.debug("Ground truth:\n%s", y_true.value_counts(dropna=False).to_string())
        logger.debug("Predictions:\n%s", y_pred.value_counts(dropna=False).to_string())

        if len(y_true) == 0:
            logger.warning("No valid entries for column '%s'", column)
            reports[column] = "No valid data for metrics calculation."
            continue

        try:
            # Skip traditional classification metrics for textbox fields
            if annotation_type == 'textbox':
                # Just calculate percentage agreement
                y_true_clean = y_true.fillna('missing')
                y_pred_clean = y_pred.fillna('missing')
                percentage_agreement_scores[column] = calculate_percentage_agreement(y_true_clean, y_pred_clean)
                continue
            
            # For metrics calculation, we need to handle NaN values
            y_true_clean = y_true.fillna('missing')
            y_pred_clean = y_pred.fillna('missing')
            
            # Calculate agreement metrics
            percentage_agreement_scores[column] = calculate_percentage_agreement(y_true_clean, y_pred_clean)
            
            # Get unique valid labels (excluding NaN)
            true_labels = set(x for x in y_true.unique() if pd.notna(x))
            pred_labels = set(x for x in y_pred.unique() if pd.notna(x))
            all_labels = sorted(true_labels | pred_labels)
            logger.debug("All unique labels: %s", all_labels)
            
            # Calculate metrics
            accuracy_scores[column] = accuracy_score(y_true_clean, y_pred_clean)
            precision_scores[column] = precision_score(y_true_clean, y_pred_clean, 
                                                    average='macro', 
                                                    zero_division=0,
                                                    labels=all_labels)
            recall_scores[column] = recall_score(y_true_clean, y_pred_clean, 
                                               average='macro', 
                                               zero_division=0,
                                               labels=all_labels)
            f1_scores[column] = f1_score(y_true_clean, y_pred_clean, 
                                       average='macro', 
                                       zero_division=0,
                                       labels=all_labels)
            cohen_kappa_scores[column] = cohen_kappa_score(y_true_clean, y_pred_clean)

            # For Krippendorff's alpha
            label_to_int = {label: i for i, label in enumerate(['missing'] + all_labels)}
            y_true_encoded = np.array([label_to_int[value] for value in y_true_clean.tolist()])
            y_pred_encoded = np.array([label_to_int[value] for value in y_pred_clean.tolist()])
            data = np.array([y_true_encoded, y_pred_encoded])
            krippendorff_alpha_scores[column] = krippendorff.alpha(reliability_data=data)

            # Generate classification report
            reports[column] = classification_report(y_true_clean, y_pred_clean, 
                                                 labels=all_labels,
                                                 zero_division=0)
        except Exception as e:
            logger.warning("Error processing column '%s': %s: %s", column, type(e).__name__, e)
            logger.debug("Current state - y_true sample:\n%s", y_true.head().to_string())
            logger.debug("Current state - y_pred sample:\n%s", y_pred.head().to_string())
            reports[column] = f"Error: {str(e)}"

    return (accuracy_scores, precision_scores, recall_scores, f1_scores,
            cohen_kappa_scores, krippendorff_alpha_scores, percentage_agreement_scores,
            spearman_corr_scores, quadratic_kappa_scores,
            norm_levenshtein_scores, bleu_scores,
            rouge1_f_scores, rouge2_f_scores, rougeL_f_scores,
            cosine_scores,
            bertscore_p_scores, bertscore_r_scores, bertscore_f1_scores,
            token_f1_scores, exact_match_f1_scores, char_iou_scores,
            reports)

def evaluate_textbox_performance(y_true, y_pred):
    """
    Calculate similarity metrics for textbox annotations.

    Args:
        y_true: Pandas Series of reference textbox annotations.
        y_pred: Pandas Series of predicted textbox annotations.

    Returns:
        Dictionary of textbox similarity metrics plus an internal
        ``_missing_dependencies`` list when optional packages are unavailable.
    """
    missing_dependencies = []

    try:
        import Levenshtein
    except ImportError:
        Levenshtein = None
        missing_dependencies.append('python-Levenshtein')

    try:
        import nltk
        from nltk.translate.bleu_score import sentence_bleu
    except ImportError:
        nltk = None
        sentence_bleu = None
        missing_dependencies.append('nltk')

    try:
        from rouge_score import rouge_scorer
    except ImportError:
        rouge_scorer = None
        missing_dependencies.append('rouge-score')

    if importlib.util.find_spec("sentence_transformers") is None:
        missing_dependencies.append("sentence-transformers")

    if importlib.util.find_spec("bert_score") is None:
        missing_dependencies.append("bert-score")

    try:
        import torch
    except ImportError:
        torch = None
        missing_dependencies.append('torch')

    # Normalize text - remove trailing whitespace and lowercase
    y_true_norm = y_true.fillna('').str.lower().str.strip()
    y_pred_norm = y_pred.fillna('').str.lower().str.strip()
    
    logger.info("Starting textbox evaluation with %s entries", len(y_true))
    logger.debug("Non-empty ground truth: %s", sum(y_true_norm != ''))
    logger.debug("Non-empty predictions: %s", sum(y_pred_norm != ''))
    
    # Initialize containers for individual scores
    norm_levenshtein_scores = []
    cosine_scores = []
    
    # Lists to collect text pairs for batch metrics
    valid_refs = []
    valid_cands = []
    
    # Process each text pair
    for i in range(len(y_true_norm)):
        true_text = y_true_norm.iloc[i]
        pred_text = y_pred_norm.iloc[i]
        
        # Add to valid pairs if both texts have content
        if true_text != '' and pred_text != '':
            valid_refs.append(true_text)
            valid_cands.append(pred_text)
        
        # 1. Normalized Levenshtein Distance
        if Levenshtein is not None and (true_text != '' or pred_text != ''):
            lev_dist = Levenshtein.distance(true_text, pred_text)
            max_len = max(len(true_text), len(pred_text))
            if max_len > 0:
                norm_lev_sim = 1 - (lev_dist / max_len)
                norm_levenshtein_scores.append(norm_lev_sim)
    
    logger.debug("Found %s valid pairs for batch metrics", len(valid_refs))
    
    # Skip batch metrics if no valid pairs
    if not valid_refs:
        return {
            'norm_levenshtein': np.mean(norm_levenshtein_scores) if norm_levenshtein_scores else float('nan'),
            'bleu': float('nan'),
            'rouge1_f': float('nan'),
            'rouge2_f': float('nan'),
            'rougeL_f': float('nan'),
            'cosine_similarity': float('nan'),
            'bertscore_precision': float('nan'),
            'bertscore_recall': float('nan'),
            'bertscore_f1': float('nan'),
            '_missing_dependencies': missing_dependencies
        }
    
    # 2. BLEU Score calculation
    bleu_score = float('nan')
    if nltk is not None and sentence_bleu is not None:
        try:
            # Calculate BLEU score manually with NLTK
            bleu_scores = []
            for ref, cand in zip(valid_refs, valid_cands):
                reference = [ref.split()]
                candidate = cand.split()
                if reference[0] and candidate:  # Ensure non-empty
                    try:
                        # Use smoothing for short segments
                        smoothing_function = nltk.translate.bleu_score.SmoothingFunction().method1
                        score = sentence_bleu(reference, candidate,
                                             weights=(0.25, 0.25, 0.25, 0.25),
                                             smoothing_function=smoothing_function)
                        bleu_scores.append(score)
                    except Exception as e:
                        logger.debug("Individual BLEU error: %s", e)

            bleu_score = np.mean(bleu_scores) if bleu_scores else float('nan')
            logger.debug("Calculated %s BLEU scores, mean: %s", len(bleu_scores), bleu_score)
        except Exception as e:
            logger.warning("BLEU calculation error: %s", e)
    else:
        logger.info("Skipping BLEU calculation because nltk is not installed.")
    
    # 3. ROUGE Score calculation
    rouge1_f = rouge2_f = rougeL_f = float('nan')
    if rouge_scorer is not None:
        try:
            # Calculate ROUGE directly
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []

            for ref, cand in zip(valid_refs, valid_cands):
                try:
                    score = scorer.score(ref, cand)
                    rouge1_scores.append(score['rouge1'].fmeasure)
                    rouge2_scores.append(score['rouge2'].fmeasure)
                    rougeL_scores.append(score['rougeL'].fmeasure)
                except Exception as e:
                    logger.debug("Individual ROUGE error: %s", e)

            rouge1_f = np.mean(rouge1_scores) if rouge1_scores else float('nan')
            rouge2_f = np.mean(rouge2_scores) if rouge2_scores else float('nan')
            rougeL_f = np.mean(rougeL_scores) if rougeL_scores else float('nan')
            logger.debug("Calculated %s ROUGE scores", len(rouge1_scores))
        except Exception as e:
            logger.warning("ROUGE calculation error: %s", e)
    else:
        logger.info("Skipping ROUGE calculation because rouge-score is not installed.")
    
    # 4. Cosine Similarity with Sentence Embeddings
    if "sentence-transformers" not in missing_dependencies and torch is not None:
        try:
            model = _load_sentence_transformer_model()

            true_embeddings = model.encode(valid_refs, convert_to_tensor=True)
            pred_embeddings = model.encode(valid_cands, convert_to_tensor=True)

            for i in range(len(valid_refs)):
                cos_sim = torch.nn.functional.cosine_similarity(
                    true_embeddings[i].unsqueeze(0),
                    pred_embeddings[i].unsqueeze(0)
                ).item()
                cosine_scores.append(cos_sim)
            logger.debug("Calculated %s cosine similarity scores", len(cosine_scores))
        except Exception as e:
            logger.warning("Embedding error: %s", e)
    else:
        logger.info("Skipping cosine similarity because sentence-transformers and/or torch are not installed.")
    
    # 5. BERTScore calculation
    bertscore_p = bertscore_r = bertscore_f1 = float('nan')
    if "bert-score" not in missing_dependencies and torch is not None:
        try:
            bert_score = _load_bert_score_module()

            # Calculate BERTScore directly
            with _quiet_optional_nlp_output():
                P, R, F1 = bert_score.score(
                    valid_cands,
                    valid_refs,
                    lang="en",
                    model_type="roberta-large",
                    rescale_with_baseline=True,
                    verbose=False
                )

            # Convert tensor outputs to Python floats
            bertscore_p = torch.mean(P).item() if len(P) > 0 else float('nan')
            bertscore_r = torch.mean(R).item() if len(R) > 0 else float('nan')
            bertscore_f1 = torch.mean(F1).item() if len(F1) > 0 else float('nan')
            logger.debug("Calculated BERTScore metrics")
        except Exception as e:
            logger.warning("BERTScore calculation error: %s", e)
    else:
        logger.info("Skipping BERTScore because bert-score and/or torch are not installed.")
    
    # Gather all results
    results = {
        'norm_levenshtein': np.mean(norm_levenshtein_scores) if norm_levenshtein_scores else float('nan'),
        'bleu': bleu_score,
        'rouge1_f': rouge1_f,
        'rouge2_f': rouge2_f,
        'rougeL_f': rougeL_f,
        'cosine_similarity': np.mean(cosine_scores) if cosine_scores else float('nan'),
        'bertscore_precision': bertscore_p,
        'bertscore_recall': bertscore_r,
        'bertscore_f1': bertscore_f1,
        '_missing_dependencies': missing_dependencies
    }
    
    logger.info("Textbox metrics calculated successfully")
    for metric, value in results.items():
        if metric != '_missing_dependencies':
            logger.debug("  %s: %s", metric, value)
    if missing_dependencies:
        logger.warning(
            "Textbox metrics missing dependencies: %s. %s",
            ", ".join(sorted(set(missing_dependencies))),
            TEXTBOX_DEPENDENCY_GUIDANCE,
        )
    
    return results

_CATEGORICAL_METRICS = ("accuracy", "precision", "recall", "f1", "cohen_kappa", "krippendorff_alpha")
_LIKERT_METRICS = ("spearman_corr", "quadratic_kappa")
_TEXTBOX_METRICS = (
    "norm_levenshtein", "bleu", "rouge1_f", "rouge2_f", "rougeL_f",
    "cosine_similarity", "bertscore_precision", "bertscore_recall", "bertscore_f1",
)
_SPAN_METRICS = ("token_f1", "exact_match_f1", "char_iou")

RUN_COLUMNS = [
    "run_id", "timestamp", "label", "model_id", "quantization_type", "temperature", "top_p",
    "prompt_type", "use_examples", "process_textbox", "process_span",
    "codebook_path", "experiment_directory", "n_queries",
    "total_inference_time_s", "avg_inference_time_per_query_s",
    "total_input_chars", "avg_input_chars_per_query",
    "total_output_chars", "avg_output_chars_per_query",
    "total_energy_kwh", "avg_energy_kwh_per_query",
    "total_emissions_kg", "avg_emissions_kg_per_query",
    "cpu_model", "gpu_model",
]
METRIC_COLUMNS = ["run_id", "column", "annotation_type", "metric", "value"]


def _metric_names_for_type(annotation_type):
    """Return the ordered metric names recorded for one annotation type."""
    metrics = []
    if annotation_type != "span":
        metrics.append("percentage_agreement")
    if annotation_type in ("dropdown", "checkbox", "likert"):
        metrics.extend(_CATEGORICAL_METRICS)
    if annotation_type == "likert":
        metrics.extend(_LIKERT_METRICS)
    if annotation_type == "textbox":
        metrics.extend(_TEXTBOX_METRICS)
    if annotation_type == "span":
        metrics.extend(_SPAN_METRICS)
    return metrics


def _per_query(total, n_queries):
    """Return ``total / n_queries`` (per-query average), or None if not computable."""
    if total is None or not n_queries:
        return None
    try:
        return total / n_queries
    except (TypeError, ZeroDivisionError):
        return None


def _append_rows(path, rows, columns):
    """Append rows to a tidy CSV with a fixed schema, writing a header if new."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    pd.DataFrame(rows, columns=columns).to_csv(path, mode="a", header=write_header, index=False)


def write_metrics(metrics_csv, runs_csv, run_id, label, model_id, quantization_type, temperature, top_p,
                  codebook_path, columns_to_compare, accuracy_scores, precision_scores, recall_scores,
                  f1_scores, cohen_kappa_scores, krippendorff_alpha_scores,
                  percentage_agreement_scores, spearman_corr_scores, quadratic_kappa_scores,
                  norm_levenshtein_scores, bleu_scores,
                  rouge1_f_scores, rouge2_f_scores, rougeL_f_scores,
                  cosine_scores,
                  bertscore_p_scores, bertscore_r_scores, bertscore_f1_scores,
                  token_f1_scores, exact_match_f1_scores, char_iou_scores,
                  column_info,
                  prompt_type=None, use_examples=None, process_textbox=None, process_span=None,
                  emissions=None, energy_consumed=None, cpu_model=None, gpu_model=None,
                  total_inference_time=None, avg_inference_time=None,
                  input_chars=None, output_chars=None, n_queries=None,
                  timestamp=None, experiment_directory=None):
    """Record one evaluation run to the tidy two-table metrics log.

    Writes two CSVs with stable schemas (so appends never widen the file):

    * ``runs_csv``: one row per run, keyed by ``run_id``, holding the run
      configuration plus efficiency metrics. Every numeric efficiency metric is
      stored as both a total and a per-query average (per model inference call).
    * ``metrics_csv``: long/tidy table, one row per ``(run_id, column, metric)``
      with the metric ``value``. Only metrics applicable to each annotation
      type are emitted.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- runs table: config + efficiency (totals and per-query averages) ---
    run_row = {
        "run_id": run_id,
        "timestamp": timestamp,
        "label": label,
        "model_id": model_id,
        "quantization_type": quantization_type,
        "temperature": temperature,
        "top_p": top_p,
        "prompt_type": prompt_type,
        "use_examples": use_examples,
        "process_textbox": process_textbox,
        "process_span": process_span,
        "codebook_path": codebook_path,
        "experiment_directory": experiment_directory,
        "n_queries": n_queries,
        "total_inference_time_s": total_inference_time,
        "avg_inference_time_per_query_s": (
            avg_inference_time if avg_inference_time is not None
            else _per_query(total_inference_time, n_queries)
        ),
        "total_input_chars": input_chars,
        "avg_input_chars_per_query": _per_query(input_chars, n_queries),
        "total_output_chars": output_chars,
        "avg_output_chars_per_query": _per_query(output_chars, n_queries),
        "total_energy_kwh": energy_consumed,
        "avg_energy_kwh_per_query": _per_query(energy_consumed, n_queries),
        "total_emissions_kg": emissions,
        "avg_emissions_kg_per_query": _per_query(emissions, n_queries),
        "cpu_model": cpu_model,
        "gpu_model": gpu_model,
    }
    _append_rows(runs_csv, [run_row], RUN_COLUMNS)

    # --- metrics table: one tidy row per (run, column, metric) ---
    score_dicts = {
        "percentage_agreement": percentage_agreement_scores,
        "accuracy": accuracy_scores,
        "precision": precision_scores,
        "recall": recall_scores,
        "f1": f1_scores,
        "cohen_kappa": cohen_kappa_scores,
        "krippendorff_alpha": krippendorff_alpha_scores,
        "spearman_corr": spearman_corr_scores,
        "quadratic_kappa": quadratic_kappa_scores,
        "norm_levenshtein": norm_levenshtein_scores,
        "bleu": bleu_scores,
        "rouge1_f": rouge1_f_scores,
        "rouge2_f": rouge2_f_scores,
        "rougeL_f": rougeL_f_scores,
        "cosine_similarity": cosine_scores,
        "bertscore_precision": bertscore_p_scores,
        "bertscore_recall": bertscore_r_scores,
        "bertscore_f1": bertscore_f1_scores,
        "token_f1": token_f1_scores,
        "exact_match_f1": exact_match_f1_scores,
        "char_iou": char_iou_scores,
    }
    metric_rows = []
    for col in columns_to_compare:
        annotation_type = column_info.get(col, {}).get("type", "dropdown")
        for metric in _metric_names_for_type(annotation_type):
            metric_rows.append({
                "run_id": run_id,
                "column": col,
                "annotation_type": annotation_type,
                "metric": metric,
                "value": score_dicts[metric].get(col),
            })
    _append_rows(metrics_csv, metric_rows, METRIC_COLUMNS)


def write_classification_reports(output_report_file, columns_to_compare, reports):
    """Write human-readable per-column classification reports to a text file.

    Args:
        output_report_file: Destination text file path.
        columns_to_compare: Ordered list of annotation columns to include.
        reports: Mapping of annotation column name to report text.
    """
    with open(output_report_file, mode='w') as file:
        for column in columns_to_compare:
            file.write(f"Classification Report for '{column}':\n")
            file.write(reports[column])
            file.write("\n" + ("-" * 50) + "\n")


def _build_metrics_by_column(
    columns_to_compare,
    column_info,
    accuracy_scores,
    precision_scores,
    recall_scores,
    f1_scores,
    cohen_kappa_scores,
    krippendorff_alpha_scores,
    percentage_agreement_scores,
    spearman_corr_scores,
    quadratic_kappa_scores,
    norm_levenshtein_scores,
    bleu_scores,
    rouge1_f_scores,
    rouge2_f_scores,
    rougeL_f_scores,
    cosine_scores,
    bertscore_p_scores,
    bertscore_r_scores,
    bertscore_f1_scores,
    token_f1_scores,
    exact_match_f1_scores,
    char_iou_scores,
):
    """Assemble a compact nested metrics dictionary keyed by column name.

    Args:
        columns_to_compare: Ordered list of annotation columns to summarize.
        column_info: Annotation metadata mapping from the codebook.
        accuracy_scores: Per-column accuracy values.
        precision_scores: Per-column precision values.
        recall_scores: Per-column recall values.
        f1_scores: Per-column F1 values.
        cohen_kappa_scores: Per-column Cohen's kappa values.
        krippendorff_alpha_scores: Per-column Krippendorff's alpha values.
        percentage_agreement_scores: Per-column exact agreement values.
        spearman_corr_scores: Per-column Spearman correlations.
        quadratic_kappa_scores: Per-column quadratic weighted kappa values.
        norm_levenshtein_scores: Per-column normalized Levenshtein scores.
        bleu_scores: Per-column BLEU scores.
        rouge1_f_scores: Per-column ROUGE-1 F values.
        rouge2_f_scores: Per-column ROUGE-2 F values.
        rougeL_f_scores: Per-column ROUGE-L F values.
        cosine_scores: Per-column cosine similarity values.
        bertscore_p_scores: Per-column BERTScore precision values.
        bertscore_r_scores: Per-column BERTScore recall values.
        bertscore_f1_scores: Per-column BERTScore F1 values.

    Returns:
        Nested metrics dictionary keyed by annotation column name.
    """
    metrics_by_column = {}
    for column in columns_to_compare:
        annotation_type = column_info.get(column, {}).get('type', 'dropdown')
        column_metrics = {
            "annotation_type": annotation_type,
        }
        if annotation_type != 'span':
            column_metrics["percentage_agreement"] = percentage_agreement_scores[column]

        if annotation_type in ['dropdown', 'checkbox', 'likert']:
            column_metrics.update({
                "accuracy": accuracy_scores[column],
                "precision": precision_scores[column],
                "recall": recall_scores[column],
                "f1": f1_scores[column],
                "cohen_kappa": cohen_kappa_scores[column],
                "krippendorff_alpha": krippendorff_alpha_scores[column],
            })

        if annotation_type == 'likert':
            column_metrics.update({
                "spearman_corr": spearman_corr_scores[column],
                "quadratic_kappa": quadratic_kappa_scores[column],
            })

        if annotation_type == 'textbox':
            column_metrics.update({
                "norm_levenshtein": norm_levenshtein_scores[column],
                "bleu": bleu_scores[column],
                "rouge1_f": rouge1_f_scores[column],
                "rouge2_f": rouge2_f_scores[column],
                "rougeL_f": rougeL_f_scores[column],
                "cosine_similarity": cosine_scores[column],
                "bertscore_precision": bertscore_p_scores[column],
                "bertscore_recall": bertscore_r_scores[column],
                "bertscore_f1": bertscore_f1_scores[column],
            })

        if annotation_type == 'span':
            column_metrics.update({
                "token_f1": token_f1_scores[column],
                "exact_match_f1": exact_match_f1_scores[column],
                "char_iou": char_iou_scores[column],
            })

        metrics_by_column[column] = column_metrics

    return metrics_by_column


def _format_metric_value(value) -> str:
    """Format a scalar metric for compact human-readable summaries."""
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
    except TypeError:
        pass

    if isinstance(value, (int, float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def _format_count_value(value) -> str:
    """Format integer-like counters for human-readable summaries."""
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):
            return "n/a"
    except TypeError:
        pass
    return f"{int(value):,}"


def format_metrics_summary(
    metrics_by_column: dict[str, dict],
    *,
    total_inference_time=None,
    avg_inference_time=None,
    input_chars=None,
    output_chars=None,
    energy_consumed=None,
    emissions=None,
    cpu_model=None,
    gpu_model=None,
    n_queries=None,
) -> str:
    """Render a compact plain-text summary of performance and efficiency metrics."""
    if not metrics_by_column:
        return "No metrics were produced."

    lines = ["Run Summary", "", "Performance"]
    for column, metrics in metrics_by_column.items():
        annotation_type = metrics.get("annotation_type", "unknown")
        if annotation_type == "textbox":
            metric_order = [
                ("agreement", "percentage_agreement"),
                ("levenshtein", "norm_levenshtein"),
                ("bleu", "bleu"),
                ("rougeL", "rougeL_f"),
                ("cosine", "cosine_similarity"),
                ("bertscore_f1", "bertscore_f1"),
            ]
        elif annotation_type == "likert":
            metric_order = [
                ("agreement", "percentage_agreement"),
                ("accuracy", "accuracy"),
                ("f1", "f1"),
                ("spearman", "spearman_corr"),
                ("quadratic_kappa", "quadratic_kappa"),
            ]
        elif annotation_type == "span":
            metric_order = [
                ("token_f1", "token_f1"),
                ("exact_match_f1", "exact_match_f1"),
                ("char_iou", "char_iou"),
            ]
        else:
            metric_order = [
                ("agreement", "percentage_agreement"),
                ("accuracy", "accuracy"),
                ("f1", "f1"),
                ("kappa", "cohen_kappa"),
                ("alpha", "krippendorff_alpha"),
            ]

        rendered = ", ".join(
            f"{label}={_format_metric_value(metrics.get(key))}"
            for label, key in metric_order
            if key in metrics
        )
        lines.append(f"- {column} [{annotation_type}]: {rendered}")

    lines.extend([
        "",
        "Efficiency",
        f"- Queries (model calls): {_format_count_value(n_queries)}",
        f"- Total inference time (s): {_format_metric_value(total_inference_time)}",
        f"- Average inference time per query (s): {_format_metric_value(avg_inference_time)}",
        f"- Prompt characters: {_format_count_value(input_chars)} "
        f"(avg/query: {_format_metric_value(_per_query(input_chars, n_queries))})",
        f"- Output characters: {_format_count_value(output_chars)} "
        f"(avg/query: {_format_metric_value(_per_query(output_chars, n_queries))})",
        f"- Energy consumed (kWh): {_format_metric_value(energy_consumed)} "
        f"(avg/query: {_format_metric_value(_per_query(energy_consumed, n_queries))})",
        f"- Emissions (kg CO2eq): {_format_metric_value(emissions)} "
        f"(avg/query: {_format_metric_value(_per_query(emissions, n_queries))})",
    ])

    if cpu_model:
        lines.append(f"- CPU: {cpu_model}")
    if gpu_model and str(gpu_model).lower() != "none":
        lines.append(f"- GPU: {gpu_model}")

    return "\n".join(lines)

def run_metrics(
    *,
    ground_truth_csv,
    llm_output_csv,
    label,
    output_csv,
    model_id,
    codebook_path,
    report_file,
    columns=None,
    quantization_type=None,
    temperature=None,
    top_p=None,
    prompt_type=None,
    use_examples=None,
    process_textbox=False,
    process_span=False,
    emissions_file=None,
    experiment_directory=None,
    timestamp=None,
    timing_file=None,
    char_counts_file=None,
):
    """Evaluate one model-output CSV against ground truth and persist the results.

    Args:
        ground_truth_csv: Path to the human-labeled reference CSV.
        llm_output_csv: Path to the model-generated annotation CSV.
        label: Experiment label written to the metrics CSV, usually the task name.
        output_csv: Path to the aggregate metrics CSV to create or update.
        model_id: Stable identifier for the model and prompt configuration.
        codebook_path: Path to the codebook used for the run.
        report_file: Path to the per-column report text file.
        columns: Optional subset of annotation column names to evaluate.
        quantization_type: Optional quantization metadata string.
        temperature: Optional temperature metadata value.
        top_p: Optional top-p metadata value.
        prompt_type: Optional prompt wrapper name stored as metadata.
        use_examples: Optional boolean-like metadata flag.
        process_textbox: Whether textbox metrics should be computed.
        emissions_file: Optional path to ``emissions.csv``.
        experiment_directory: Optional path to the per-run output directory.
        timestamp: Optional timestamp string.
        timing_file: Optional path to ``timing_data.json``.
        char_counts_file: Optional path to ``char_counts.json``.

    Returns:
        :class:`codebook_lab.types.MetricsRunResult` for the completed evaluation.
    """
    process_textbox = str(process_textbox).lower() in ('true', 'yes', '1', 't', 'y')
    process_span = str(process_span).lower() in ('true', 'yes', '1', 't', 'y')

    output_csv = Path(output_csv)
    report_file = Path(report_file)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    emissions = energy_consumed = cpu_model = gpu_model = None
    if emissions_file:
        emissions, energy_consumed, cpu_model, gpu_model = read_emissions_data(emissions_file)

    total_inference_time = avg_inference_time = n_queries = None
    if timing_file:
        total_inference_time, avg_inference_time, n_queries = read_timing_data(timing_file)

    input_chars = output_chars = None
    if char_counts_file:
        input_chars, output_chars = read_char_counts(char_counts_file)

    column_info = extract_column_info_from_codebook(codebook_path)
    columns_to_compare = list(columns) if columns else list(column_info.keys())
    if not columns_to_compare:
        raise ValueError("No columns could be extracted from the codebook and none were provided.")

    merged_df = load_data(ground_truth_csv, llm_output_csv, columns_to_compare)
    all_columns = [f'{col}_gt' for col in columns_to_compare] + [f'{col}_llm' for col in columns_to_compare]
    fill_missing_values(merged_df, all_columns, fill_value='')

    # When span metrics are enabled, source texts are required so we can align
    # per-row offsets against the original document.
    texts = None
    has_span_columns = any(
        column_info.get(col, {}).get('type') == 'span' for col in columns_to_compare
    )
    if process_span and has_span_columns:
        try:
            with open(codebook_path, 'r') as fh:
                codebook = json.load(fh)
            text_column = codebook.get('text_column')
            if text_column:
                gt_df = pd.read_csv(ground_truth_csv)
                if text_column in gt_df.columns:
                    texts = gt_df[text_column].fillna('').astype(str).tolist()
                else:
                    logger.warning(
                        "text_column '%s' from codebook is not present in %s; "
                        "span metrics will be skipped.",
                        text_column,
                        ground_truth_csv,
                    )
            else:
                logger.warning(
                    "Codebook %s does not declare a text_column; span metrics will be skipped.",
                    codebook_path,
                )
        except Exception as exc:
            logger.warning("Could not load source texts for span metrics: %s", exc)

    results = evaluate_performance(
        merged_df,
        columns_to_compare,
        column_info,
        process_textbox=process_textbox,
        process_span=process_span,
        texts=texts,
    )
    (
        accuracy_scores, precision_scores, recall_scores, f1_scores,
        cohen_kappa_scores, krippendorff_alpha_scores, percentage_agreement_scores,
        spearman_corr_scores, quadratic_kappa_scores,
        norm_levenshtein_scores, bleu_scores,
        rouge1_f_scores, rouge2_f_scores, rougeL_f_scores,
        cosine_scores,
        bertscore_p_scores, bertscore_r_scores, bertscore_f1_scores,
        token_f1_scores, exact_match_f1_scores, char_iou_scores,
        reports
    ) = results

    run_id = uuid.uuid4().hex[:12]
    output_csv = Path(output_csv)
    runs_csv = output_csv.with_name(f"{output_csv.stem}_runs.csv")
    write_metrics(
        str(output_csv), str(runs_csv), run_id, label, model_id, quantization_type,
        temperature, top_p, codebook_path, columns_to_compare,
        accuracy_scores, precision_scores, recall_scores, f1_scores,
        cohen_kappa_scores, krippendorff_alpha_scores, percentage_agreement_scores,
        spearman_corr_scores, quadratic_kappa_scores,
        norm_levenshtein_scores, bleu_scores,
        rouge1_f_scores, rouge2_f_scores, rougeL_f_scores,
        cosine_scores,
        bertscore_p_scores, bertscore_r_scores, bertscore_f1_scores,
        token_f1_scores, exact_match_f1_scores, char_iou_scores,
        column_info,
        prompt_type=prompt_type, use_examples=use_examples,
        process_textbox=str(process_textbox).lower(),
        process_span=str(process_span).lower(),
        emissions=emissions, energy_consumed=energy_consumed,
        cpu_model=cpu_model, gpu_model=gpu_model,
        total_inference_time=total_inference_time, avg_inference_time=avg_inference_time,
        input_chars=input_chars, output_chars=output_chars, n_queries=n_queries,
        timestamp=timestamp, experiment_directory=experiment_directory,
    )

    write_classification_reports(str(report_file), columns_to_compare, reports)
    metrics_by_column = _build_metrics_by_column(
        columns_to_compare,
        column_info,
        accuracy_scores,
        precision_scores,
        recall_scores,
        f1_scores,
        cohen_kappa_scores,
        krippendorff_alpha_scores,
        percentage_agreement_scores,
        spearman_corr_scores,
        quadratic_kappa_scores,
        norm_levenshtein_scores,
        bleu_scores,
        rouge1_f_scores,
        rouge2_f_scores,
        rougeL_f_scores,
        cosine_scores,
        bertscore_p_scores,
        bertscore_r_scores,
        bertscore_f1_scores,
        token_f1_scores,
        exact_match_f1_scores,
        char_iou_scores,
    )

    logger.info("Results successfully written to %s and %s", output_csv, report_file)
    return MetricsRunResult(
        output_csv=output_csv,
        report_file=report_file,
        columns_to_compare=columns_to_compare,
        metrics_by_column=metrics_by_column,
        reports=reports,
        total_inference_time=total_inference_time,
        avg_inference_time=avg_inference_time,
        input_chars=input_chars,
        output_chars=output_chars,
        energy_consumed=energy_consumed,
        emissions=emissions,
        cpu_model=cpu_model,
        gpu_model=gpu_model,
        summary_text=format_metrics_summary(
            metrics_by_column,
            total_inference_time=total_inference_time,
            avg_inference_time=avg_inference_time,
            input_chars=input_chars,
            output_chars=output_chars,
            energy_consumed=energy_consumed,
            emissions=emissions,
            cpu_model=cpu_model,
            gpu_model=gpu_model,
            n_queries=n_queries,
        ),
        run_id=run_id,
        runs_csv=runs_csv,
        n_queries=n_queries,
    )
