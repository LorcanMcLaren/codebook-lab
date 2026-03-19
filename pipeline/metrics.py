import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, cohen_kappa_score
import numpy as np
import argparse
import os
import json
from datetime import datetime
import krippendorff
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix

def extract_column_info_from_codebook(codebook_path):
    """
    Extract column names and their annotation types from the codebook file.
    Returns a dictionary mapping column names to their annotation types and properties.
    """
    with open(codebook_path, 'r') as file:
        codebook = json.load(file)
    
    column_info = {}
    
    for key, section in codebook.items():
        if key.startswith('section_'):
            section_name = section['section_name']
            annotations = section['annotations']
            
            for annotation_key, annotation in annotations.items():
                name = annotation['name']
                
                column_name = f"{section_name}_{name}"
                
                # Extract annotation type and relevant properties
                annotation_type = annotation.get('type', 'dropdown')  # Default to dropdown for backward compatibility
                
                properties = {
                    'type': annotation_type
                }
                
                # Add type-specific properties
                if annotation_type == 'dropdown':
                    properties['options'] = annotation.get('options', [])
                elif annotation_type == 'likert':
                    properties['min_value'] = annotation.get('min_value', 0)
                    properties['max_value'] = annotation.get('max_value', 5)
                
                column_info[column_name] = properties
    
    print(f"Extracted column info from codebook: {column_info}")
    return column_info

def load_data(ground_truth_path, llm_output_path, columns_to_compare):
    ground_truth_df = pd.read_csv(ground_truth_path)
    llm_output_df = pd.read_csv(llm_output_path)
    
    # Print dataframe shapes
    print(f"\nGround truth dataframe shape: {ground_truth_df.shape}")
    print(f"LLM output dataframe shape: {llm_output_df.shape}")
    
    # Verify the row counts match
    if len(ground_truth_df) != len(llm_output_df):
        print(f"\nWARNING: Dataframes have different numbers of rows!")
        print(f"Ground truth rows: {len(ground_truth_df)}, LLM output rows: {len(llm_output_df)}")
        print("Proceeding with merge based on row order, but results may be incorrect.")
    else:
        print(f"\nBoth dataframes have {len(ground_truth_df)} rows. Proceeding with row-based merge.")
    
    # Print unique values before merge
    print("\nDEBUG - Before merge:")
    for column in columns_to_compare:
        print(f"\nColumn: {column}")
        # Check if column exists in both dataframes
        if column in ground_truth_df.columns and column in llm_output_df.columns:
            # Convert to string and handle NaN values
            gt_values = ground_truth_df[column].fillna('').astype(str)
            llm_values = llm_output_df[column].fillna('').astype(str)
            # Remove empty strings from the unique values
            gt_unique = sorted([x for x in gt_values.unique() if x != ''])
            llm_unique = sorted([x for x in llm_values.unique() if x != ''])
            print("Ground truth unique values:", gt_unique)
            print("LLM output unique values:", llm_unique)
        else:
            if column not in ground_truth_df.columns:
                print(f"Warning: Column '{column}' not found in ground truth dataframe")
            if column not in llm_output_df.columns:
                print(f"Warning: Column '{column}' not found in LLM output dataframe")

    # Create new dataframe by adding suffix to column names
    gt_columns = {col: f"{col}_gt" for col in ground_truth_df.columns if col in columns_to_compare}
    llm_columns = {col: f"{col}_llm" for col in llm_output_df.columns if col in columns_to_compare}
    
    # Create copies with renamed columns for the columns we want to compare
    gt_renamed = ground_truth_df.rename(columns=gt_columns)
    llm_renamed = llm_output_df.rename(columns=llm_columns)
    
    # Concatenate horizontally (cbind/hstack) based on row position
    merged_df = pd.concat([gt_renamed, llm_renamed], axis=1)
    
    print(f"\nMerged dataframe has {len(merged_df)} rows and {merged_df.shape[1]} columns")
    
    # Print unique values after merge
    print("\nDEBUG - After merge:")
    for column in columns_to_compare:
        column_gt = f'{column}_gt'
        column_llm = f'{column}_llm'
        
        # Check if columns exist in merged dataframe
        if column_gt in merged_df.columns and column_llm in merged_df.columns:
            print(f"\nColumn: {column}")
            # Convert to string and handle NaN values
            gt_values = merged_df[column_gt].fillna('').astype(str)
            llm_values = merged_df[column_llm].fillna('').astype(str)
            # Remove empty strings from the unique values
            gt_unique = sorted([x for x in gt_values.unique() if x != ''])
            llm_unique = sorted([x for x in llm_values.unique() if x != ''])
            print(f"Ground truth unique values: {gt_unique}")
            print(f"LLM output unique values: {llm_unique}")
            
            # Print counts of each value
            print("\nValue counts in ground truth:")
            print(merged_df[column_gt].fillna('').astype(str).value_counts().to_string())
            print("\nValue counts in LLM output:")
            print(merged_df[column_llm].fillna('').astype(str).value_counts().to_string())
            
            # Check for any rows where the values don't match
            # Convert both to string for comparison
            gt_col = merged_df[column_gt].fillna('').astype(str)
            llm_col = merged_df[column_llm].fillna('').astype(str)
            mismatches = merged_df[gt_col != llm_col]
            if len(mismatches) > 0:
                print(f"\nFound {len(mismatches)} mismatches for {column}. First few examples:")
                for idx, row in mismatches.head().iterrows():
                    print(f"Row index: {idx}")
                    print(f"Ground truth: '{row[column_gt]}'")
                    print(f"LLM output: '{row[column_llm]}'")
                    print("---")
        else:
            if column_gt not in merged_df.columns:
                print(f"Warning: Column '{column_gt}' not found in merged dataframe")
            if column_llm not in merged_df.columns:
                print(f"Warning: Column '{column_llm}' not found in merged dataframe")

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
    for column in columns:
        if column in df.columns:
            df[column] = df[column].fillna(fill_value)
        else:
            print(f"Warning: Column '{column}' not found in DataFrame.")

def fill_missing_values(df, columns_to_compare, fill_value="unknown"):
    for column in columns_to_compare:
        if column in df.columns:
            df[column] = df[column].fillna(fill_value)
        else:
            print(f"Warning: Column '{column}' not found in DataFrame.")

def calculate_percentage_agreement(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)

def read_emissions_data(emissions_file):
    """Read emissions and energy data from the emissions CSV file."""
    try:
        emissions_df = pd.read_csv(emissions_file)
        if len(emissions_df) > 0:
            latest_row = emissions_df.iloc[-1]
            return (latest_row['emissions'], latest_row['energy_consumed'], 
                   latest_row['cpu_model'], latest_row['gpu_model'])
        return None, None, None, None
    except Exception as e:
        print(f"Error reading emissions file: {str(e)}")
        return None, None, None, None
    
def read_timing_data(timing_file):
    """Read timing data from the timing JSON file."""
    try:
        with open(timing_file, 'r') as file:
            timing_data = json.load(file)
            return (timing_data.get('total_inference_time', None), 
                   timing_data.get('avg_inference_time', None))
    except Exception as e:
        print(f"Error reading timing file: {str(e)}")
        return None, None
    
def read_char_counts(char_counts_file):
    """Read character count data from the JSON file."""
    try:
        with open(char_counts_file, 'r') as file:
            char_counts = json.load(file)
            return (char_counts.get('input_chars', None), 
                   char_counts.get('output_chars', None))
    except Exception as e:
        print(f"Error reading character counts file: {str(e)}")
        return None, None

def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculate the quadratic weighted kappa.
    
    This metric gives partial credit for "near misses" based on how far apart the ratings are,
    which is ideal for ordinal scales like Likert.
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

def evaluate_performance(merged_df, columns_to_compare, column_info, process_textbox=False):
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

    for column in columns_to_compare:
        column_gt = f'{column}_gt'
        column_llm = f'{column}_llm'
        
        # Check if both columns exist in the dataframe
        if column_gt not in merged_df.columns or column_llm not in merged_df.columns:
            print(f"Skipping column '{column}' as one or both corresponding columns are missing in the merged dataframe")
            reports[column] = "Column missing in dataframe."
            continue
        
        # Get annotation type
        annotation_type = column_info.get(column, {}).get('type', 'dropdown')
        print(f"\nDEBUG - Processing column: {column} (Type: {annotation_type})")
        
        # Skip textbox annotations if process_textbox is False
        if annotation_type == 'textbox' and not process_textbox:
            print(f"Skipping textbox column '{column}' as process_textbox is False")
            reports[column] = "Textbox processing skipped."
            continue
            
        y_true = merged_df[column_gt]
        y_pred = merged_df[column_llm]

        # Handle values based on annotation type
        if annotation_type == 'checkbox':
            print("Handling checkbox column - converting to binary values")
            y_true = y_true.fillna('0')
            y_pred = y_pred.fillna('0')
            # Ensure values are '0' or '1'
            y_true = y_true.map(lambda x: '1' if str(x).lower() in ['1', 'true', 'yes', 'checked'] else '0')
            y_pred = y_pred.map(lambda x: '1' if str(x).lower() in ['1', 'true', 'yes', 'checked'] else '0')
        elif annotation_type == 'likert':
            print("Handling likert column - calculating both ordinal and classification metrics")
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
            print("Handling textbox column - calculating text similarity metrics")
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
                        "Install requirements.txt to enable the full textbox metric suite."
                    )
                else:
                    reports[column] = "Textbox field, specialized metrics calculated."
            except Exception as e:
                print(f"Error calculating textbox metrics for column '{column}':")
                print(f"Exception: {str(e)}")
                reports[column] = f"Error in textbox metrics: {str(e)}"

        # Print initial data stats
        print(f"Initial data shape - y_true: {y_true.shape}, y_pred: {y_pred.shape}")
        
        # Handle NaN values when getting unique values
        true_unique = sorted([x for x in y_true.unique() if pd.notna(x)])
        pred_unique = sorted([x for x in y_pred.unique() if pd.notna(x)])
        print("Initial unique values in y_true:", true_unique)
        print("Initial unique values in y_pred:", pred_unique)
        
        # Print detailed counts
        print("\nDetailed value counts:")
        print("Ground truth:")
        print(y_true.value_counts(dropna=False).to_string())
        print("\nPredictions:")
        print(y_pred.value_counts(dropna=False).to_string())

        if len(y_true) == 0:
            print(f"Warning: No valid entries for column '{column}'")
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
            print("\nAll unique labels:", all_labels)
            
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
            y_true_encoded = np.array([label_to_int[y_true_clean[i]] for i in range(len(y_true_clean))])
            y_pred_encoded = np.array([label_to_int[y_pred_clean[i]] for i in range(len(y_pred_clean))])
            data = np.array([y_true_encoded, y_pred_encoded])
            krippendorff_alpha_scores[column] = krippendorff.alpha(reliability_data=data)

            # Generate classification report
            reports[column] = classification_report(y_true_clean, y_pred_clean, 
                                                 labels=all_labels,
                                                 zero_division=0)
        except Exception as e:
            print(f"Error processing column '{column}':")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {str(e)}")
            print("Current state:")
            print(f"y_true sample: {y_true.head().to_string()}")
            print(f"y_pred sample: {y_pred.head().to_string()}")
            reports[column] = f"Error: {str(e)}"

    return (accuracy_scores, precision_scores, recall_scores, f1_scores, 
            cohen_kappa_scores, krippendorff_alpha_scores, percentage_agreement_scores, 
            spearman_corr_scores, quadratic_kappa_scores, 
            norm_levenshtein_scores, bleu_scores, 
            rouge1_f_scores, rouge2_f_scores, rougeL_f_scores,
            cosine_scores, 
            bertscore_p_scores, bertscore_r_scores, bertscore_f1_scores,
            reports)

def evaluate_textbox_performance(y_true, y_pred):
    """
    Calculate similarity metrics for textbox annotations.
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

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        SentenceTransformer = None
        missing_dependencies.append('sentence-transformers')

    try:
        import bert_score
    except ImportError:
        bert_score = None
        missing_dependencies.append('bert-score')

    try:
        import torch
    except ImportError:
        torch = None
        missing_dependencies.append('torch')

    # Normalize text - remove trailing whitespace and lowercase
    y_true_norm = y_true.fillna('').str.lower().str.strip()
    y_pred_norm = y_pred.fillna('').str.lower().str.strip()
    
    print(f"Starting textbox evaluation with {len(y_true)} entries")
    print(f"Non-empty ground truth: {sum(y_true_norm != '')}")
    print(f"Non-empty predictions: {sum(y_pred_norm != '')}")
    
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
    
    print(f"Found {len(valid_refs)} valid pairs for batch metrics")
    
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
                        print(f"Individual BLEU error: {str(e)}")

            bleu_score = np.mean(bleu_scores) if bleu_scores else float('nan')
            print(f"Calculated {len(bleu_scores)} BLEU scores, mean: {bleu_score}")
        except Exception as e:
            print(f"BLEU calculation error: {str(e)}")
    else:
        print("Skipping BLEU calculation because nltk is not installed.")
    
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
                    print(f"Individual ROUGE error: {str(e)}")

            rouge1_f = np.mean(rouge1_scores) if rouge1_scores else float('nan')
            rouge2_f = np.mean(rouge2_scores) if rouge2_scores else float('nan')
            rougeL_f = np.mean(rougeL_scores) if rougeL_scores else float('nan')
            print(f"Calculated {len(rouge1_scores)} ROUGE scores")
        except Exception as e:
            print(f"ROUGE calculation error: {str(e)}")
    else:
        print("Skipping ROUGE calculation because rouge-score is not installed.")
    
    # 4. Cosine Similarity with Sentence Embeddings
    if SentenceTransformer is not None and torch is not None:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')

            true_embeddings = model.encode(valid_refs, convert_to_tensor=True)
            pred_embeddings = model.encode(valid_cands, convert_to_tensor=True)

            for i in range(len(valid_refs)):
                cos_sim = torch.nn.functional.cosine_similarity(
                    true_embeddings[i].unsqueeze(0),
                    pred_embeddings[i].unsqueeze(0)
                ).item()
                cosine_scores.append(cos_sim)
            print(f"Calculated {len(cosine_scores)} cosine similarity scores")
        except Exception as e:
            print(f"Embedding error: {str(e)}")
    else:
        print("Skipping cosine similarity because sentence-transformers and/or torch are not installed.")
    
    # 5. BERTScore calculation
    bertscore_p = bertscore_r = bertscore_f1 = float('nan')
    if bert_score is not None and torch is not None:
        try:
            # Calculate BERTScore directly
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
            print(f"Calculated BERTScore metrics")
        except Exception as e:
            print(f"BERTScore calculation error: {str(e)}")
    else:
        print("Skipping BERTScore because bert-score and/or torch are not installed.")
    
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
    
    print("Textbox metrics calculated successfully:")
    for metric, value in results.items():
        if metric != '_missing_dependencies':
            print(f"  {metric}: {value}")
    if missing_dependencies:
        print(f"Textbox metrics missing dependencies: {', '.join(sorted(set(missing_dependencies)))}")
    
    return results

def append_metrics_to_csv(output_csv, label, model_id, quantization_type, temperature, top_p, codebook_path,
                         columns_to_compare, accuracy_scores, precision_scores, recall_scores,
                         f1_scores, cohen_kappa_scores, krippendorff_alpha_scores,
                         percentage_agreement_scores, spearman_corr_scores, quadratic_kappa_scores, 
                         norm_levenshtein_scores, bleu_scores, 
                         rouge1_f_scores, rouge2_f_scores, rougeL_f_scores,
                         cosine_scores, 
                         bertscore_p_scores, bertscore_r_scores, bertscore_f1_scores,
                         column_info,
                         prompt_type=None, use_examples=None, process_textbox=None,
                         emissions=None, energy_consumed=None, cpu_model=None, gpu_model=None,
                         total_inference_time=None, avg_inference_time=None,
                         input_chars=None, output_chars=None,
                         timestamp=None, experiment_directory=None):
    """
    Append metrics results to a CSV file, storing only metrics relevant to each field type.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    file_exists = os.path.isfile(output_csv)

    # Create base header with metadata columns - ADD TIMING AND CHARACTER COUNT COLUMNS
    base_header = ['Timestamp', 'Label', 'Model ID', 'Quantization Type', 'Temperature', 'Top_P',
                 'Prompt Type', 'Use Examples', 'Process Textbox', 'Codebook Path', 'Experiment Directory',
                 'CPU Model', 'GPU Model', 'Emissions (kg CO₂eq)', 'Energy Consumed (kWh)',
                 'Total Inference Time (s)', 'Avg Inference Time (s)', 
                 'Total Input Chars', 'Total Output Chars']
    
    # Dynamically create header with only relevant metrics for each column
    metric_header = []
    for col in columns_to_compare:
        annotation_type = column_info.get(col, {}).get('type', 'dropdown')
        
        # Common metric for all types
        metric_header.append(f'{col}_percentage_agreement')
        
        # Metrics for categorical fields (dropdown, checkbox)
        if annotation_type in ['dropdown', 'checkbox'] or annotation_type == 'likert':
            metric_header.extend([
                f'{col}_accuracy', 
                f'{col}_precision', 
                f'{col}_recall', 
                f'{col}_f1', 
                f'{col}_cohen_kappa', 
                f'{col}_krippendorff_alpha'
            ])
        
        # Metrics for likert fields
        if annotation_type == 'likert':
            metric_header.extend([
                f'{col}_spearman_corr', 
                f'{col}_quadratic_kappa'
            ])
        
        # Metrics for textbox fields
        if annotation_type == 'textbox':
            metric_header.extend([
                f'{col}_norm_levenshtein', 
                f'{col}_bleu', 
                f'{col}_rouge1_f', 
                f'{col}_rouge2_f', 
                f'{col}_rougeL_f', 
                f'{col}_cosine_similarity', 
                f'{col}_bertscore_precision', 
                f'{col}_bertscore_recall', 
                f'{col}_bertscore_f1'
            ])
    
    # Combine base and metric headers
    new_header = base_header + metric_header

    if file_exists:
        try:
            df = pd.read_csv(output_csv)  # Use pandas for easier data manipulation
        except pd.errors.EmptyDataError:  # Handle empty file
            df = pd.DataFrame(columns=[])

        if set(new_header) != set(df.columns):  # Compare sets of headers using pandas
            # If new columns are added, add them to the existing dataframe
            missing_cols = set(new_header) - set(df.columns)
            for col in missing_cols:
                df[col] = pd.NA
            
            # If columns are removed, keep them in the dataframe for backward compatibility
            # but we'll maintain the right order
            for col in new_header:
                if col not in df.columns:
                    df[col] = pd.NA
            
            # Reorder columns to match new_header plus any existing columns
            all_cols = new_header + [c for c in df.columns if c not in new_header]
            df = df[all_cols]
            
            df.to_csv(output_csv, index=False)  # Write to CSV, overwriting

        # Create new row with base metadata
        new_row = {
            'Timestamp': timestamp,
            'Label': label,
            'Model ID': model_id,
            'Quantization Type': quantization_type,
            'Temperature': temperature,
            'Top_P': top_p,
            'Prompt Type': prompt_type,
            'Use Examples': use_examples,
            'Process Textbox': process_textbox,
            'Codebook Path': codebook_path,
            'Experiment Directory': experiment_directory,
            'CPU Model': cpu_model,
            'GPU Model': gpu_model,
            'Emissions (kg CO₂eq)': emissions,
            'Energy Consumed (kWh)': energy_consumed,
            'Total Inference Time (s)': total_inference_time,
            'Avg Inference Time (s)': avg_inference_time,
            'Total Input Chars': input_chars,
            'Total Output Chars': output_chars,
        }
        
        # Add metrics based on column type
        for col in columns_to_compare:
            annotation_type = column_info.get(col, {}).get('type', 'dropdown')
            
            # Common metric for all types
            new_row[f'{col}_percentage_agreement'] = percentage_agreement_scores[col]
            
            # Metrics for categorical fields (dropdown, checkbox)
            if annotation_type in ['dropdown', 'checkbox'] or annotation_type == 'likert':
                new_row[f'{col}_accuracy'] = accuracy_scores[col]
                new_row[f'{col}_precision'] = precision_scores[col]
                new_row[f'{col}_recall'] = recall_scores[col]
                new_row[f'{col}_f1'] = f1_scores[col]
                new_row[f'{col}_cohen_kappa'] = cohen_kappa_scores[col]
                new_row[f'{col}_krippendorff_alpha'] = krippendorff_alpha_scores[col]
            
            # Metrics for likert fields
            if annotation_type == 'likert':
                new_row[f'{col}_spearman_corr'] = spearman_corr_scores[col]
                new_row[f'{col}_quadratic_kappa'] = quadratic_kappa_scores[col]
            
            # Metrics for textbox fields
            if annotation_type == 'textbox':
                new_row[f'{col}_norm_levenshtein'] = norm_levenshtein_scores[col]
                new_row[f'{col}_bleu'] = bleu_scores[col]
                new_row[f'{col}_rouge1_f'] = rouge1_f_scores[col]
                new_row[f'{col}_rouge2_f'] = rouge2_f_scores[col]
                new_row[f'{col}_rougeL_f'] = rougeL_f_scores[col]
                new_row[f'{col}_cosine_similarity'] = cosine_scores[col]
                new_row[f'{col}_bertscore_precision'] = bertscore_p_scores[col]
                new_row[f'{col}_bertscore_recall'] = bertscore_r_scores[col]
                new_row[f'{col}_bertscore_f1'] = bertscore_f1_scores[col]

        # Add any missing columns with NaN values
        for col in df.columns:
            if col not in new_row:
                new_row[col] = pd.NA
                
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)  # Append using pandas
        df.to_csv(output_csv, index=False)  # Write to CSV
    else:  # File doesn't exist, create it
        # Create new row with base metadata
        new_row = {
            'Timestamp': timestamp,
            'Label': label,
            'Model ID': model_id,
            'Quantization Type': quantization_type,
            'Temperature': temperature,
            'Top_P': top_p,
            'Prompt Type': prompt_type,
            'Use Examples': use_examples,
            'Process Textbox': process_textbox,
            'Codebook Path': codebook_path,
            'Experiment Directory': experiment_directory,
            'CPU Model': cpu_model,
            'GPU Model': gpu_model,
            'Emissions (kg CO₂eq)': emissions,
            'Energy Consumed (kWh)': energy_consumed,
            'Total Inference Time (s)': total_inference_time,
            'Avg Inference Time (s)': avg_inference_time,
            'Total Input Chars': input_chars,
            'Total Output Chars': output_chars,
        }
        
        # Add metrics based on column type
        for col in columns_to_compare:
            annotation_type = column_info.get(col, {}).get('type', 'dropdown')
            
            # Common metric for all types
            new_row[f'{col}_percentage_agreement'] = percentage_agreement_scores[col]
            
            # Metrics for categorical fields (dropdown, checkbox)
            if annotation_type in ['dropdown', 'checkbox'] or annotation_type == 'likert':
                new_row[f'{col}_accuracy'] = accuracy_scores[col]
                new_row[f'{col}_precision'] = precision_scores[col]
                new_row[f'{col}_recall'] = recall_scores[col]
                new_row[f'{col}_f1'] = f1_scores[col]
                new_row[f'{col}_cohen_kappa'] = cohen_kappa_scores[col]
                new_row[f'{col}_krippendorff_alpha'] = krippendorff_alpha_scores[col]
            
            # Metrics for likert fields
            if annotation_type == 'likert':
                new_row[f'{col}_spearman_corr'] = spearman_corr_scores[col]
                new_row[f'{col}_quadratic_kappa'] = quadratic_kappa_scores[col]
            
            # Metrics for textbox fields
            if annotation_type == 'textbox':
                new_row[f'{col}_norm_levenshtein'] = norm_levenshtein_scores[col]
                new_row[f'{col}_bleu'] = bleu_scores[col]
                new_row[f'{col}_rouge1_f'] = rouge1_f_scores[col]
                new_row[f'{col}_rouge2_f'] = rouge2_f_scores[col]
                new_row[f'{col}_rougeL_f'] = rougeL_f_scores[col]
                new_row[f'{col}_cosine_similarity'] = cosine_scores[col]
                new_row[f'{col}_bertscore_precision'] = bertscore_p_scores[col]
                new_row[f'{col}_bertscore_recall'] = bertscore_r_scores[col]
                new_row[f'{col}_bertscore_f1'] = bertscore_f1_scores[col]

        df = pd.DataFrame([new_row], columns=new_header)
        df.to_csv(output_csv, index=False)
 
def write_classification_reports(output_report_file, columns_to_compare, reports):
    with open(output_report_file, mode='w') as file:
        for column in columns_to_compare:
            file.write(f"Classification Report for '{column}':\n")
            file.write(reports[column])
            file.write("\n" + ("-" * 50) + "\n")

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM performance metrics.')
    parser.add_argument('ground_truth_csv', type=str, help='Path to the ground truth CSV file.')
    parser.add_argument('llm_output_csv', type=str, help='Path to the LLM output CSV file.')
    
    # Make columns argument optional by adding a --columns flag
    parser.add_argument('--columns', type=str, nargs='+', 
                        help='Optional: specific columns to compare between ground truth and LLM output. If not provided, columns will be extracted from the codebook.')
    
    parser.add_argument('--label', type=str, required=True, help='Label for the experiment.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to the CSV file where metrics will be recorded.')
    parser.add_argument('--model_id', type=str, required=True, help='Model ID used in the experiment.')
    parser.add_argument('--quantization_type', type=str, required=False, default=None, help='Quantization type used in the experiment.')
    parser.add_argument('--temperature', type=str, required=False, default=None, help='Temperature used in the experiment.')
    parser.add_argument('--top_p', type=str, required=False, default=None, help='Top-p value used in the experiment.')
    parser.add_argument('--prompt_type', type=str, required=False, default=None, help='Prompt type used in the experiment.')
    parser.add_argument('--use_examples', type=str, required=False, default=None, help='Whether examples were used in the experiment.')
    parser.add_argument('--process_textbox', type=str, required=False, default='false', help='Whether to include textbox annotations in metrics calculation.')
    parser.add_argument('--codebook_path', type=str, required=True, help='Path to the codebook file used in the experiment.')
    parser.add_argument('--report_file', type=str, required=True, help='Path to the text file where classification reports will be recorded.')
    parser.add_argument('--emissions_file', type=str, required=False, help='Path to the emissions CSV file.')
    parser.add_argument('--experiment_directory', type=str, required=False, default=None, help='Directory where experiment outputs are stored.')
    parser.add_argument('--timestamp', type=str, required=False, default=None, help='Timestamp for the experiment.')
    parser.add_argument('--timing_file', type=str, required=False, help='Path to the timing JSON file.')
    parser.add_argument('--char_counts_file', type=str, required=False, help='Path to the character counts JSON file.')

    args = parser.parse_args()

    # Process boolean flags
    process_textbox = args.process_textbox.lower() in ('true', 'yes', '1', 't', 'y')

    # Extract model name from the model ID
    model_name = os.path.basename(args.model_id)

    # Read emissions data if provided
    emissions = energy_consumed = cpu_model = gpu_model = None
    if args.emissions_file:
        emissions, energy_consumed, cpu_model, gpu_model = read_emissions_data(args.emissions_file)

    # Read timing data if provided
    total_inference_time = avg_inference_time = None
    if args.timing_file:
        total_inference_time, avg_inference_time = read_timing_data(args.timing_file)
        
    # Read character count data if provided
    input_chars = output_chars = None
    if args.char_counts_file:
        input_chars, output_chars = read_char_counts(args.char_counts_file)

    # Extract column info from codebook
    column_info = extract_column_info_from_codebook(args.codebook_path)
    
    # Determine columns to compare
    columns_to_compare = args.columns
    if not columns_to_compare:
        print("No columns specified, extracting columns from codebook...")
        columns_to_compare = list(column_info.keys())
        if not columns_to_compare:
            print("Error: No columns could be extracted from codebook.")
            return

    merged_df = load_data(args.ground_truth_csv, args.llm_output_csv, columns_to_compare)
    
    # Adjust column names for fill_missing_values
    all_columns = [f'{col}_gt' for col in columns_to_compare] + [f'{col}_llm' for col in columns_to_compare]
    fill_missing_values(merged_df, all_columns, fill_value='')

    # Evaluate performance with the updated metrics
    results = evaluate_performance(merged_df, columns_to_compare, column_info, process_textbox)
    
    # Unpack results
    (accuracy_scores, precision_scores, recall_scores, f1_scores, 
     cohen_kappa_scores, krippendorff_alpha_scores, percentage_agreement_scores, 
     spearman_corr_scores, quadratic_kappa_scores, 
     norm_levenshtein_scores, bleu_scores, 
     rouge1_f_scores, rouge2_f_scores, rougeL_f_scores,
     cosine_scores, 
     bertscore_p_scores, bertscore_r_scores, bertscore_f1_scores,
     reports) = results

    # Print metrics including emissions data
    print("\n==== METRIC RESULTS ====")
    for column in columns_to_compare:
        annotation_type = column_info.get(column, {}).get('type', 'dropdown')
        print(f"\nMetrics for '{column}' (Type: {annotation_type}):")
        
        # Print metrics based on annotation type
        if annotation_type == 'likert':
            print(f"  Spearman Correlation: {spearman_corr_scores[column]:.4f}")
            print(f"  Quadratic Weighted Kappa: {quadratic_kappa_scores[column]:.4f}")
            print(f"  Percentage Agreement: {percentage_agreement_scores[column]:.4f}")
            print(f"  Accuracy:  {accuracy_scores[column]:.4f}")
            print(f"  Precision: {precision_scores[column]:.4f}")
            print(f"  Recall:    {recall_scores[column]:.4f}")
            print(f"  F1 Score:  {f1_scores[column]:.4f}")
            print(f"  Cohen's Kappa: {cohen_kappa_scores[column]:.4f}")
            print(f"  Krippendorff's Alpha: {krippendorff_alpha_scores[column]:.4f}")
        elif annotation_type == 'textbox' and process_textbox:
            print(f"  Percentage Agreement: {percentage_agreement_scores[column]:.4f}")
            print(f"  Normalized Levenshtein: {norm_levenshtein_scores[column]:.4f}")
            print(f"  BLEU Score: {bleu_scores[column]:.4f}")
            print(f"  ROUGE-1 F1: {rouge1_f_scores[column]:.4f}")
            print(f"  ROUGE-2 F1: {rouge2_f_scores[column]:.4f}")
            print(f"  ROUGE-L F1: {rougeL_f_scores[column]:.4f}")
            print(f"  Cosine Similarity: {cosine_scores[column]:.4f}")
            print(f"  BERTScore Precision: {bertscore_p_scores[column]:.4f}")
            print(f"  BERTScore Recall: {bertscore_r_scores[column]:.4f}")
            print(f"  BERTScore F1: {bertscore_f1_scores[column]:.4f}")
        elif annotation_type == 'textbox' and not process_textbox:
            print("  Textbox processing skipped.")
        else:  # dropdown or checkbox
            print(f"  Accuracy:  {accuracy_scores[column]:.4f}")
            print(f"  Precision: {precision_scores[column]:.4f}")
            print(f"  Recall:    {recall_scores[column]:.4f}")
            print(f"  F1 Score:  {f1_scores[column]:.4f}")
            print(f"  Cohen's Kappa: {cohen_kappa_scores[column]:.4f}")
            print(f"  Krippendorff's Alpha: {krippendorff_alpha_scores[column]:.4f}")
            print(f"  Percentage Agreement: {percentage_agreement_scores[column]:.4f}")


    if emissions is not None:
        print(f"\nEmissions: {emissions:.6e} kgCO2eq")
    if energy_consumed is not None:
        print(f"Energy consumed: {energy_consumed:.6e} kWh")
    if total_inference_time is not None:
        print(f"Total inference time: {total_inference_time:.2f} seconds")
    if avg_inference_time is not None:
        print(f"Average inference time: {avg_inference_time:.2f} seconds per call")
    if input_chars is not None:
        print(f"Total input characters: {input_chars}")
    if output_chars is not None:
        print(f"Total output characters: {output_chars}")

    # Append all metrics to CSV with timing and character count data
    append_metrics_to_csv(
        args.output_csv, args.label, args.model_id, args.quantization_type, 
        args.temperature, args.top_p, args.codebook_path, columns_to_compare, 
        accuracy_scores, precision_scores, recall_scores, f1_scores, 
        cohen_kappa_scores, krippendorff_alpha_scores, percentage_agreement_scores, 
        spearman_corr_scores, quadratic_kappa_scores,
        norm_levenshtein_scores, bleu_scores, 
        rouge1_f_scores, rouge2_f_scores, rougeL_f_scores,
        cosine_scores, 
        bertscore_p_scores, bertscore_r_scores, bertscore_f1_scores,
        column_info,
        prompt_type=args.prompt_type, use_examples=args.use_examples,
        process_textbox=args.process_textbox,
        emissions=emissions, energy_consumed=energy_consumed, 
        cpu_model=cpu_model, gpu_model=gpu_model,
        total_inference_time=total_inference_time, avg_inference_time=avg_inference_time,
        input_chars=input_chars, output_chars=output_chars,
        timestamp=args.timestamp, experiment_directory=args.experiment_directory
    )

    write_classification_reports(args.report_file, columns_to_compare, reports)
    
    print(f"\nResults successfully written to {args.output_csv} and {args.report_file}")

if __name__ == "__main__":
    main()
