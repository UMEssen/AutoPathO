import os
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import delayed, Parallel
from evaluation import preprocess_df_icdo, evaluation_complete, preprocess_df_icd10, preprocess_deepseek_icdo, preprocess_deepseek_icd10, evaluation_three_characters, evaluation_complete_per_class, evaluation_three_characters_per_class

def save_results_to_csv(all_results, output_dir="results"):
    """
    Save evaluation results to CSV files.
    
    Args:
        all_results: Dictionary with structure {icd_type: {model_name: {eval_type: results}}}
        output_dir: Directory to save the CSV files
    """
    # Create a directory for results if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process each ICD type
    for icd_type, models_data in all_results.items():
        # Create DataFrames for complete and three-character evaluations
        complete_data = []
        three_char_data = []
        
        # Collect data for each model
        for model_name, model_results in models_data.items():
            # Complete code evaluation
            complete_metrics = model_results['complete']
            complete_row = {
                'Model': model_name,
                'Micro_Precision': complete_metrics['micro_precision'],
                'Micro_Precision_CI_Lower': complete_metrics['micro_precision_ci'][0],
                'Micro_Precision_CI_Upper': complete_metrics['micro_precision_ci'][1],
                'Micro_Recall': complete_metrics['micro_recall'],
                'Micro_Recall_CI_Lower': complete_metrics['micro_recall_ci'][0],
                'Micro_Recall_CI_Upper': complete_metrics['micro_recall_ci'][1],
                'Micro_F1': complete_metrics['micro_f1'],
                'Micro_F1_CI_Lower': complete_metrics['micro_f1_ci'][0],
                'Micro_F1_CI_Upper': complete_metrics['micro_f1_ci'][1],
                'Macro_Precision': complete_metrics['macro_precision'],
                'Macro_Precision_CI_Lower': complete_metrics['macro_precision_ci'][0],
                'Macro_Precision_CI_Upper': complete_metrics['macro_precision_ci'][1],
                'Macro_Recall': complete_metrics['macro_recall'],
                'Macro_Recall_CI_Lower': complete_metrics['macro_recall_ci'][0],
                'Macro_Recall_CI_Upper': complete_metrics['macro_recall_ci'][1],
                'Macro_F1': complete_metrics['macro_f1'],
                'Macro_F1_CI_Lower': complete_metrics['macro_f1_ci'][0],
                'Macro_F1_CI_Upper': complete_metrics['macro_f1_ci'][1],
                'Subset_Accuracy': complete_metrics['subset_accuracy'],
                'Subset_Accuracy_CI_Lower': complete_metrics['subset_accuracy_ci'][0],
                'Subset_Accuracy_CI_Upper': complete_metrics['subset_accuracy_ci'][1],
                'Hamming_Loss': complete_metrics['hamming_loss'],
                'Hamming_Loss_CI_Lower': complete_metrics['hamming_loss_ci'][0],
                'Hamming_Loss_CI_Upper': complete_metrics['hamming_loss_ci'][1],
                'TP': complete_metrics['tp'],
                'FP': complete_metrics['fp'],
                'FN': complete_metrics['fn'],
                'TN': complete_metrics['tn']
            }
            complete_data.append(complete_row)
            
            # Three-character evaluation
            three_char_metrics = model_results['three_char']
            three_char_row = {
                'Model': model_name,
                'Micro_Precision': three_char_metrics['micro_precision'],
                'Micro_Precision_CI_Lower': three_char_metrics['micro_precision_ci'][0],
                'Micro_Precision_CI_Upper': three_char_metrics['micro_precision_ci'][1],
                'Micro_Recall': three_char_metrics['micro_recall'],
                'Micro_Recall_CI_Lower': three_char_metrics['micro_recall_ci'][0],
                'Micro_Recall_CI_Upper': three_char_metrics['micro_recall_ci'][1],
                'Micro_F1': three_char_metrics['micro_f1'],
                'Micro_F1_CI_Lower': three_char_metrics['micro_f1_ci'][0],
                'Micro_F1_CI_Upper': three_char_metrics['micro_f1_ci'][1],
                'Macro_Precision': three_char_metrics['macro_precision'],
                'Macro_Precision_CI_Lower': three_char_metrics['macro_precision_ci'][0],
                'Macro_Precision_CI_Upper': three_char_metrics['macro_precision_ci'][1],
                'Macro_Recall': three_char_metrics['macro_recall'],
                'Macro_Recall_CI_Lower': three_char_metrics['macro_recall_ci'][0],
                'Macro_Recall_CI_Upper': three_char_metrics['macro_recall_ci'][1],
                'Macro_F1': three_char_metrics['macro_f1'],
                'Macro_F1_CI_Lower': three_char_metrics['macro_f1_ci'][0],
                'Macro_F1_CI_Upper': three_char_metrics['macro_f1_ci'][1],
                'Subset_Accuracy': three_char_metrics['subset_accuracy'],
                'Subset_Accuracy_CI_Lower': three_char_metrics['subset_accuracy_ci'][0],
                'Subset_Accuracy_CI_Upper': three_char_metrics['subset_accuracy_ci'][1],
                'Hamming_Loss': three_char_metrics['hamming_loss'],
                'Hamming_Loss_CI_Lower': three_char_metrics['hamming_loss_ci'][0],
                'Hamming_Loss_CI_Upper': three_char_metrics['hamming_loss_ci'][1],
                'TP': three_char_metrics['tp'],
                'FP': three_char_metrics['fp'],
                'FN': three_char_metrics['fn'],
                'TN': three_char_metrics['tn']
            }
            three_char_data.append(three_char_row)
        
        # Create DataFrames
        complete_df = pd.DataFrame(complete_data)
        three_char_df = pd.DataFrame(three_char_data)
        
        # Save to CSV
        complete_filename = f"{output_dir}/{icd_type}_complete_{timestamp}.csv"
        three_char_filename = f"{output_dir}/{icd_type}_three_char_{timestamp}.csv"
        
        complete_df.to_csv(complete_filename, index=False)
        three_char_df.to_csv(three_char_filename, index=False)
        
        print(f"Results for {icd_type} saved to:")
        print(f"  Complete code evaluation: {complete_filename}")
        print(f"  Three-character evaluation: {three_char_filename}")
    
    # Also save a combined file with all results
    combined_data = []
    for icd_type, models_data in all_results.items():
        for model_name, model_results in models_data.items():
            # Complete evaluation
            complete_metrics = model_results['complete']
            combined_row = {
                'ICD_Type': icd_type,
                'Model': model_name,
                'Evaluation_Type': 'Complete',
                'Micro_Precision': complete_metrics['micro_precision'],
                'Micro_Precision_CI_Lower': complete_metrics['micro_precision_ci'][0],
                'Micro_Precision_CI_Upper': complete_metrics['micro_precision_ci'][1],
                'Micro_Recall': complete_metrics['micro_recall'],
                'Micro_Recall_CI_Lower': complete_metrics['micro_recall_ci'][0],
                'Micro_Recall_CI_Upper': complete_metrics['micro_recall_ci'][1],
                'Micro_F1': complete_metrics['micro_f1'],
                'Micro_F1_CI_Lower': complete_metrics['micro_f1_ci'][0],
                'Micro_F1_CI_Upper': complete_metrics['micro_f1_ci'][1],
                'Macro_Precision': complete_metrics['macro_precision'],
                'Macro_Precision_CI_Lower': complete_metrics['macro_precision_ci'][0],
                'Macro_Precision_CI_Upper': complete_metrics['macro_precision_ci'][1],
                'Macro_Recall': complete_metrics['macro_recall'],
                'Macro_Recall_CI_Lower': complete_metrics['macro_recall_ci'][0],
                'Macro_Recall_CI_Upper': complete_metrics['macro_recall_ci'][1],
                'Macro_F1': complete_metrics['macro_f1'],
                'Macro_F1_CI_Lower': complete_metrics['macro_f1_ci'][0],
                'Macro_F1_CI_Upper': complete_metrics['macro_f1_ci'][1],
                'Subset_Accuracy': complete_metrics['subset_accuracy'],
                'Subset_Accuracy_CI_Lower': complete_metrics['subset_accuracy_ci'][0],
                'Subset_Accuracy_CI_Upper': complete_metrics['subset_accuracy_ci'][1],
                'Hamming_Loss': complete_metrics['hamming_loss'],
                'Hamming_Loss_CI_Lower': complete_metrics['hamming_loss_ci'][0],
                'Hamming_Loss_CI_Upper': complete_metrics['hamming_loss_ci'][1],
                'TP': complete_metrics['tp'],
                'FP': complete_metrics['fp'],
                'FN': complete_metrics['fn'],
                'TN': complete_metrics['tn']
            }
            combined_data.append(combined_row)
            
            # Three-character evaluation
            three_char_metrics = model_results['three_char']
            combined_row = {
                'ICD_Type': icd_type,
                'Model': model_name,
                'Evaluation_Type': 'Three_Character',
                'Micro_Precision': three_char_metrics['micro_precision'],
                'Micro_Precision_CI_Lower': three_char_metrics['micro_precision_ci'][0],
                'Micro_Precision_CI_Upper': three_char_metrics['micro_precision_ci'][1],
                'Micro_Recall': three_char_metrics['micro_recall'],
                'Micro_Recall_CI_Lower': three_char_metrics['micro_recall_ci'][0],
                'Micro_Recall_CI_Upper': three_char_metrics['micro_recall_ci'][1],
                'Micro_F1': three_char_metrics['micro_f1'],
                'Micro_F1_CI_Lower': three_char_metrics['micro_f1_ci'][0],
                'Micro_F1_CI_Upper': three_char_metrics['micro_f1_ci'][1],
                'Macro_Precision': three_char_metrics['macro_precision'],
                'Macro_Precision_CI_Lower': three_char_metrics['macro_precision_ci'][0],
                'Macro_Precision_CI_Upper': three_char_metrics['macro_precision_ci'][1],
                'Macro_Recall': three_char_metrics['macro_recall'],
                'Macro_Recall_CI_Lower': three_char_metrics['macro_recall_ci'][0],
                'Macro_Recall_CI_Upper': three_char_metrics['macro_recall_ci'][1],
                'Macro_F1': three_char_metrics['macro_f1'],
                'Macro_F1_CI_Lower': three_char_metrics['macro_f1_ci'][0],
                'Macro_F1_CI_Upper': three_char_metrics['macro_f1_ci'][1],
                'Subset_Accuracy': three_char_metrics['subset_accuracy'],
                'Subset_Accuracy_CI_Lower': three_char_metrics['subset_accuracy_ci'][0],
                'Subset_Accuracy_CI_Upper': three_char_metrics['subset_accuracy_ci'][1],
                'Hamming_Loss': three_char_metrics['hamming_loss'],
                'Hamming_Loss_CI_Lower': three_char_metrics['hamming_loss_ci'][0],
                'Hamming_Loss_CI_Upper': three_char_metrics['hamming_loss_ci'][1],
                'TP': three_char_metrics['tp'],
                'FP': three_char_metrics['fp'],
                'FN': three_char_metrics['fn'],
                'TN': three_char_metrics['tn']
            }
            combined_data.append(combined_row)
    
    # Save combined results
    combined_df = pd.DataFrame(combined_data)
    combined_filename = f"{output_dir}/all_results_{timestamp}.csv"
    combined_df.to_csv(combined_filename, index=False)
    print(f"\nCombined results saved to: {combined_filename}")


def calculate_bootstrap_sample(bootstrap_df, gt_column, pred_column, eval_type="complete"):
    """Process a single bootstrap sample and return metrics."""
    # Prepare for evaluation
    if eval_type == "complete":
        micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1, subset_accuracy, hamming_loss, tp, fp, fn, tn = evaluation_complete(bootstrap_df, gt_column, pred_column)
    else:  # three_characters
        micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1, subset_accuracy, hamming_loss, tp, fp, fn, tn = evaluation_three_characters(bootstrap_df, gt_column, pred_column)
    
    return {
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'subset_accuracy': subset_accuracy,
        'hamming_loss': hamming_loss
    }

def calculate_bootstrap_ci(df, gt_column, pred_column, n_bootstrap=1000, ci=95,
                           eval_type="complete", seed=42, n_jobs=-1):
    """
    Calculate confidence intervals for evaluation metrics using parallel bootstrap sampling.
    
    Args:
        df: DataFrame containing ground truth and predictions
        gt_column: Column name for ground truth
        pred_column: Column name for predictions
        n_bootstrap: Number of bootstrap samples (default: 1000)
        ci: Confidence interval percentage (default: 95)
        eval_type: "complete" or "three_characters" to specify evaluation type
        seed: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 uses all cores)
        
    Returns:
        Dictionary containing lower and upper bounds for each metric
    """
    np.random.seed(seed)
    n_samples = len(df)
    alpha = (100 - ci) / 2 / 100  # Convert CI to alpha for percentile calculation
    
    # Pre-generate all bootstrap indices
    bootstrap_indices = [np.random.choice(n_samples, n_samples, replace=True)
                         for _ in range(n_bootstrap)]
    
    # Use parallel processing to calculate metrics for each bootstrap sample
    results = Parallel(n_jobs=n_jobs)(
        delayed(calculate_bootstrap_sample)(
            df.iloc[indices].reset_index(drop=True), 
            gt_column, 
            pred_column,
            eval_type
        )
        for indices in bootstrap_indices
    )
    
    # Reorganize results
    bootstrap_results = {
        'micro_precision': [], 'micro_recall': [], 'micro_f1': [],
        'macro_precision': [], 'macro_recall': [], 'macro_f1': [],
        'subset_accuracy': [], 'hamming_loss': []
    }
    
    for res in results:
        for metric, value in res.items():
            bootstrap_results[metric].append(value)
    
    # Calculate confidence intervals
    confidence_intervals = {}
    for metric, values in bootstrap_results.items():
        lower = np.percentile(values, alpha * 100)
        upper = np.percentile(values, (1 - alpha) * 100)
        confidence_intervals[metric] = (lower, upper)
    
    return confidence_intervals

def evaluation_with_ci(df, gt_column, pred_column, n_bootstrap=1000, ci=95, n_jobs=-1):
    """
    Run evaluation with confidence intervals and print results.
    """
    
    # Complete ICD code evaluation
    ci_complete = calculate_bootstrap_ci(df, gt_column, pred_column, 
                                        n_bootstrap=n_bootstrap, ci=ci,
                                        eval_type="complete", n_jobs=n_jobs)
    
    
    # Three-character evaluation
    ci_three_char = calculate_bootstrap_ci(df, gt_column, pred_column,
                                          n_bootstrap=n_bootstrap, ci=ci,
                                          eval_type="three_characters", n_jobs=n_jobs)
    
    return ci_complete, ci_three_char

def filter_common_reports(dataframes, match_columns=['url', 'id', 'title', 'issued', "Befunde"], 
                          save_filtered=True, output_dir="filtered_data", icd_type=None):
    """
    Filter all dataframes to include only rows where all specified columns match across dataframes.
    Saves a single file of common reports and a single file of uncommon reports if requested.
    
    Args:
        dataframes: Dict of dataframes with model names as keys
        match_columns: List of column names to match on (will use all available from the list)
        save_filtered: Whether to save common and uncommon reports to CSV
        output_dir: Directory to save the CSV files
        icd_type: Type of ICD codes being processed (for filename)
        
    Returns:
        Dict of filtered dataframes with the same keys
    """
    
    # Create output directory if saving is enabled
    if save_filtered:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create prefix for filenames if icd_type is provided
        prefix = f"{icd_type}_" if icd_type else ""
    
    # Find all columns that exist in all dataframes
    available_columns = []
    for col in match_columns:
        if all(col in df.columns for df in dataframes.values()):
            available_columns.append(col)
    
    if not available_columns:
        raise ValueError(f"None of the specified columns {match_columns} exist in all dataframes.")
    
    print(f"Matching on columns: {available_columns}")
    
    # Create a unique identifier for each report based on the available columns
    for name, df in dataframes.items():
        # Combine all matching columns into a single identifier
        df['_match_key'] = df[available_columns].astype(str).agg('|'.join, axis=1)
    
    # Find common match keys across all dataframes
    match_key_sets = [set(df['_match_key']) for df in dataframes.values()]
    common_match_keys = set.intersection(*match_key_sets)
    
    print(f"Found {len(common_match_keys)} common reports across all dataframes")
    
    # Check for duplicates in match keys
    for name, df in dataframes.items():
        duplicate_count = df['_match_key'].duplicated().sum()
        if duplicate_count > 0:
            print(f"  Warning: {name} has {duplicate_count} duplicate match keys")
    
    # Filter each dataframe to include only common reports
    filtered_dfs = {}
    uncommon_reports = {}
    
    # Choose a reference dataframe for saving common reports (using the first one)
    reference_model = list(dataframes.keys())[0]
    reference_df = dataframes[reference_model].copy()
    
    # Filter and process each dataframe
    for name, df in dataframes.items():
        # Keep only the first occurrence of each match key to handle duplicates consistently
        filtered_df = df[df['_match_key'].isin(common_match_keys)].copy()
        filtered_df = filtered_df.drop_duplicates(subset=['_match_key'], keep='first')
        
        # Extract uncommon reports for this model
        uncommon_df = df[~df['_match_key'].isin(common_match_keys)].copy()
        uncommon_reports[name] = uncommon_df
        
        # Further ensure exact same reports by keeping only reports with match keys in the exact same order
        match_key_order = {key: i for i, key in enumerate(sorted(common_match_keys))}
        filtered_df['_match_key_idx'] = filtered_df['_match_key'].map(match_key_order)
        filtered_df = filtered_df.sort_values('_match_key_idx').reset_index(drop=True)
        
        # Remove the temporary columns
        filtered_df = filtered_df.drop(['_match_key', '_match_key_idx'], axis=1)
        
        filtered_dfs[name] = filtered_df
        print(f"  {name}: {len(filtered_df)} rows kept, {len(uncommon_df)} rows filtered out")
    
    # Save a single common reports file and uncommon reports summary
    if save_filtered:
        # Select the reference dataframe's common reports to save
        common_refs = reference_df[reference_df['_match_key'].isin(common_match_keys)].copy()
        common_refs = common_refs.drop_duplicates(subset=['_match_key'], keep='first')
        
        # Sort in the same order as the filtered dataframes
        match_key_order = {key: i for i, key in enumerate(sorted(common_match_keys))}
        common_refs['_match_key_idx'] = common_refs['_match_key'].map(match_key_order)
        common_refs = common_refs.sort_values('_match_key_idx').reset_index(drop=True)
        
        # Keep the match keys in the common reports file for reference
        common_refs = common_refs.drop('_match_key_idx', axis=1)
        
        # Save common reports file
        common_filename = f"{output_dir}/{prefix}common_reports_{timestamp}.csv"
        common_refs.to_csv(common_filename, index=False)
        print(f"Saved {len(common_refs)} common reports to: {common_filename}")
        
        # Create a combined uncommon reports summary
        all_uncommon_dfs_list = []
        for uncommon_df_iter in uncommon_reports.values(): # uncommon_reports.values() are DFs of rows not in common_match_keys
            if not uncommon_df_iter.empty:
                all_uncommon_dfs_list.append(uncommon_df_iter)
        
        if all_uncommon_dfs_list:
            # Combine all uncommon reports into one dataframe
            uncommon_combined_df = pd.concat(all_uncommon_dfs_list, ignore_index=True)
            # Drop duplicates based on the match key to get a unique list of reports that were not common
            uncommon_combined_df = uncommon_combined_df.drop_duplicates(subset=['_match_key'], keep='first')
            
            # Save the unique uncommon reports. The _match_key column will be included for reference.
            uncommon_filename = f"{output_dir}/{prefix}uncommon_reports_{timestamp}.csv"
            uncommon_combined_df.to_csv(uncommon_filename, index=False)
            print(f"Saved {len(uncommon_combined_df)} unique uncommon reports to: {uncommon_filename}")
        else:
            print("No uncommon reports found to save.")
        
        # Also save a simple statistics summary
        summary_data = []
        for name, df in dataframes.items():
            summary_data.append({
                "Model": name,
                "Total_Reports": len(df),
                "Common_Reports": len(filtered_dfs[name]),
                "Uncommon_Reports": len(uncommon_reports[name]),
                "Percentage_Common": round(len(filtered_dfs[name]) / len(df) * 100, 2)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_filename = f"{output_dir}/{prefix}filtering_summary_{timestamp}.csv"
        summary_df.to_csv(summary_filename, index=False)
        print(f"Saved filtering summary to: {summary_filename}")
    
    # Clean up the temporary columns in the returned filtered dataframes
    for name, df in filtered_dfs.items():
        if '_match_key' in df.columns:
            filtered_dfs[name] = df.drop('_match_key', axis=1)
    
    # Final check - all filtered dataframes should have exactly the same number of rows
    row_counts = [len(df) for df in filtered_dfs.values()]
    if len(set(row_counts)) != 1:
        print("Warning: Filtered dataframes have different row counts. This should not happen!")
    else:
        print(f"Successfully filtered to {row_counts[0]} common reports across all models")
    
    return filtered_dfs

def evaluate_all_models(icd_type="icdo", n_bootstrap=500, ci=95, n_jobs=50, all_results=None):
    """
    Evaluate all models on the same set of reports for fair comparison.
    
    Args:
        icd_type: "icdo" or "icd10" for which type of codes to evaluate
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval percentage
        n_jobs: Number of parallel jobs
    """
    print(f"\n{'='*20} EVALUATING {icd_type.upper()} CODES {'='*20}\n")
    
    # Load all dataframes
    df_llama_70b = pd.read_csv("data/patho_icdo_prediction_llama_70b.csv")
    df_gemma3_12b = pd.read_csv("data/patho_icdo_prediction_gemma3_12b.csv")
    df_deepseek_70b = pd.read_csv("data/patho_icdo_prediction_deepseek_llama_70b.csv")
    df_deepseek_8b = pd.read_csv("data/patho_icdo_prediction_deepseek_llama_8b.csv")
    #df_qwen3_235 = pd.read_csv("data/patho_icdo_prediction_qwen3.csv")
    df_qwen3_235 = pd.read_csv("data/patho_icdo_prediction_qwen3_with_thinking.csv")
    
    print(f"Before preprocessing:")
    print(f"  Llama-3 70B: {len(df_llama_70b)} rows")
    print(f"  DeepSeek 70B: {len(df_deepseek_70b)} rows")
    print(f"  Gemma-3 12B: {len(df_gemma3_12b)} rows")
    print(f"  DeepSeek 8B: {len(df_deepseek_8b)} rows")
    print(f"  Qwen 3 235B: {len(df_qwen3_235)} rows")
    
    # Preprocess each dataframe depending on the ICD code type
    if icd_type == "icd10":
        df_llama_70b = preprocess_df_icd10(df_llama_70b)
        df_deepseek_70b = preprocess_deepseek_icd10(df_deepseek_70b)
        df_gemma3_12b = preprocess_df_icd10(df_gemma3_12b)
        df_deepseek_8b = preprocess_deepseek_icd10(df_deepseek_8b)
        df_qwen3_235 = preprocess_df_icd10(df_qwen3_235)
        
        gt_col = "GT_ICD-10"
        pred_col = "Generated_ICD-10"
    elif icd_type == "icd10_wo_locs": # or "Generated_ICD-10_wo_locs" if needed
        df_llama_70b = preprocess_df_icd10(df_llama_70b)
        df_deepseek_70b = preprocess_deepseek_icd10(df_deepseek_70b)
        df_gemma3_12b = preprocess_df_icd10(df_gemma3_12b)
        df_deepseek_8b = preprocess_deepseek_icd10(df_deepseek_8b)
        df_qwen3_235 = preprocess_df_icd10(df_qwen3_235)
        
        gt_col = "GT_ICD-10"
        pred_col = "Generated_ICD-10_wo_locs"
    else:  # icdo
        df_llama_70b = preprocess_df_icdo(df_llama_70b)
        df_deepseek_70b = preprocess_deepseek_icdo(df_deepseek_70b)
        df_gemma3_12b = preprocess_df_icdo(df_gemma3_12b)
        df_deepseek_8b = preprocess_deepseek_icdo(df_deepseek_8b)
        df_qwen3_235 = preprocess_df_icdo(df_qwen3_235)
        
        gt_col = "GT_ICD-O"
        pred_col = "Generated_ICD-O"
    
    print("After preprocessing:")
    print(f"  Llama-3 70B: {len(df_llama_70b)} rows")
    print(f"  DeepSeek 70B: {len(df_deepseek_70b)} rows")
    print(f"  Gemma-3 12B: {len(df_gemma3_12b)} rows")
    print(f"  DeepSeek 8B: {len(df_deepseek_8b)} rows")
    print(f"  Qwen 3 235B: {len(df_qwen3_235)} rows")
    
    # Create a dictionary of dataframes
    all_dfs = {
        "Llama-3 70B": df_llama_70b,
        "DeepSeek 70B": df_deepseek_70b,
        "Gemma-3 12B": df_gemma3_12b,
        "DeepSeek 8B": df_deepseek_8b,
        "Qwen 3 235B": df_qwen3_235,
    }
    
    # Filter to include only common reports using multiple matching columns
    #filtered_dfs = filter_common_reports(all_dfs, match_columns=['url', 'id', 'title', 'issued', "Befunde"])
    filtered_dfs = filter_common_reports(all_dfs, match_columns=['url'])
    
    #filtered_dfs["Llama-3 70B"].to_csv(f"results/{icd_type}_filtered_Llama-3_70B.csv", index=False)

    # Evaluate each model
    for model_name, df in filtered_dfs.items():
        print(f"\n{'-'*20} MODEL: {model_name} {'-'*20}")
        # Complete code evaluation
        print("Complete Code Evaluation:")
        micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1, subset_acc, hamming, tp, fp, fn, tn = evaluation_complete(df, gt_col, pred_col)
        print("Complete Code Evaluation Finished")
        # Three-character evaluation
        print("\nThree-Character Evaluation:")
        micro_p_3c, micro_r_3c, micro_f1_3c, macro_p_3c, macro_r_3c, macro_f1_3c, subset_acc_3c, hamming_3c, tp_3c, fp_3c, fn_3c, tn_3c = evaluation_three_characters(df, gt_col, pred_col)
        print("Three-Character Evaluation Finished")
        
        # Calculate confidence intervals
        print("\nCalculating confidence intervals...")
        # Print confidence intervals for all modelsa
        ci_complete, ci_three_char = evaluation_with_ci(df, gt_col, pred_col, n_bootstrap=n_bootstrap, ci=ci, n_jobs=n_jobs)
        print(f"Confidence intervals calculation finished")
        # Store results if needed
        if all_results is not None:
            all_results[icd_type][model_name] = {
                'complete': {
                    'micro_precision': micro_p,
                    'micro_precision_ci': ci_complete['micro_precision'],
                    'micro_recall': micro_r,
                    'micro_recall_ci': ci_complete['micro_recall'],
                    'micro_f1': micro_f1,
                    'micro_f1_ci': ci_complete['micro_f1'],
                    'macro_precision': macro_p,
                    'macro_precision_ci': ci_complete['macro_precision'],
                    'macro_recall': macro_r,
                    'macro_recall_ci': ci_complete['macro_recall'],
                    'macro_f1': macro_f1,
                    'macro_f1_ci': ci_complete['macro_f1'],
                    'subset_accuracy': subset_acc,
                    'subset_accuracy_ci': ci_complete['subset_accuracy'],
                    'hamming_loss': hamming,
                    'hamming_loss_ci': ci_complete['hamming_loss'],
                    'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
                },
                'three_char': {
                    'micro_precision': micro_p_3c,
                    'micro_precision_ci': ci_three_char['micro_precision'],
                    'micro_recall': micro_r_3c,
                    'micro_recall_ci': ci_three_char['micro_recall'],
                    'micro_f1': micro_f1_3c,
                    'micro_f1_ci': ci_three_char['micro_f1'],
                    'macro_precision': macro_p_3c,
                    'macro_precision_ci': ci_three_char['macro_precision'],
                    'macro_recall': macro_r_3c,
                    'macro_recall_ci': ci_three_char['macro_recall'],
                    'macro_f1': macro_f1_3c,
                    'macro_f1_ci': ci_three_char['macro_f1'],
                    'subset_accuracy': subset_acc_3c,
                    'subset_accuracy_ci': ci_three_char['subset_accuracy'],
                    'hamming_loss': hamming_3c,
                    'hamming_loss_ci': ci_three_char['hamming_loss'],
                    'tp': tp_3c, 'fp': fp_3c, 'fn': fn_3c, 'tn': tn_3c
                }
            }
            
if __name__ == "__main__":
    n_bootstrap = 1000
    n_parallel_jobs = 50
    
        # Create output directory
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store all results
    all_results = {
        "icd10": {
            "Llama-3 70B": {},
            "Gemma-3 12B": {},
            "DeepSeek 70B": {},
            "DeepSeek 8B": {},
            "Qwen 3 235B": {}
        },
        "icd10_wo_locs": {
            "Llama-3 70B": {},
            "Gemma-3 12B": {},
            "DeepSeek 70B": {},
            "DeepSeek 8B": {},
            "Qwen 3 235B": {}
        },
        "icdo": {
            "Llama-3 70B": {},
            "Gemma-3 12B": {},
            "DeepSeek 70B": {},
            "DeepSeek 8B": {},
            "Qwen 3 235B": {}
        }
    }
    
    # Evaluate ICD-10 codes
    evaluate_all_models(icd_type="icd10", n_bootstrap=n_bootstrap, ci=95, n_jobs=n_parallel_jobs, all_results=all_results)
    # Evaluate ICD-10 codes without locations
    evaluate_all_models(icd_type="icd10_wo_locs", n_bootstrap=n_bootstrap, ci=95, n_jobs=n_parallel_jobs, all_results=all_results)
    # Evaluate ICD-O codes
    evaluate_all_models(icd_type="icdo", n_bootstrap=n_bootstrap, ci=95, n_jobs=n_parallel_jobs, all_results=all_results)
    # Save all results to CSV files
    save_results_to_csv(all_results, output_dir=output_dir)