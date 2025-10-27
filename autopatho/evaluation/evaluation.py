import re
import ast
import json
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, multilabel_confusion_matrix

def load_localization_codes() -> list:
    loc_json_file = Path('data/localizations.json')
    if not loc_json_file.exists():
        loc = pd.read_excel('data/Diagnosecodeliste_ICD-O-T_DIMDI.xlsx')
        loc_codes = [{'code':row['Value'], 'localization':row['Bedeutung']} for _, row in loc.iterrows()]
        json.dump(loc_codes, open(str(loc_json_file), 'w'), indent=4)
        return loc_codes
    return json.load(open(str(loc_json_file)))

def extract_icd_code(text):
    match_icd_10 = re.findall(r'[A-TV-Z][0-9][0-9AB]\.?[0-9A-TV-Z]{0,4}', text)
    # remove "." suffix from the ICD-10 codes if it extists
    icd_10 = set(code.rstrip('.') for code in match_icd_10) if match_icd_10 else None
    icd_10 = [code.replace('-', '') for code in match_icd_10] if match_icd_10 else None
    icd_10 = match_icd_10 if match_icd_10 else None
    return icd_10
    
def extract_icd_o_code(text):
    match_icd_o = re.findall(r'\d{3,5}\/*\d{0,2}', text)
    # remove "." suffix from the ICD-10 codes if it extists
    icd_o = set(code.rstrip('.') for code in match_icd_o) if match_icd_o else None
    icd_o = match_icd_o if match_icd_o else None
    return icd_o

def preprocess_df_icd10(df):
    df = df[df["Generated_ICD-10"].isnull() == False]
    df = df[df["Generated_ICD-10_wo_locs"].isnull() == False]
    df["GT_ICD-10"] = df["GT_ICD-10"].apply(lambda x: set(ast.literal_eval(x)) if pd.notna(x) else set())

    df["Generated_ICD-10"] = df["Generated_ICD-10"].apply(lambda x: extract_icd_code(x) if pd.notna(x) else None)
    df["Generated_ICD-10_wo_locs"] = df["Generated_ICD-10_wo_locs"].apply(lambda x: extract_icd_code(x) if pd.notna(x) else None)

    df = df[df["Generated_ICD-10"].isnull() == False]
    df = df[df["Generated_ICD-10_wo_locs"].isnull() == False]
    
    return df

def preprocess_deepseek_icd10(df):
    df = df[df["Generated_ICD-10"].isnull() == False]
    df = df[df["Generated_ICD-10_wo_locs"].isnull() == False]

    df["GT_ICD-10"] = df["GT_ICD-10"].apply(lambda x: set(ast.literal_eval(x)))

    def remove_think_tags(text, tokenizer):
        # List of regex patterns to try
        patterns = [
            r'<think>.*?<\/think>[\r\n\\n]*',
            r'Okay.*?<\/think>[\r\n\\n]*',
            r'Alright.*?<\/think>[\r\n\\n]*',
            r"Okay.*?\*\*Antwort:\*\*(\s)*[\r\n\\n]*",
            r"Okay.*?Antwort:(\s)*[\r\n\\n]*",
            r"Zunächst.*?Antwort:(\s)*[\r\n\\n]*",
            r"Zunächst.*?<\/think>(\s)*[\r\n\\n]*",
            r"Dabei.*?<\/think>(\s)*[\r\n\\n]*"
        ]
        
        # List of prefixes that should be rejected
        invalid_prefixes = ["[<think>", "[Okay", "[Alright", "[Zunächst", "[Dabei"]
        
        # Try each pattern in sequence
        for pattern in patterns:
            # Clean the text using the current pattern
            extracted_text = re.sub(pattern, '', text).replace('\'', '').removesuffix('.')
            
            # Extract ICD codes
            icd_codes = extract_icd_code(extracted_text)
            
            # Check if we found valid codes and the text doesn't start with invalid prefixes
            if icd_codes \
            and not any(extracted_text.startswith(prefix) for prefix in invalid_prefixes) \
            and "</think>" not in extracted_text:
                #if len(icd_codes) > 4:
                    #print("More than 4 codes found")
                return icd_codes
        
        #print(text)
        #print("\n\n")
        return None
    
    loc_codes = load_localization_codes()
    prompt = [
        'Bestimme den am besten passenden Topographiecode der ICD-O-3 Klassifikation für die anatomische Lokalisation des Tumors aus dem folgenden Pathologiebefund.',
        '\nWähle den am besten passenden Code aus der folgenden Liste und gib ausschließlich den Code ohne zusätzliche Erklärung oder Beschreibung zurück.'
        '\nFalls keine exakte Übereinstimmung möglich ist, gib den nächstliegenden Code an.'
        '\nListe der möglichen Topographie-Codes: ' + ', '.join(f"{lc['localization']} ({lc['code']})" for lc in loc_codes),
        "\nAntwort: "
    ]
    prompt = '\n'.join(prompt)
    tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-large")
            
    for i, row in df.iterrows():
        report = row["Befunde"]
        encoded_prompt = tokenizer.encode_plus(prompt)
        prompt_length = len(encoded_prompt['input_ids'])
        report_length = len(tokenizer.encode_plus(report)['input_ids'])
        icd_codes = remove_think_tags(row["Generated_ICD-10"], tokenizer)
        
    
    df["Generated_ICD-10_wo_locs"] = df["Generated_ICD-10_wo_locs"].apply(lambda x: remove_think_tags(x) if pd.notna(x) else "")   
    # filter out rows with empty "Generated_ICD-10" column

    df = df[df["Generated_ICD-10"].isnull() == False]
    df = df[df["Generated_ICD-10_wo_locs"].isnull() == False]
    return df

def preprocess_df_icdo(df):
    df = df[df["Generated_ICD-O"].isnull() == False]
    df["GT_ICD-O"] = df["GT_ICD-O"].apply(lambda x: set(ast.literal_eval(x)) if pd.notna(x) else set())
    #df["Generated_ICD-O"] = df["Generated_ICD-O"].apply(lambda x: set(code.rstrip('.') for code in ast.literal_eval(x)) if pd.notna(x) else set())

    df["Generated_ICD-O"] = df["Generated_ICD-O"].apply(lambda x: extract_icd_o_code(x) if pd.notna(x) else None)
    # filter out rows with empty "Generated_ICD-10" column
    df = df[df["Generated_ICD-O"].isnull() == False]
    return df

def preprocess_deepseek_icdo(df):
    df = df[df["Generated_ICD-O"].isnull() == False]

    df["GT_ICD-O"] = df["GT_ICD-O"].apply(lambda x: set(ast.literal_eval(x)))

    # remove <think> tags and their content from the generated ICD-10 codes
    def remove_think_tags(text):
        #extracted_text = re.sub(r'<think>.*?</think>\\n\\n', '', text).replace('\'', '').removesuffix('.')
        patterns = [
            r'Okay.*?<\/think>[\r\n\\n]*',
            r'<think>.*?<\/think>[\r\n\\n]*',
            r'Alright.*?<\/think>[\r\n\\n]*',
            r'Ok.*?<\/think>[\r\n\\n]*',
            r'Hmm.*?<\/think>[\r\n\\n]*',
        ]
        for pattern in patterns:
            extracted_text = re.sub(pattern, '', text).replace('\'', '').removesuffix('.')
            icd_o_codes = extract_icd_o_code(extracted_text)
            # Check if we found valid codes
            if icd_o_codes and "</think>" not in extracted_text:
                #if len(icd_o_codes) > 4:
                    #print("More than 4 codes found")  
                return icd_o_codes
        return None

    df["Generated_ICD-O"] = df["Generated_ICD-O"].apply(lambda x: remove_think_tags(x) if pd.notna(x) else None)

    # filter out rows with empty "Generated_ICD-10" column
    df = df[df["Generated_ICD-O"].isnull() == False]
    return df

def evaluation_complete(df, gt_column, pred_column):
    """Evaluate complete ICD-10 codes using multi-label classification metrics."""
    
    # Collect all unique ICD codes for binarization
    all_codes = set()
    for idx, row in df.iterrows():
        all_codes.update(row[gt_column])
        all_codes.update(row[pred_column])
        #all_codes.update(row["Generated_ICD-10_wo_locs"])
    
    # Remove incomplete codes (ending with '-') from the codes for binarization
    all_complete_codes = {code for code in all_codes if not code.endswith('-')}
    
    # Initialize multi-label binarizer
    mlb = MultiLabelBinarizer(classes=sorted(list(all_complete_codes)))
    
    # Prepare ground truth and predictions in appropriate format
    y_true_lists = []
    y_pred_lists = []
    
    for idx, row in df.iterrows():
        gt = row[gt_column]
        preds = set(row[pred_column])
        
        # Process ground truth codes ending with "-"
        incomplete_codes = {code for code in gt if code.endswith('-')}
        complete_gt = gt - incomplete_codes
        
        # Add complete ground truth codes
        gt_for_eval = set(complete_gt)
        
        # Handle incomplete codes by finding matching predictions
        for inc_code in incomplete_codes:
            prefix = inc_code[:-1]
            matches = {pred for pred in preds if pred.startswith(prefix)}
            if matches:
                # If we have matching predictions, add one of them to ground truth
                # This treats predicting any code that matches the prefix as correct
                gt_for_eval.add(next(iter(matches)))
            else:
                # If no matches, we'd normally count this as a false negative
                # But we can't add the incomplete code directly to the binary matrix
                # So we'll track these separately
                pass
        
        y_true_lists.append(list(gt_for_eval))
        y_pred_lists.append(list(preds))
    
    # Transform to multi-label format
    y_true_binary = mlb.fit_transform(y_true_lists)
    y_pred_binary = mlb.transform(y_pred_lists)
    
        # Calculate multilabel confusion matrix
    mcm = multilabel_confusion_matrix(y_true_binary, y_pred_binary)
    
    # Calculate aggregated confusion matrix values
    tn = np.sum(mcm[:, 0, 0])  # True Negatives
    fp = np.sum(mcm[:, 0, 1])  # False Positives
    fn = np.sum(mcm[:, 1, 0])  # False Negatives
    tp = np.sum(mcm[:, 1, 1])  # True Positives
    
    # Calculate metrics
    micro_precision = precision_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
    micro_recall = recall_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
    micro_f1 = f1_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
    
    macro_precision = precision_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
    macro_recall = recall_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
    
    # Calculate subset accuracy (exact match)
    subset_accuracy = accuracy_score(y_true_binary, y_pred_binary)
    
    # Calculate Hamming loss (fraction of labels incorrectly predicted)
    hamming_loss = np.sum(np.logical_xor(y_true_binary, y_pred_binary)) / (y_true_binary.shape[0] * y_true_binary.shape[1])
    
    return micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1, subset_accuracy, hamming_loss, tp, fp, fn, tn

def evaluation_three_characters(df, gt_column, pred_column):
    """Evaluate first three characters of ICD-10 codes using multi-label classification metrics."""
    
    # Collect all unique 3-character ICD codes
    all_3char_codes = set()
    for idx, row in df.iterrows():
        gt = row[gt_column]
        all_3char_codes.update(code[:3] for code in gt)
        
        preds = row[pred_column]
        #preds = set(row["Generated_ICD-10_wo_locs"])
        all_3char_codes.update(code[:3] for code in preds)
    
    # Initialize multi-label binarizer
    mlb = MultiLabelBinarizer(classes=sorted(list(all_3char_codes)))
    
    # Prepare ground truth and predictions in appropriate format
    y_true_lists = []
    y_pred_lists = []
    
    for idx, row in df.iterrows():
        gt = row[gt_column]
        preds = set(row[pred_column])
        #preds = set(row["Generated_ICD-10_wo_locs"])
        
        gt_3char = {code[:3] for code in gt}
        pred_3char = {code[:3] for code in preds} if preds else set()
        
        y_true_lists.append(list(gt_3char))
        y_pred_lists.append(list(pred_3char))
    
    # Transform to multi-label format
    y_true_binary = mlb.fit_transform(y_true_lists)
    y_pred_binary = mlb.transform(y_pred_lists)
    
    # Calculate multilabel confusion matrix
    mcm = multilabel_confusion_matrix(y_true_binary, y_pred_binary)
    
    # Calculate aggregated confusion matrix values
    tn = np.sum(mcm[:, 0, 0])  # True Negatives
    fp = np.sum(mcm[:, 0, 1])  # False Positives
    fn = np.sum(mcm[:, 1, 0])  # False Negatives
    tp = np.sum(mcm[:, 1, 1])  # True Positives
    
    # Calculate metrics
    micro_precision = precision_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
    micro_recall = recall_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
    micro_f1 = f1_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
    
    macro_precision = precision_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
    macro_recall = recall_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
    
    # Calculate subset accuracy (exact match)
    subset_accuracy = accuracy_score(y_true_binary, y_pred_binary)
    
    # Calculate Hamming loss (fraction of labels incorrectly predicted)
    hamming_loss = np.sum(np.logical_xor(y_true_binary, y_pred_binary)) / (y_true_binary.shape[0] * y_true_binary.shape[1])
    
    return micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1, subset_accuracy, hamming_loss, tp, fp, fn, tn

def evaluation_complete_per_class(df, gt_column, pred_column):
    """
    Evaluate and return performance metrics for each individual ICD code.
    
    Returns a DataFrame with precision, recall, F1-score, support, and confusion 
    matrix values for each ICD code.
    """
    # Collect all unique ICD codes
    all_codes = set()
    for idx, row in df.iterrows():
        all_codes.update(row[gt_column])
        all_codes.update(row[pred_column])
    
    # Remove incomplete codes (ending with '-')
    all_complete_codes = {code for code in all_codes if not code.endswith('-')}
    
    # Initialize multi-label binarizer
    mlb = MultiLabelBinarizer(classes=sorted(list(all_complete_codes)))
    
    # Prepare ground truth and predictions
    y_true_lists = []
    y_pred_lists = []
    
    for idx, row in df.iterrows():
        gt = row[gt_column]
        preds = set(row[pred_column])
        
        # Process ground truth codes ending with "-"
        incomplete_codes = {code for code in gt if code.endswith('-')}
        complete_gt = gt - incomplete_codes
        
        # Add complete ground truth codes
        gt_for_eval = set(complete_gt)
        
        # Handle incomplete codes by finding matching predictions
        for inc_code in incomplete_codes:
            prefix = inc_code[:-1]
            matches = {pred for pred in preds if pred.startswith(prefix)}
            if matches:
                gt_for_eval.add(next(iter(matches)))
        
        y_true_lists.append(list(gt_for_eval))
        y_pred_lists.append(list(preds))
    
    # Transform to multi-label format
    y_true_binary = mlb.fit_transform(y_true_lists)
    y_pred_binary = mlb.transform(y_pred_lists)
    
    # Get class names
    class_names = mlb.classes_
    
    # Calculate per-class metrics
    results = []
    
    for i, class_name in enumerate(class_names):
        # Extract true and predicted values for this class
        y_true_class = y_true_binary[:, i]
        y_pred_class = y_pred_binary[:, i]
        
        # Calculate confusion matrix values for this class
        tp = np.sum((y_true_class == 1) & (y_pred_class == 1))
        fp = np.sum((y_true_class == 0) & (y_pred_class == 1))
        fn = np.sum((y_true_class == 1) & (y_pred_class == 0))
        tn = np.sum((y_true_class == 0) & (y_pred_class == 0))
        
        # Calculate metrics
        support = tp + fn
        
        # Calculate precision, recall, and F1 with zero_division=0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'ICD_code': class_name,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support,
            'TP': tp,
            'FP': fp, 
            'FN': fn,
            'TN': tn
        })
    
    # Convert to DataFrame and sort by support (frequency) descending
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('support', ascending=False)
    
    return results_df

def evaluation_three_characters_per_class(df, gt_column, pred_column):
    """
    Evaluate and return performance metrics for each individual ICD code.
    
    Returns a DataFrame with precision, recall, F1-score, support, and confusion 
    matrix values for each ICD code.
    """
    all_3char_codes = set()
    for idx, row in df.iterrows():
        gt = row[gt_column]
        all_3char_codes.update(code[:3] for code in gt)
        
        preds = row[pred_column]
        #preds = set(row["Generated_ICD-10_wo_locs"])
        all_3char_codes.update(code[:3] for code in preds)
    
    # Initialize multi-label binarizer
    mlb = MultiLabelBinarizer(classes=sorted(list(all_3char_codes)))
    
    # Prepare ground truth and predictions in appropriate format
    y_true_lists = []
    y_pred_lists = []
    
    for idx, row in df.iterrows():
        gt = row[gt_column]
        preds = set(row[pred_column])
        #preds = set(row["Generated_ICD-10_wo_locs"])
        
        gt_3char = {code[:3] for code in gt}
        pred_3char = {code[:3] for code in preds} if preds else set()
        
        y_true_lists.append(list(gt_3char))
        y_pred_lists.append(list(pred_3char))
    
    # Transform to multi-label format
    y_true_binary = mlb.fit_transform(y_true_lists)
    y_pred_binary = mlb.transform(y_pred_lists)
    
    # Get class names
    class_names = mlb.classes_
    
    # Calculate per-class metrics
    results = []
    
    for i, class_name in enumerate(class_names):
        # Extract true and predicted values for this class
        y_true_class = y_true_binary[:, i]
        y_pred_class = y_pred_binary[:, i]
        
        # Calculate confusion matrix values for this class
        tp = np.sum((y_true_class == 1) & (y_pred_class == 1))
        fp = np.sum((y_true_class == 0) & (y_pred_class == 1))
        fn = np.sum((y_true_class == 1) & (y_pred_class == 0))
        tn = np.sum((y_true_class == 0) & (y_pred_class == 0))
        
        # Calculate metrics
        support = tp + fn
        
        # Calculate precision, recall, and F1 with zero_division=0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'ICD_code': class_name,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support,
            'TP': tp,
            'FP': fp, 
            'FN': fn,
            'TN': tn
        })
    
    # Convert to DataFrame and sort by support (frequency) descending
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('support', ascending=False)
    
    return results_df

def evaluation_complete_instance_based(df, gt_column, pred_column):
    """
    Evaluate precision, recall, and F1-score for each instance (row) in the DataFrame.
    Adds the metrics as new columns to the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing ground truth and predictions.
        gt_column (str): The column name for ground truth labels.
        pred_column (str): The column name for predicted labels.

    Returns:
        pd.DataFrame: The DataFrame with added columns for instance-based metrics.
    """
    # Initialize lists to store metrics for each row
    precisions = []
    recalls = []
    f1_scores = []

    for idx, row in df.iterrows():
        gt = set(row[gt_column])
        preds = set(row[pred_column])

        # Calculate true positives, false positives, and false negatives
        tp = len(gt & preds)
        fp = len(preds - gt)
        fn = len(gt - preds)

        # Calculate precision, recall, and F1-score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Append metrics to the lists
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Add metrics as new columns to the DataFrame
    df['Instance_Precision'] = precisions
    df['Instance_Recall'] = recalls
    df['Instance_F1'] = f1_scores

    return df

def evaluation_three_chars_instance_based(df, gt_column, pred_column):
    """
    Evaluate precision, recall, and F1-score for each instance (row) in the DataFrame.
    Adds the metrics as new columns to the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing ground truth and predictions.
        gt_column (str): The column name for ground truth labels.
        pred_column (str): The column name for predicted labels.

    Returns:
        pd.DataFrame: The DataFrame with added columns for instance-based metrics.
    """
    # Initialize lists to store metrics for each row
    precisions = []
    recalls = []
    f1_scores = []

    for idx, row in df.iterrows():
        gt = set(row[gt_column])
        preds = set(row[pred_column])
        
        gt_3char = {code[:3] for code in gt}
        pred_3char = {code[:3] for code in preds} if preds else set()

        # Calculate true positives, false positives, and false negatives
        tp = len(gt_3char & pred_3char)
        fp = len(pred_3char - gt_3char)
        fn = len(gt_3char - pred_3char)

        # Calculate precision, recall, and F1-score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Append metrics to the lists
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Add metrics as new columns to the DataFrame
    df['Instance_Precision'] = precisions
    df['Instance_Recall'] = recalls
    df['Instance_F1'] = f1_scores

    return df

if __name__ == "__main__":
    # Start with ICD-OT with context
    gt_column = "GT_ICD-10"
    pred_column = "Generated_ICD-10"
    
    # Instance-based evaluation for ICD-10 codes - Deepseek 8B
    df_deepseek_8b = pd.read_csv("data/patho_icdo_prediction_deepseek_llama_8b.csv")
    df_deepseek_8b = preprocess_deepseek_icd10(df_deepseek_8b)
    deepseek_8b_instance_three_chars = evaluation_three_chars_instance_based(df_deepseek_8b, gt_column, pred_column)
    deepseek_8b_instance_three_chars.to_csv("results/instance-based/icd10_three_chars_instance_based_metrics_Deepseek-8B.csv", index=False)
    deepseek_8b_instance_complete = evaluation_complete_instance_based(df_deepseek_8b, gt_column, pred_column)
    deepseek_8b_instance_complete.to_csv("results/instance-based/icd10_complete_instance_based_metrics_Deepseek-8B.csv", index=False)