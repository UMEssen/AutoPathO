import os
import ast
import glob
import json
import torch
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, recall_score, precision_score
from transformers import Trainer, AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification


# Configuration parameters - MUST match your training script
TARGET_COLUMN = "GT_ICD-O"  # Change this to match what you trained on
model_name = 'EuroBERT/EuroBERT-610m'

# Determine model suffix based on target column
if TARGET_COLUMN == "GT_ICD-10":
    model_suffix = "icd10"
    target_type = "ICD-10"
    df = pd.read_csv('data/icd-10/patho_icdo_test_dataset.csv')
elif TARGET_COLUMN == "GT_ICD-O":
    model_suffix = "icdo"
    target_type = "ICD-O"
    df = pd.read_csv('data/icd-o/common_reports_20251010_110642.csv')
else:
    raise ValueError(f"Unsupported target column: {TARGET_COLUMN}")

print(f"Evaluation configuration:")
print(f"  Target column: {TARGET_COLUMN}")
print(f"  Target type: {target_type}")
print(f"  Model suffix: {model_suffix}")

# Create directories for cached data
cache_dir = f'./cached_data/{model_suffix}'
os.makedirs(cache_dir, exist_ok=True)

def save_dataset_splits(df, target_encoded, train_idx, test_idx, target_classes):
    """Save dataset splits to CSV files for future use"""
    print("Saving dataset splits to CSV files...")
    
    # Save full dataset with encoded targets
    df_with_encoded = df.copy()
    df_with_encoded['target_encoded_str'] = [str(row.tolist()) for row in target_encoded]
    df_with_encoded.to_csv(os.path.join(cache_dir, 'full_dataset_with_encoded.csv'), index=False)
    
    # Save train/test indices
    split_info = {
        'train_indices': train_idx.tolist(),
        'test_indices': test_idx.tolist(),
        'target_classes': target_classes.tolist(),
        'num_labels': len(target_classes),
        'target_column': TARGET_COLUMN,
        'random_seed': 42
    }
    
    with open(os.path.join(cache_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Save test set separately for easy access
    test_df = df.iloc[test_idx].copy()
    test_df['target_encoded_str'] = [str(target_encoded[i].tolist()) for i in test_idx]
    test_df.to_csv(os.path.join(cache_dir, 'test_dataset.csv'), index=False)
    
    print(f"Dataset splits saved to {cache_dir}")

def load_dataset_splits():
    """Load dataset splits from CSV files if they exist"""
    split_file = os.path.join(cache_dir, 'split_info.json')
    test_file = os.path.join(cache_dir, 'test_dataset.csv')
    
    if os.path.exists(split_file) and os.path.exists(test_file):
        print("Loading cached dataset splits...")
        
        # Load split info
        with open(split_file, 'r') as f:
            split_info = json.load(f)
        
        # Verify it matches current configuration
        if (split_info['target_column'] == TARGET_COLUMN and 
            split_info['random_seed'] == 42):
            
            # Load test dataset
            test_df = pd.read_csv(test_file)
            
            # Reconstruct target encoding
            target_encoded_test = np.array([
                ast.literal_eval(row) for row in test_df['target_encoded_str']
            ])
            
            X_test = test_df['Befunde_filtered'].values
            test_groups = test_df['pid'].values
            target_classes = np.array(split_info['target_classes'])
            
            print(f"Loaded cached splits: {len(X_test)} test samples")
            return X_test, target_encoded_test, test_groups, target_classes, split_info['num_labels']
    
    return None

# Try to load cached data first
cached_data = load_dataset_splits()

if cached_data is not None:
    X_test, y_test, test_groups, target_classes, num_labels = cached_data
else:
    # Load and prepare data (original logic)

    # Convert target codes to lists if they're in string format
    def convert_to_list(x):
        if isinstance(x, str):
            try:
                return list(set(ast.literal_eval(x)))
            except:
                return [x.strip()]
        elif isinstance(x, list):
            return x
        else:
            return []

    # Apply conversion to the selected target column
    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(convert_to_list)

    # One-hot encode the target codes
    mlb = MultiLabelBinarizer()
    target_encoded = mlb.fit_transform(df[TARGET_COLUMN])
    target_classes = mlb.classes_
    num_labels = len(target_classes)

    print(f"Number of unique {target_type} codes: {num_labels}")

    # Recreate the same train-test split used during training
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['pid']))

    # Extract holdout test set (same as used in training)
    X_test = df['Befunde'].values[test_idx]
    y_test = target_encoded[test_idx]
    test_groups = df['pid'].values[test_idx]
    
    # Save splits for future use
    save_dataset_splits(df, target_encoded, train_idx, test_idx, target_classes)

print(f"Holdout test samples: {len(X_test)}")
print(f"Unique groups in test set: {len(np.unique(test_groups))}")

# Dataset class (same as training)
class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=8192):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx].astype(np.float32)
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }
        
        return item

# Metrics computation function (same as training)
def compute_metrics(pred):
    logits = pred.predictions
    labels = pred.label_ids
    
    # Convert logits to binary predictions
    preds = (logits > 0).astype(int)
    
    # Calculate metrics
    micro_f1 = f1_score(labels, preds, average='micro')
    macro_f1 = f1_score(labels, preds, average='macro')
    micro_precision = precision_score(labels, preds, average='micro', zero_division=0)
    micro_recall = recall_score(labels, preds, average='micro', zero_division=0)
    macro_precision = precision_score(labels, preds, average='macro', zero_division=0)
    macro_recall = recall_score(labels, preds, average='macro', zero_division=0)
    
    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall
    }

def extract_code_prefix(code, prefix_length=3):
    """Extract the first N characters of a medical code"""
    if isinstance(code, str):
        return code[:prefix_length]
    return code

def compute_prefix_metrics(logits, labels, target_classes, prefix_length=3):
    """Compute metrics based on code prefixes (e.g., first 3 characters for ICD groups)"""
    
    # Convert logits to binary predictions
    preds = (logits > 0).astype(int)
    
    # Create mapping from full codes to prefixes
    full_to_prefix = {}
    prefix_to_indices = {}
    
    for i, code in enumerate(target_classes):
        prefix = extract_code_prefix(code, prefix_length)
        full_to_prefix[i] = prefix
        
        if prefix not in prefix_to_indices:
            prefix_to_indices[prefix] = []
        prefix_to_indices[prefix].append(i)
    
    # Get unique prefixes
    unique_prefixes = list(prefix_to_indices.keys())
    
    # Convert predictions and labels to prefix level
    n_samples = len(labels)
    n_prefixes = len(unique_prefixes)
    
    prefix_preds = np.zeros((n_samples, n_prefixes), dtype=int)
    prefix_labels = np.zeros((n_samples, n_prefixes), dtype=int)
    
    for sample_idx in range(n_samples):
        for prefix_idx, prefix in enumerate(unique_prefixes):
            # If any full code with this prefix is predicted/true, mark prefix as predicted/true
            indices = prefix_to_indices[prefix]
            
            # Prediction: OR across all codes with this prefix
            prefix_preds[sample_idx, prefix_idx] = int(np.any(preds[sample_idx, indices]))
            
            # True label: OR across all codes with this prefix
            prefix_labels[sample_idx, prefix_idx] = int(np.any(labels[sample_idx, indices]))
    
    # Calculate metrics at prefix level
    prefix_micro_f1 = f1_score(prefix_labels, prefix_preds, average='micro')
    prefix_macro_f1 = f1_score(prefix_labels, prefix_preds, average='macro')
    prefix_micro_precision = precision_score(prefix_labels, prefix_preds, average='micro', zero_division=0)
    prefix_micro_recall = recall_score(prefix_labels, prefix_preds, average='micro', zero_division=0)
    prefix_macro_precision = precision_score(prefix_labels, prefix_preds, average='macro', zero_division=0)
    prefix_macro_recall = recall_score(prefix_labels, prefix_preds, average='macro', zero_division=0)
    
    return {
        'prefix_micro_f1': prefix_micro_f1,
        'prefix_macro_f1': prefix_macro_f1,
        'prefix_micro_precision': prefix_micro_precision,
        'prefix_micro_recall': prefix_micro_recall,
        'prefix_macro_precision': prefix_macro_precision,
        'prefix_macro_recall': prefix_macro_recall,
        'unique_prefixes': unique_prefixes,
        'prefix_predictions': prefix_preds,
        'prefix_labels': prefix_labels,
        'full_to_prefix_mapping': full_to_prefix
    }

def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for a list of values"""
    n = len(values)
    mean = np.mean(values)
    std_err = stats.sem(values)  # Standard error of the mean
    
    # Calculate confidence interval using t-distribution
    h = std_err * stats.t.ppf((1 + confidence) / 2., n-1)
    
    return mean - h, mean + h

def find_cv_models() -> List[str]:
    """Find all available CV model directories"""
    pattern = f'./models/eurobert_{model_suffix}_classifier_fold_*'
    model_dirs = glob.glob(pattern)
    model_dirs.sort()  # Sort to ensure consistent order
    
    # Validate that all required files exist
    valid_dirs = []
    for model_dir in model_dirs:
        config_path = os.path.join(model_dir, 'config.json')
        model_path = os.path.join(model_dir, 'model.safetensors')
        tokenizer_path = os.path.join(model_dir, 'tokenizer.json')
        mlb_path = os.path.join(model_dir, 'multilabelbinarizer.pkl')
        
        if all(os.path.exists(p) for p in [config_path, model_path, tokenizer_path, mlb_path]):
            valid_dirs.append(model_dir)
        else:
            print(f"Warning: Skipping incomplete model directory: {model_dir}")
    
    return valid_dirs

def evaluate_single_model(model_dir: str, tokenizer, test_dataset) -> Dict:
    """Evaluate a single fold model on the test set"""
    fold_num = os.path.basename(model_dir).split('_')[-1]
    
    # Check if predictions are already cached
    predictions_file = os.path.join(cache_dir, f'predictions_fold_{fold_num}.npz')
    
    if os.path.exists(predictions_file):
        print(f"Loading cached predictions for Fold {fold_num}...")
        
        # Load cached predictions
        data = np.load(predictions_file)
        logits = data['logits']
        labels = data['labels']
        
        # Calculate metrics from cached predictions
        preds = (logits > 0).astype(int)
        
        # Full code metrics
        micro_f1 = f1_score(labels, preds, average='micro')
        macro_f1 = f1_score(labels, preds, average='macro')
        micro_precision = precision_score(labels, preds, average='micro', zero_division=0)
        micro_recall = recall_score(labels, preds, average='micro', zero_division=0)
        macro_precision = precision_score(labels, preds, average='macro', zero_division=0)
        macro_recall = recall_score(labels, preds, average='macro', zero_division=0)
        
        # Prefix metrics (first 3 characters)
        prefix_metrics = compute_prefix_metrics(logits, labels, target_classes, prefix_length=3)
        
        metrics = {
            'fold': int(fold_num),
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            # Add prefix metrics
            'prefix_micro_f1': prefix_metrics['prefix_micro_f1'],
            'prefix_macro_f1': prefix_metrics['prefix_macro_f1'],
            'prefix_micro_precision': prefix_metrics['prefix_micro_precision'],
            'prefix_micro_recall': prefix_metrics['prefix_micro_recall'],
            'prefix_macro_precision': prefix_metrics['prefix_macro_precision'],
            'prefix_macro_recall': prefix_metrics['prefix_macro_recall']
        }
        
        print(f"  Fold {fold_num} (cached) - Full: Micro F1: {metrics['micro_f1']:.4f}, Prefix: Micro F1: {metrics['prefix_micro_f1']:.4f}")
        return metrics
    
    # If not cached, run full evaluation
    print(f"Evaluating Fold {fold_num}...")
    
    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        trust_remote_code=True,
    )
    
    # Create trainer for evaluation
    eval_args = TrainingArguments(
        output_dir='./temp_eval',
        per_device_eval_batch_size=64,
        fp16=torch.cuda.is_available(),
    )
    
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Evaluate model and get raw outputs
    results = trainer.evaluate()
    
    # Get predictions for caching
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    labels = predictions.label_ids
    
    # Calculate prefix metrics
    prefix_metrics = compute_prefix_metrics(logits, labels, target_classes, prefix_length=3)
    
    # Save predictions to cache
    np.savez_compressed(
        predictions_file,
        logits=logits,
        labels=labels,
        fold=int(fold_num)
    )
    print(f"  Predictions cached to {predictions_file}")
    
    # Extract metric values including prefix metrics
    metrics = {
        'fold': int(fold_num),
        'micro_f1': results['eval_micro_f1'],
        'macro_f1': results['eval_macro_f1'],
        'micro_precision': results['eval_micro_precision'],
        'micro_recall': results['eval_micro_recall'],
        'macro_precision': results['eval_macro_precision'],
        'macro_recall': results['eval_macro_recall'],
        # Add prefix metrics
        'prefix_micro_f1': prefix_metrics['prefix_micro_f1'],
        'prefix_macro_f1': prefix_metrics['prefix_macro_f1'],
        'prefix_micro_precision': prefix_metrics['prefix_micro_precision'],
        'prefix_micro_recall': prefix_metrics['prefix_micro_recall'],
        'prefix_macro_precision': prefix_metrics['prefix_macro_precision'],
        'prefix_macro_recall': prefix_metrics['prefix_macro_recall']
    }
    
    print(f"  Fold {fold_num} - Full: Micro F1: {metrics['micro_f1']:.4f}, Prefix: Micro F1: {metrics['prefix_micro_f1']:.4f}")
    
    return metrics

def clear_cache():
    """Clear all cached data - useful if you want to force regeneration"""
    import shutil
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cache cleared: {cache_dir}")

def main():
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION MODEL EVALUATION ON HOLDOUT TEST SET")
    print(f"Target: {target_type} ({TARGET_COLUMN})")
    print(f"{'='*60}")
    
    # Check if user wants to clear cache (you can modify this condition)
    # Uncomment the next line if you want to force cache clearing
    # clear_cache()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Create test dataset
    test_dataset = MultiLabelDataset(X_test, y_test, tokenizer)
    
    # Find all CV models
    model_dirs = find_cv_models()
    
    if not model_dirs:
        print(f"Error: No valid CV models found with pattern './models/eurobert_{model_suffix}_classifier_fold_*'")
        print("Please ensure you have trained models using the cross-validation script.")
        return
    
    print(f"Found {len(model_dirs)} valid CV models:")
    for model_dir in model_dirs:
        print(f"  {model_dir}")
    
    # Evaluate all models
    all_results = []
    for model_dir in model_dirs:
        try:
            results = evaluate_single_model(model_dir, tokenizer, test_dataset)
            all_results.append(results)
        except Exception as e:
            print(f"Error evaluating {model_dir}: {str(e)}")
            continue
    
    if not all_results:
        print("Error: No models could be evaluated successfully.")
        return
    
    # Calculate statistics across all folds
    full_metrics = ['micro_f1', 'macro_f1', 'micro_precision', 'micro_recall', 'macro_precision', 'macro_recall']
    prefix_metrics = ['prefix_micro_f1', 'prefix_macro_f1', 'prefix_micro_precision', 'prefix_micro_recall', 'prefix_macro_precision', 'prefix_macro_recall']
    all_metrics = full_metrics + prefix_metrics
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS SUMMARY - {target_type}")
    print(f"{'='*60}")
    print(f"Number of folds evaluated: {len(all_results)}")
    print(f"Test set size: {len(X_test)} samples")
    print(f"Number of unique {target_type} codes: {num_labels}")
    
    # Get unique prefixes for summary
    if all_results:
        sample_logits = None
        sample_labels = None
        for model_dir in model_dirs:
            fold_num = os.path.basename(model_dir).split('_')[-1]
            predictions_file = os.path.join(cache_dir, f'predictions_fold_{fold_num}.npz')
            if os.path.exists(predictions_file):
                data = np.load(predictions_file)
                sample_logits = data['logits']
                sample_labels = data['labels']
                break
        
        if sample_logits is not None:
            prefix_info = compute_prefix_metrics(sample_logits, sample_labels, target_classes, prefix_length=3)
            unique_prefixes = prefix_info['unique_prefixes']
            print(f"Number of unique 3-character prefixes: {len(unique_prefixes)}")
    
    print()
    
    statistics = {}
    
    # Calculate statistics for full code metrics
    print("FULL CODE EVALUATION:")
    print("-" * 25)
    for metric in full_metrics:
        values = [result[metric] for result in all_results]
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)  # Sample standard deviation
        min_val = np.min(values)
        max_val = np.max(values)
        
        # Calculate 95% confidence interval
        ci_lower, ci_upper = calculate_confidence_interval(values, confidence=0.95)
        
        statistics[metric] = {
            'values': values,
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'n_folds': len(values)
        }
        
        # Format metric name for display
        metric_display = metric.replace('_', ' ').title()
        
        print(f"{metric_display}:")
        print(f"  Mean ± Std:     {mean_val:.4f} ± {std_val:.4f}")
        print(f"  95% CI:         [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  Range:          [{min_val:.4f}, {max_val:.4f}]")
        print()
    
    # Calculate statistics for prefix metrics
    print("PREFIX (3-CHARACTER) EVALUATION:")
    print("-" * 35)
    for metric in prefix_metrics:
        values = [result[metric] for result in all_results]
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        min_val = np.min(values)
        max_val = np.max(values)
        
        # Calculate 95% confidence interval
        ci_lower, ci_upper = calculate_confidence_interval(values, confidence=0.95)
        
        statistics[metric] = {
            'values': values,
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'n_folds': len(values)
        }
        
        # Format metric name for display
        metric_display = metric.replace('prefix_', '').replace('_', ' ').title()
        
        print(f"{metric_display}:")
        print(f"  Mean ± Std:     {mean_val:.4f} ± {std_val:.4f}")
        print(f"  95% CI:         [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  Range:          [{min_val:.4f}, {max_val:.4f}]")
        print()
    
    # Comparison between full and prefix performance
    print("FULL vs PREFIX PERFORMANCE COMPARISON:")
    print("-" * 40)
    full_micro_f1_mean = statistics['micro_f1']['mean']
    prefix_micro_f1_mean = statistics['prefix_micro_f1']['mean']
    improvement = prefix_micro_f1_mean - full_micro_f1_mean
    
    print(f"Full Code Micro F1:      {full_micro_f1_mean:.4f}")
    print(f"Prefix (3-char) Micro F1: {prefix_micro_f1_mean:.4f}")
    print(f"Prefix Improvement:      {improvement:+.4f} ({improvement/full_micro_f1_mean*100:+.1f}%)")
    print()
    
    # Identify best and worst performing folds
    micro_f1_values = [result['micro_f1'] for result in all_results]
    best_fold_idx = np.argmax(micro_f1_values)
    worst_fold_idx = np.argmin(micro_f1_values)
    
    print("Best performing fold (by Micro F1):")
    best_result = all_results[best_fold_idx]
    print(f"  Fold {best_result['fold']}: Micro F1 = {best_result['micro_f1']:.4f}")
    
    print("Worst performing fold (by Micro F1):")
    worst_result = all_results[worst_fold_idx]
    print(f"  Fold {worst_result['fold']}: Micro F1 = {worst_result['micro_f1']:.4f}")
    
    print(f"Performance spread: {micro_f1_values[best_fold_idx] - micro_f1_values[worst_fold_idx]:.4f}")
    
    # Save detailed results
    output_dir = f'./evaluation_results/{model_suffix}_cv_evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual fold results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, 'individual_fold_results.csv'), index=False)
    
    # Create detailed predictions DataFrame
    print("Creating detailed predictions file...")
    predictions_data = []
    
    for model_dir in model_dirs:
        fold_num = os.path.basename(model_dir).split('_')[-1]
        predictions_file = os.path.join(cache_dir, f'predictions_fold_{fold_num}.npz')
        
        if os.path.exists(predictions_file):
            data = np.load(predictions_file)
            logits = data['logits']
            labels = data['labels']
            preds = (logits > 0).astype(int)
            
            # Calculate prefix predictions
            prefix_info = compute_prefix_metrics(logits, labels, target_classes, prefix_length=3)
            prefix_preds = prefix_info['prefix_predictions']
            prefix_labels = prefix_info['prefix_labels']
            unique_prefixes = prefix_info['unique_prefixes']
            full_to_prefix = prefix_info['full_to_prefix_mapping']
            
            # Add predictions for each sample
            for i in range(len(X_test)):
                # Convert full code predictions and labels to readable format
                predicted_labels = [target_classes[j] for j in range(len(target_classes)) if preds[i, j] == 1]
                true_labels = [target_classes[j] for j in range(len(target_classes)) if labels[i, j] == 1]
                
                # Convert prefix predictions and labels
                predicted_prefixes = [unique_prefixes[j] for j in range(len(unique_prefixes)) if prefix_preds[i, j] == 1]
                true_prefixes = [unique_prefixes[j] for j in range(len(unique_prefixes)) if prefix_labels[i, j] == 1]
                
                predictions_data.append({
                    'fold': int(fold_num),
                    'sample_idx': i,
                    'text': X_test[i][:200] + '...' if len(X_test[i]) > 200 else X_test[i],  # Truncate for readability
                    'group_id': test_groups[i],
                    'true_labels': str(true_labels),
                    'predicted_labels': str(predicted_labels),
                    'true_prefixes': str(true_prefixes),
                    'predicted_prefixes': str(predicted_prefixes),
                    'num_true_labels': len(true_labels),
                    'num_predicted_labels': len(predicted_labels),
                    'num_true_prefixes': len(true_prefixes),
                    'num_predicted_prefixes': len(predicted_prefixes),
                    'prediction_scores': str(logits[i].tolist())  # Raw logit scores
                })
    
    if predictions_data:
        predictions_df = pd.DataFrame(predictions_data)
        predictions_df.to_csv(os.path.join(output_dir, 'detailed_predictions.csv'), index=False)
        print(f"Detailed predictions saved to {os.path.join(output_dir, 'detailed_predictions.csv')}")
    
    # Also save aggregated predictions (ensemble-like analysis)
    print("Creating aggregated predictions analysis...")
    if len(model_dirs) > 1:
        # Load all predictions
        all_logits = []
        for model_dir in model_dirs:
            fold_num = os.path.basename(model_dir).split('_')[-1]
            predictions_file = os.path.join(cache_dir, f'predictions_fold_{fold_num}.npz')
            
            if os.path.exists(predictions_file):
                data = np.load(predictions_file)
                all_logits.append(data['logits'])
        
        if all_logits:
            # Calculate mean predictions across all folds
            mean_logits = np.mean(all_logits, axis=0)
            mean_preds = (mean_logits > 0).astype(int)
            
            # Calculate ensemble metrics for full codes
            ensemble_micro_f1 = f1_score(y_test, mean_preds, average='micro')
            ensemble_macro_f1 = f1_score(y_test, mean_preds, average='macro')
            
            # Calculate ensemble metrics for prefixes
            ensemble_prefix_info = compute_prefix_metrics(mean_logits, y_test, target_classes, prefix_length=3)
            ensemble_prefix_micro_f1 = ensemble_prefix_info['prefix_micro_f1']
            ensemble_prefix_macro_f1 = ensemble_prefix_info['prefix_macro_f1']
            
            # Create ensemble predictions DataFrame
            ensemble_data = []
            for i in range(len(X_test)):
                predicted_labels = [target_classes[j] for j in range(len(target_classes)) if mean_preds[i, j] == 1]
                true_labels = [target_classes[j] for j in range(len(target_classes)) if y_test[i, j] == 1]
                
                # Prefix predictions
                predicted_prefixes = [ensemble_prefix_info['unique_prefixes'][j] for j in range(len(ensemble_prefix_info['unique_prefixes'])) if ensemble_prefix_info['prefix_predictions'][i, j] == 1]
                true_prefixes = [ensemble_prefix_info['unique_prefixes'][j] for j in range(len(ensemble_prefix_info['unique_prefixes'])) if ensemble_prefix_info['prefix_labels'][i, j] == 1]
                
                ensemble_data.append({
                    'sample_idx': i,
                    'text': X_test[i][:200] + '...' if len(X_test[i]) > 200 else X_test[i],
                    'group_id': test_groups[i],
                    'true_labels': str(true_labels),
                    'ensemble_predicted_labels': str(predicted_labels),
                    'true_prefixes': str(true_prefixes),
                    'ensemble_predicted_prefixes': str(predicted_prefixes),
                    'num_true_labels': len(true_labels),
                    'num_ensemble_predicted_labels': len(predicted_labels),
                    'num_true_prefixes': len(true_prefixes),
                    'num_ensemble_predicted_prefixes': len(predicted_prefixes),
                    'ensemble_scores': str(mean_logits[i].tolist())
                })
            
            ensemble_df = pd.DataFrame(ensemble_data)
            ensemble_df.to_csv(os.path.join(output_dir, 'ensemble_predictions.csv'), index=False)
            
            print(f"Ensemble predictions saved.")
            print(f"Ensemble Full Codes - Micro F1: {ensemble_micro_f1:.4f}, Macro F1: {ensemble_macro_f1:.4f}")
            print(f"Ensemble Prefixes - Micro F1: {ensemble_prefix_micro_f1:.4f}, Macro F1: {ensemble_prefix_macro_f1:.4f}")
    
    # Save summary statistics
    summary_results = {
        'evaluation_config': {
            'target_column': TARGET_COLUMN,
            'target_type': target_type,
            'model_suffix': model_suffix,
            'model_name': model_name,
            'test_set_size': len(X_test),
            'num_labels': num_labels,
            'num_folds_evaluated': len(all_results),
            'evaluation_date': pd.Timestamp.now().isoformat()
        },
        'summary_statistics': statistics,
        'individual_results': all_results,
        'best_fold': {
            'fold_number': int(best_result['fold']),
            'metrics': best_result
        },
        'worst_fold': {
            'fold_number': int(worst_result['fold']),
            'metrics': worst_result
        }
    }
    
    with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w') as f:
        json.dump(summary_results, f, indent=2, default=str)
    
    # Create a formatted report
    report_lines = [
        f"Cross-Validation Model Evaluation Report",
        f"=" * 50,
        f"Target: {target_type} ({TARGET_COLUMN})",
        f"Model: {model_name}",
        f"Test Set Size: {len(X_test)} samples",
        f"Number of Labels: {num_labels}",
        f"Folds Evaluated: {len(all_results)}/10",
        f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "SUMMARY STATISTICS",
        "-" * 20
    ]
    
    for metric in all_metrics:
        stats_data = statistics[metric]
        metric_display = metric.replace('_', ' ').replace('prefix ', '').title()
        if 'prefix' in metric:
            metric_display = f"Prefix {metric_display}"
        report_lines.extend([
            f"{metric_display}:",
            f"  Mean ± Std:     {stats_data['mean']:.4f} ± {stats_data['std']:.4f}",
            f"  95% CI:         [{stats_data['ci_95_lower']:.4f}, {stats_data['ci_95_upper']:.4f}]",
            f"  Range:          [{stats_data['min']:.4f}, {stats_data['max']:.4f}]",
            ""
        ])
    
    report_lines.extend([
        "INDIVIDUAL FOLD RESULTS",
        "-" * 25,
        "Fold | Full Micro F1 | Full Macro F1 | Prefix Micro F1 | Prefix Macro F1"
    ])
    
    for result in all_results:
        report_lines.append(
            f"{result['fold']:4d} | {result['micro_f1']:12.4f} | {result['macro_f1']:12.4f} | "
            f"{result['prefix_micro_f1']:14.4f} | {result['prefix_macro_f1']:14.4f}"
        )
    
    # Save the report
    with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nDetailed results saved to: {output_dir}")
    print("Files created:")
    print(f"  - individual_fold_results.csv: Raw results for each fold")
    print(f"  - detailed_predictions.csv: Per-sample predictions for each fold")
    print(f"  - ensemble_predictions.csv: Ensemble predictions across all folds")
    print(f"  - evaluation_summary.json: Complete statistics and metadata")
    print(f"  - evaluation_report.txt: Human-readable report")
    
    print(f"\nCached data saved to: {cache_dir}")
    print("Cache files:")
    print(f"  - test_dataset.csv: Test set for future runs")
    print(f"  - split_info.json: Dataset split information")
    print(f"  - predictions_fold_*.npz: Model predictions for each fold")
    print("  (Subsequent runs will be much faster using cached data)")
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETED!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()