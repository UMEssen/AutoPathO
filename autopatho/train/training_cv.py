import os
import ast
import json
import torch
import wandb
import pickle
import numpy as np
import pandas as pd
from typing import Dict
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import f1_score, recall_score, precision_score
from transformers import Trainer, AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification

# Configuration parameters
TARGET_COLUMN = "GT_ICD-10"  # Change this to "GT_ICD-O" to train on ICD-O codes
model_name = 'EuroBERT/EuroBERT-610m'

# Determine project name and model suffix based on target column
if TARGET_COLUMN == "GT_ICD-10":
    project_name = "icd10_prediction_cv"
    model_suffix = "icd10"
    target_type = "ICD-10"
    df = pd.read_csv('data/icd-10/common_reports.csv')
elif TARGET_COLUMN == "GT_ICD-O":
    project_name = "icdo_prediction_cv"
    model_suffix = "icdo"
    target_type = "ICD-O"
    df = pd.read_csv('data/icd-o/common_reports.csv')
else:
    raise ValueError(f"Unsupported target column: {TARGET_COLUMN}")

print(f"Training configuration:")
print(f"  Target column: {TARGET_COLUMN}")
print(f"  Target type: {target_type}")
print(f"  Model suffix: {model_suffix}")
print(f"  W&B project: {project_name}")

wandb.login()
wandb.init(
    project=project_name,
    name=f"{model_name}_{model_suffix}_10fold_cv",
)

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
print(f"First few {target_type} classes: {target_classes[:5]}")
print(f"Target column '{TARGET_COLUMN}' processed successfully")

# Create initial train-test split based on 'pid' column to ensure grouping
# This will be our final holdout test set
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['pid']))

# Extract holdout test set
X_test_final = df['Befunde'].values[test_idx]
y_test_final = target_encoded[test_idx]
test_groups_final = df['per'].values[test_idx]

# Extract training set for cross-validation
X_train_full = df['Befunde'].values[train_idx]
y_train_full = target_encoded[train_idx]
train_groups = df['per'].values[train_idx]

print(f"Total training samples for CV: {len(X_train_full)}")
print(f"Final holdout test samples: {len(X_test_final)}")
print(f"Unique groups in training set: {len(np.unique(train_groups))}")

# Create a dataset class compatible with the Trainer API
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

# Define custom metrics computation
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
    
    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall
    }

def train_fold(fold_num: int, train_texts: np.ndarray, train_labels: np.ndarray, 
               val_texts: np.ndarray, val_labels: np.ndarray, 
               tokenizer, num_labels: int) -> Dict:
    """Train a single fold and return metrics"""
    
    print(f"\n=== Training Fold {fold_num + 1}/10 ===")
    print(f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}")
    
    # Create model for this fold
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        problem_type="multi_label_classification",
        trust_remote_code=True,
    )
    
    # Create datasets
    train_dataset = MultiLabelDataset(train_texts, train_labels, tokenizer)
    val_dataset = MultiLabelDataset(val_texts, val_labels, tokenizer)
    
    # Define training arguments for this fold
    output_dir = f'./models/eurobert_{model_suffix}_classifier_fold_{fold_num + 1}'
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-05,
        lr_scheduler_type="linear",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        save_total_limit=1,
        report_to="wandb",
        fp16=torch.cuda.is_available(),
        logging_dir=f'./logs/fold_{fold_num + 1}',
        logging_steps=10,
        warmup_ratio=0.1,
        optim='adafactor',
        run_name=f"fold_{fold_num + 1}",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    val_results = trainer.evaluate()
    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model for fold {fold_num + 1} saved to {output_dir}")
    
    wandb.log({
        f"fold_{fold_num + 1}_micro_f1": val_results['eval_micro_f1'],
        f"fold_{fold_num + 1}_macro_f1": val_results['eval_macro_f1'],
        f"fold_{fold_num + 1}_micro_precision": val_results['eval_micro_precision'],
        f"fold_{fold_num + 1}_micro_recall": val_results['eval_micro_recall'],
        "fold": fold_num + 1,
        "target_column": TARGET_COLUMN
    })
    
    print(f"Fold {fold_num + 1} Results ({target_type}):")
    print(f"  Micro F1: {val_results['eval_micro_f1']:.4f}")
    print(f"  Macro F1: {val_results['eval_macro_f1']:.4f}")
    print(f"  Micro Precision: {val_results['eval_micro_precision']:.4f}")
    print(f"  Micro Recall: {val_results['eval_micro_recall']:.4f}")
    
    return val_results, trainer.model, output_dir

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Set up 10-fold cross-validation with GroupKFold
n_folds = 10
group_kfold = GroupKFold(n_splits=n_folds)

# Store results from each fold
cv_results = []
fold_models = []

# Check for existing models and resume training
def check_existing_folds():
    """Check which folds have already been completed"""
    completed_folds = []
    for fold_num in range(n_folds):
        fold_dir = f'./models/eurobert_{model_suffix}_classifier_fold_{fold_num + 1}'
        config_path = os.path.join(fold_dir, 'config.json')
        model_path = os.path.join(fold_dir, 'model.safetensors')
        tokenizer_path = os.path.join(fold_dir, 'tokenizer.json')
        
        if os.path.exists(config_path) and os.path.exists(model_path) and os.path.exists(tokenizer_path):
            completed_folds.append(fold_num)
    
    return completed_folds

def load_existing_fold_results(fold_num):
    """Load results from an existing fold"""
    fold_dir = f'./models/eurobert_{model_suffix}_classifier_fold_{fold_num + 1}'
    
    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
        fold_dir,
        num_labels=num_labels,
        problem_type="multi_label_classification",
        trust_remote_code=True,
    )
    
    # Try to load training results if available
    trainer_state_path = os.path.join(fold_dir, 'trainer_state.json')
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path, 'r') as f:
            trainer_state = json.load(f)
        
        # Extract the best metrics from training history
        if 'log_history' in trainer_state:
            eval_results = [entry for entry in trainer_state['log_history'] if 'eval_micro_f1' in entry]
            if eval_results:
                best_result = max(eval_results, key=lambda x: x.get('eval_micro_f1', 0))
                return {
                    'eval_micro_f1': best_result.get('eval_micro_f1', 0),
                    'eval_macro_f1': best_result.get('eval_macro_f1', 0),
                    'eval_micro_precision': best_result.get('eval_micro_precision', 0),
                    'eval_micro_recall': best_result.get('eval_micro_recall', 0),
                }, model, fold_dir
    
    # If we can't load training results, we'll need to re-evaluate
    print(f"Warning: Could not load training results for fold {fold_num + 1}, using placeholder values")
    return {
        'eval_micro_f1': 0.0,
        'eval_macro_f1': 0.0,
        'eval_micro_precision': 0.0,
        'eval_micro_recall': 0.0,
    }, model, fold_dir

completed_folds = check_existing_folds()
if completed_folds:
    print(f"\nFound {len(completed_folds)} completed folds: {[f+1 for f in completed_folds]}")
    
    # Load existing results
    for fold_num in completed_folds:
        fold_results, fold_model, fold_dir = load_existing_fold_results(fold_num)
        cv_results.append(fold_results)
        fold_models.append((fold_results['eval_micro_f1'], fold_model, fold_num, fold_dir))
        print(f"  Loaded Fold {fold_num + 1}: Micro F1 = {fold_results['eval_micro_f1']:.4f}")
    
    remaining_folds = [i for i in range(n_folds) if i not in completed_folds]
    print(f"Remaining folds to train: {[f+1 for f in remaining_folds]}")
else:
    print(f"\nNo completed folds found. Starting fresh {n_folds}-fold cross-validation...")
    remaining_folds = list(range(n_folds))

# Perform cross-validation only for remaining folds
fold_splits = list(enumerate(group_kfold.split(X_train_full, y_train_full, train_groups)))

for fold_num, (train_fold_idx, val_fold_idx) in fold_splits:
    
    # Skip if this fold is already completed
    if fold_num in completed_folds:
        print(f"\nSkipping Fold {fold_num + 1} (already completed)")
        continue
    # Skip if this fold is already completed
    if fold_num in completed_folds:
        print(f"\nSkipping Fold {fold_num + 1} (already completed)")
        continue
    
    print(f"\n=== Starting Fold {fold_num + 1}/{n_folds} ===")
    
    # Extract fold data
    X_train_fold = X_train_full[train_fold_idx]
    y_train_fold = y_train_full[train_fold_idx]
    X_val_fold = X_train_full[val_fold_idx]
    y_val_fold = y_train_full[val_fold_idx]
    
    try:
        # Train fold
        fold_results, fold_model, fold_output_dir = train_fold(
            fold_num, X_train_fold, y_train_fold, X_val_fold, y_val_fold,
            tokenizer, num_labels
        )
        
        cv_results.append(fold_results)
        fold_models.append((fold_results['eval_micro_f1'], fold_model, fold_num, fold_output_dir))
        
        print(f"Fold {fold_num + 1} completed successfully!")
        
    except Exception as e:
        print(f"\nError training Fold {fold_num + 1}: {str(e)}")
        print("Stopping training. You can resume by running the script again.")
        
        # Save partial results if we have any
        if cv_results:
            print(f"Saving partial results for {len(cv_results)} completed folds...")
            
            # Create partial summary
            partial_summary_dir = f'./models/eurobert_{model_suffix}_classifier_cv_partial'
            os.makedirs(partial_summary_dir, exist_ok=True)
            
            partial_results = {
                'completed_folds': len(cv_results),
                'total_folds': n_folds,
                'fold_results': cv_results,
                'error_fold': fold_num + 1,
                'error_message': str(e)
            }
            
            with open(os.path.join(partial_summary_dir, 'partial_results.json'), 'w') as f:
                json.dump(partial_results, f, indent=2)
            
            print(f"Partial results saved to {partial_summary_dir}")
        
        # Re-raise the exception to stop execution
        raise e

print(f"\nAll folds completed! Total models trained: {len(cv_results)}")

# Calculate cross-validation statistics
# Handle case where some folds might need re-evaluation
folds_needing_evaluation = []
for i, (score, model, fold_idx, model_dir) in enumerate(fold_models):
    if score == 0.0:  # Placeholder value indicating missing metrics
        folds_needing_evaluation.append((i, fold_idx, model, model_dir))

if folds_needing_evaluation:
    print(f"\nRe-evaluating {len(folds_needing_evaluation)} folds with missing metrics...")
    
    for list_idx, fold_idx, model, model_dir in folds_needing_evaluation:
        print(f"Re-evaluating Fold {fold_idx + 1}...")
        
        # Get the validation data for this fold
        fold_splits_list = list(group_kfold.split(X_train_full, y_train_full, train_groups))
        _, val_fold_idx = fold_splits_list[fold_idx]
        
        X_val_fold = X_train_full[val_fold_idx]
        y_val_fold = y_train_full[val_fold_idx]
        
        # Create validation dataset
        val_dataset = MultiLabelDataset(X_val_fold, y_val_fold, tokenizer)
        
        # Create temporary trainer for evaluation
        eval_args = TrainingArguments(
            output_dir='./temp_eval',
            per_device_eval_batch_size=4,
            fp16=torch.cuda.is_available(),
        )
        
        eval_trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        # Evaluate and update results
        eval_results = eval_trainer.evaluate()
        cv_results[list_idx] = eval_results
        fold_models[list_idx] = (eval_results['eval_micro_f1'], model, fold_idx, model_dir)
        
        print(f"  Updated Fold {fold_idx + 1}: Micro F1 = {eval_results['eval_micro_f1']:.4f}")

metrics = ['eval_micro_f1', 'eval_macro_f1', 'eval_micro_precision', 'eval_micro_recall']
cv_stats = {}

print(f"\n{'='*50}")
print(f"CROSS-VALIDATION RESULTS - {target_type}")
print(f"{'='*50}")

for metric in metrics:
    values = [result[metric] for result in cv_results]
    cv_stats[metric] = {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'values': values
    }
    
    metric_name = metric.replace('eval_', '').replace('_', ' ').title()
    print(f"{metric_name}:")
    print(f"  Mean: {cv_stats[metric]['mean']:.4f} ± {cv_stats[metric]['std']:.4f}")
    print(f"  Range: [{cv_stats[metric]['min']:.4f}, {cv_stats[metric]['max']:.4f}]")
    print()

# Log CV summary to wandb
wandb.log({
    "cv_micro_f1_mean": cv_stats['eval_micro_f1']['mean'],
    "cv_micro_f1_std": cv_stats['eval_micro_f1']['std'],
    "cv_macro_f1_mean": cv_stats['eval_macro_f1']['mean'],
    "cv_macro_f1_std": cv_stats['eval_macro_f1']['std'],
    "cv_micro_precision_mean": cv_stats['eval_micro_precision']['mean'],
    "cv_micro_precision_std": cv_stats['eval_micro_precision']['std'],
    "cv_micro_recall_mean": cv_stats['eval_micro_recall']['mean'],
    "cv_micro_recall_std": cv_stats['eval_micro_recall']['std'],
    "target_column": TARGET_COLUMN,
    "num_labels": num_labels
})

# Select best model based on validation micro F1
best_score, best_model, best_fold, best_model_dir = max(fold_models, key=lambda x: x[0])
print(f"Best model from Fold {best_fold + 1} with Micro F1: {best_score:.4f}")

# Save all fold models info and create a summary
all_models_info = []
for score, model, fold_idx, model_dir in fold_models:
    model_info = {
        'fold': fold_idx + 1,
        'micro_f1': float(score),
        'model_path': model_dir,
        'is_best': fold_idx == best_fold
    }
    all_models_info.append(model_info)

# Save MultiLabelBinarizer to each fold directory
for _, _, _, fold_dir in fold_models:
    with open(os.path.join(fold_dir, 'multilabelbinarizer.pkl'), 'wb') as f:
        pickle.dump(mlb, f)

print(f"\nAll {n_folds} models saved:")
for model_info in all_models_info:
    status = " (BEST)" if model_info['is_best'] else ""
    print(f"  Fold {model_info['fold']}: {model_info['model_path']} - F1: {model_info['micro_f1']:.4f}{status}")

# Create a summary directory
summary_dir = f'./models/eurobert_{model_suffix}_classifier_cv_summary'
os.makedirs(summary_dir, exist_ok=True)

# Save CV results
cv_results_summary = {
    'cv_statistics': cv_stats,
    'fold_results': cv_results,
    'all_models_info': all_models_info,
    'best_fold': int(best_fold + 1),
    'best_score': float(best_score),
    'best_model_path': best_model_dir,
    'model_name': model_name,
    'target_column': TARGET_COLUMN,
    'target_type': target_type,
    'model_suffix': model_suffix,
    'n_folds': n_folds
}

with open(os.path.join(summary_dir, 'cv_results.json'), 'w') as f:
    json.dump(cv_results_summary, f, indent=2)

print(f"\nCV summary and results saved to {summary_dir}")

# Optional: Evaluate on final holdout test set
print(f"\n{'='*50}")
print(f"FINAL HOLDOUT TEST EVALUATION - {target_type}")
print(f"{'='*50}")

test_dataset_final = MultiLabelDataset(X_test_final, y_test_final, tokenizer)

# Create a temporary trainer for final evaluation
final_training_args = TrainingArguments(
    output_dir='./temp_final_eval',
    per_device_eval_batch_size=4,
    fp16=torch.cuda.is_available(),
)

final_trainer = Trainer(
    model=best_model,
    args=final_training_args,
    eval_dataset=test_dataset_final,
    compute_metrics=compute_metrics,
)

final_results = final_trainer.evaluate()

print(f"Final Holdout Test Results ({target_type}):")
print(f"  Micro F1: {final_results['eval_micro_f1']:.4f}")
print(f"  Macro F1: {final_results['eval_macro_f1']:.4f}")
print(f"  Micro Precision: {final_results['eval_micro_precision']:.4f}")
print(f"  Micro Recall: {final_results['eval_micro_recall']:.4f}")

# Log final results to wandb
wandb.log({
    "final_test_micro_f1": final_results['eval_micro_f1'],
    "final_test_macro_f1": final_results['eval_macro_f1'],
    "final_test_micro_precision": final_results['eval_micro_precision'],
    "final_test_micro_recall": final_results['eval_micro_recall'],
    "target_column": TARGET_COLUMN
})

# Save final test results
final_results_summary = {
    'final_test_results': final_results,
    'cv_vs_final_comparison': {
        'cv_micro_f1_mean': cv_stats['eval_micro_f1']['mean'],
        'final_micro_f1': final_results['eval_micro_f1'],
        'difference': final_results['eval_micro_f1'] - cv_stats['eval_micro_f1']['mean']
    }
}

with open(os.path.join(summary_dir, 'final_test_results.json'), 'w') as f:
    json.dump(final_results_summary, f, indent=2)

print(f"\nFinal test results saved to {summary_dir}")
print(f"Cross-validation completed! Check wandb for detailed logs.")
print(f"All models saved in individual fold directories.")
print(f"Summary and results available in {summary_dir}")
print(f"Training target: {TARGET_COLUMN} ({target_type} codes)")

wandb.finish()