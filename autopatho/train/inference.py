import os
import ast
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, recall_score, precision_score

# Path to the saved model
model_path = './models/eurobert_icd10_classifier'

# Load the test data
df = pd.read_csv('data/test_dataset.csv')

# Convert ICD-10 codes to lists if they're in string format
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

df['GT_ICD-10'] = df['GT_ICD-10'].apply(convert_to_list)

# Load the MultiLabelBinarizer used during training
with open(os.path.join(model_path, 'multilabelbinarizer.pkl'), 'rb') as f:
    mlb = pickle.load(f)

# Encode the ICD-10 codes
icd10_encoded = mlb.transform(df['GT_ICD-10'])
icd10_classes = mlb.classes_

# Create the same train-test split as in training to get the test set

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['pid']))

X_test = df['Befunde_filtered'].values[test_idx]
y_test = icd10_encoded[test_idx]
test_ids = df['id'].values[test_idx]  # Keep track of IDs for error analysis

print(f"Number of test samples: {len(X_test)}")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    problem_type="multi_label_classification",
    trust_remote_code=True,
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Dataset class for evaluation
class EvalDataset(Dataset):
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

# Create evaluation dataset and dataloader
eval_dataset = EvalDataset(X_test, y_test, tokenizer)
eval_dataloader = DataLoader(eval_dataset, batch_size=4)

# Evaluation function
def evaluate():
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()
            
            # Convert logits to predictions (binary)
            preds = (logits > 0).astype(int)
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    micro_f1 = f1_score(all_labels, all_preds, average='micro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    micro_precision = precision_score(all_labels, all_preds, average='micro')
    micro_recall = recall_score(all_labels, all_preds, average='micro')
    
    # Per-class metrics
    class_metrics = {}
    for i, class_name in enumerate(icd10_classes):
        class_metrics[class_name] = {
            'precision': precision_score(all_labels[:, i], all_preds[:, i], zero_division=0),
            'recall': recall_score(all_labels[:, i], all_preds[:, i], zero_division=0),
            'f1': f1_score(all_labels[:, i], all_preds[:, i], zero_division=0),
            'support': np.sum(all_labels[:, i])
        }
    
    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'per_class_metrics': class_metrics,
        'predictions': all_preds,
        'labels': all_labels
    }

# Run evaluation
print("Starting evaluation...")
results = evaluate()

# Print overall results
print(f"\nOverall Results:")
print(f"Micro F1: {results['micro_f1']:.4f}")
print(f"Macro F1: {results['macro_f1']:.4f}")
print(f"Micro Precision: {results['micro_precision']:.4f}")
print(f"Micro Recall: {results['micro_recall']:.4f}")

# Print per-class results for top 10 most frequent classes
print("\nTop 10 most frequent classes:")
sorted_classes = sorted([(c, results['per_class_metrics'][c]['support']) 
                         for c in icd10_classes], key=lambda x: x[1], reverse=True)

for class_name, support in sorted_classes[:10]:
    metrics = results['per_class_metrics'][class_name]
    print(f"{class_name} (support: {support}): Precision={metrics['precision']:.4f}, "
          f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

true_labels = [mlb.inverse_transform(label.reshape(1, -1))[0] for label in results['labels']]
predicted_labels = [mlb.inverse_transform(pred.reshape(1, -1))[0] for pred in results['predictions']]

# Error analysis: Save predictions and true labels for further analysis
results_df = pd.DataFrame({
    'id': test_ids,
    'text': X_test,
    'true_labels': true_labels,
    'predicted_labels': predicted_labels,
})

# Save predictions to CSV
results_df.to_csv('results/evaluation_results.csv', index=False)
print("\nDetailed predictions saved to evaluation_results.csv")

# Create a DataFrame with all original columns for the test set
predictions_df = df.iloc[test_idx].copy()

# Add true and predicted labels
predictions_df['true_labels_decoded'] = [mlb.inverse_transform(label.reshape(1, -1))[0] for label in results['labels']]
predictions_df['predicted_labels_decoded'] = [mlb.inverse_transform(pred.reshape(1, -1))[0] for pred in results['predictions']]
predictions_df.to_csv('results/evaluation_results_with_all_columns.csv', index=False)
print("\nDetailed predictions with all columns saved to results/evaluation_results_with_all_columns.csv")