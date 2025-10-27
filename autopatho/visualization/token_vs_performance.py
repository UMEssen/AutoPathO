import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# ICD-10 Complete Instance-Based Metrics
save_dir = "plots/figure7a_token_vs_f1_histogram_icdot_complete.png"
df_llama = pd.read_csv("results/instance-based/icd10_complete_instance_based_metrics_Llama-70B.csv")
df_deepseek_70b = pd.read_csv("results/instance-based/icd10_complete_instance_based_metrics_Deepseek-70B.csv")
df_qwen3 = pd.read_csv("results/instance-based/icd10_complete_instance_based_metrics_Qwen3.csv")
df_deepseek_8b = pd.read_csv("results/instance-based/icd10_complete_instance_based_metrics_Deepseek-8B.csv")
df_gemma3 = pd.read_csv("results/instance-based/icd10_complete_instance_based_metrics_Gemma3.csv")
bin_size = 200


def tokenize(text, tokenizer):
    encoded = tokenizer.encode_plus(text, return_tensors='pt')
    return len(encoded['input_ids'][0])

tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-large", device='cuda:2')
# Calculate token length for each dataframe using transformers tokenizer
df_llama["token"] = [tokenize(row["Befunde"], tokenizer) for i, row in df_llama.iterrows()]
df_deepseek_70b["token"] = [tokenize(row["Befunde"], tokenizer) for i, row in df_deepseek_70b.iterrows()]
df_qwen3["token"] = [tokenize(row["Befunde"], tokenizer) for i, row in df_qwen3.iterrows()]
df_deepseek_8b["token"] = [tokenize(row["Befunde"], tokenizer) for i, row in df_deepseek_8b.iterrows()]
df_gemma3["token"] = [tokenize(row["Befunde"], tokenizer) for i, row in df_gemma3.iterrows()]

f1_col = 'Instance_F1'

# Improve font size and style
plt.rcParams.update({'font.size': 14})
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Create bins with even, intuitive intervals
min_token = df_llama['token'].min()
max_token = df_llama['token'].max()

# Round the min and max to nice values for token counts
min_token_rounded = int(np.floor(min_token / bin_size) * bin_size)  # Round down to nearest bin_size
max_token_rounded = int(np.ceil(max_token / bin_size) * bin_size)   # Round up to nearest bin_size

# Create intuitive bin edges (e.g., 50, 100, 150, etc. for tokens)

bin_edges = list(range(min_token_rounded, max_token_rounded + bin_size, bin_size))

# Ensure no data points fall outside the bins
if min_token < bin_edges[0]:
    bin_edges[0] = int(min_token)
if max_token > bin_edges[-1]:
    bin_edges[-1] = int(max_token) + 1

# Use cut to assign tokens to bins
df_llama['bin_idx'] = pd.cut(df_llama['token'], bins=bin_edges, labels=False, include_lowest=True)
df_llama['token_bin'] = pd.cut(df_llama['token'], bins=bin_edges, include_lowest=True)

# Calculate mean F1-score per bin
# Filter out NaN values that might occur if some bins are empty
binned_performance = df_llama.groupby('bin_idx')[f1_col].mean().reset_index()
binned_performance = binned_performance.dropna()

# Function to process dataframes and compute binned performance
def compute_binned_performance(df, bin_edges, f1_col):
    # Use cut to assign tokens to bins
    df['bin_idx'] = pd.cut(df['token'], bins=bin_edges, labels=False, include_lowest=True)
    df['token_bin'] = pd.cut(df['token'], bins=bin_edges, include_lowest=True)
    
    # Calculate mean F1-score per bin and filter out NaNs
    binned_performance = df.groupby('bin_idx')[f1_col].mean().reset_index()
    binned_performance = binned_performance.dropna()
    
    # Create bin labels
    bin_labels = [f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(len(bin_edges)-1)]
    binned_performance['bin_label'] = binned_performance['bin_idx'].apply(
        lambda x: bin_labels[int(x)] if pd.notna(x) and int(x) < len(bin_labels) else "N/A"
    )
    
    # Convert F1 scores to percentages
    binned_performance[f'{f1_col}_percentage'] = binned_performance[f1_col] * 100
    
    return binned_performance.sort_values('bin_idx')

# Create bin labels that show the range (lower bound - upper bound)
bin_labels = [f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(len(bin_edges)-1)]

# Process all dataframes
binned_llama = compute_binned_performance(df_llama, bin_edges, f1_col)
binned_deepseek_70b = compute_binned_performance(df_deepseek_70b, bin_edges, f1_col)
binned_qwen3 = compute_binned_performance(df_qwen3, bin_edges, f1_col)
binned_deepseek_8b = compute_binned_performance(df_deepseek_8b, bin_edges, f1_col)
binned_gemma3 = compute_binned_performance(df_gemma3, bin_edges, f1_col)

plt.figure(figsize=(8, 5))

# Create line plots with markers for each model
plt.plot(binned_llama['bin_label'], binned_llama[f'{f1_col}_percentage'], 
         marker='o', markersize=8, linewidth=2, label='Llama-3.3-70B-Instruct')

plt.plot(binned_deepseek_70b['bin_label'], binned_deepseek_70b[f'{f1_col}_percentage'], 
         marker='s', markersize=8, linewidth=2, label='DeepSeek-R1-Distill-Llama-70B')

plt.plot(binned_qwen3['bin_label'], binned_qwen3[f'{f1_col}_percentage'], 
         marker='^', markersize=8, linewidth=2, label='Qwen3-235B-A22B')

plt.plot(binned_deepseek_8b['bin_label'], binned_deepseek_8b[f'{f1_col}_percentage'], 
         marker='D', markersize=8, linewidth=2, label='DeepSeek-R1-Distill-Llama-8B')

plt.plot(binned_gemma3['bin_label'], binned_gemma3[f'{f1_col}_percentage'], 
         marker='*', markersize=10, linewidth=2, label='Gemma-3-12B-it')

# Show only a subset of x-tick labels if there are too many
if len(bin_labels) > 10:
    step = max(1, len(bin_labels) // 10)
    plt.xticks(range(0, len(bin_labels), step), 
               [bin_labels[i] for i in range(0, len(bin_labels), step)], 
               rotation=45, ha='right', fontsize=16)
else:
    plt.xticks(rotation=45, ha='right', fontsize=16)

# Add legend with model names
plt.legend(loc='lower center', bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False, fontsize=10)

# Format y-axis as percentages
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))

plt.xlabel('Text Length (Tokens)', fontsize=16)
plt.ylabel(f'Average {f1_col.replace("_", "-")} (%)', fontsize=16)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()

plt.savefig(save_dir, dpi=600, bbox_inches='tight')