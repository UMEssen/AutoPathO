import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set style
plt.style.use('ggplot')
sns.set_palette("viridis")

# Create DataFrame from the results
models = ['Llama3-70B', 'DeepSeek-R1-Distill-Llama-70B', 'Gemma3-12B', 'DeepSeek-R1-Distill-Llama-8B']

# Complete ICD Code metrics
complete_metrics = pd.DataFrame({
    'Model': models,
    'Accuracy': [34.43, 46.58, 5.37, 8.86],
    'Precision': [54.14, 64.14, 10.75, 16.81],
    'Recall': [48.60, 62.98, 9.70, 15.79],
    'F1 Score': [51.22, 63.56, 10.19, 16.28]
})

# First three characters ICD Code metrics
first_three_metrics = pd.DataFrame({
    'Model': models,
    'Accuracy': [85.99, 77.38, 14.91, 35.99],
    'Precision': [92.48, 87.10, 25.96, 52.81],
    'Recall': [92.46, 87.40, 25.94, 53.05],
    'F1 Score': [92.47, 87.25, 25.95, 52.93]
})

# Complete ICD confusion matrices
complete_confusion = [
    np.array([[7821, 6626], [8271, 0]]),
    np.array([[4318, 2414], [2538, 0]]),
    np.array([[1273, 10573], [11856, 0]]),
    np.array([[714, 3534], [3807, 0]])
]

# First three characters confusion matrices
first_three_confusion = [
    np.array([[13272, 1079], [1083, 0]]),
    np.array([[5672, 840], [818, 0]]),
    np.array([[3075, 8771], [8777, 0]]),
    np.array([[2234, 1996], [1977, 0]])
]

# 1. Complete ICD Code Metrics Plot
plt.figure(figsize=(12, 8))
df_melted = pd.melt(complete_metrics, id_vars=['Model'], var_name='Metric', value_name='Value')
ax = sns.barplot(x='Model', y='Value', hue='Metric', data=df_melted)

# Customize
plt.xlabel('')
plt.ylabel('Percentage (%)', fontsize=12)
plt.ylim(0, 70)

# Add value labels on bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', fontsize=12)

plt.legend(title='', fontsize=12, loc='upper right')
plt.xticks(rotation=30, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('plots/complete_icd_metrics.png', dpi=600, bbox_inches='tight')
plt.close()

# 2. First Three Characters ICD Code Metrics Plot
plt.figure(figsize=(12, 8))
df_melted = pd.melt(first_three_metrics, id_vars=['Model'], var_name='Metric', value_name='Value')
ax = sns.barplot(x='Model', y='Value', hue='Metric', data=df_melted)

# Customize
plt.xlabel('')
plt.ylabel('Percentage (%)', fontsize=12)
plt.ylim(0, 100)

# Add value labels on bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', fontsize=10)

plt.legend(title='', fontsize=12, loc='upper right')
plt.xticks(rotation=30, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('plots/first_three_icd_metrics.png', dpi=600, bbox_inches='tight')
plt.close()

# 3. Complete ICD Code Confusion Matrices Plot
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Flatten the axes array for easy iteration
axes = axes.flatten()

for i, (cm, model) in enumerate(zip(complete_confusion, models)):
    # Calculate percentages for text annotation (but keep original values for display)
    row_sums = cm.sum(axis=1)
    cm_percentages = np.zeros_like(cm, dtype=float)
    for row in range(cm.shape[0]):
        if row_sums[row] > 0:
            cm_percentages[row, :] = cm[row, :] / row_sums[row] * 100
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True, ax=axes[i], annot_kws={"size": 12})
    
    # Add percentages in brackets
    for j in range(cm.shape[0]):
        for k in range(cm.shape[1]):
            if cm[j, k] > 0:
                text = axes[i].texts[j * cm.shape[1] + k]
                current_text = text.get_text()
                text.set_text(f"{current_text}")
    
    axes[i].set_title(f"{model}", fontsize=14)
    axes[i].set_xlabel('Predicted', fontsize=12)
    axes[i].set_ylabel('Actual', fontsize=12)
    axes[i].set_xticklabels(['True', 'False'])
    axes[i].set_yticklabels(['True', 'False'])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('plots/complete_icd_confusion_matrices.png', dpi=600, bbox_inches='tight')
plt.close()

# 4. First Three Characters ICD Code Confusion Matrices Plot
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Flatten the axes array for easy iteration
axes = axes.flatten()

for i, (cm, model) in enumerate(zip(first_three_confusion, models)):
    # Calculate percentages for text annotation
    row_sums = cm.sum(axis=1)
    cm_percentages = np.zeros_like(cm, dtype=float)
    for row in range(cm.shape[0]):
        if row_sums[row] > 0:
            cm_percentages[row, :] = cm[row, :] / row_sums[row] * 100
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True, ax=axes[i], annot_kws={"size": 12})
    
    # Add percentages in brackets
    for j in range(cm.shape[0]):
        for k in range(cm.shape[1]):
            if cm[j, k] > 0:
                text = axes[i].texts[j * cm.shape[1] + k]
                current_text = text.get_text()
                text.set_text(f"{current_text}")
    
    axes[i].set_title(f"{model}", fontsize=14)
    axes[i].set_xlabel('Predicted', fontsize=12)
    axes[i].set_ylabel('Actual', fontsize=12)
    axes[i].set_xticklabels(['True', 'False'])
    axes[i].set_yticklabels(['True', 'False'])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('plots/first_three_icd_confusion_matrices.png', dpi=600, bbox_inches='tight')
plt.close()