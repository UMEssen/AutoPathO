import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

initial_csv_path = "data/initial_dataset.csv"
cases = pd.read_csv(initial_csv_path)
icd_10_cases = cases[cases['GT_ICD-10'].notna()]
# Before counting the frequency of ICD-10 codes, we need to break down multiple ICD-10 codes
# in the same cell into separate rows
icd_10_cases = icd_10_cases['GT_ICD-10'].str.split(',').explode()
# Replace any '[' or ']' characters in the ICD-10 codes
icd_10_cases = icd_10_cases.str.replace('[', '', regex=False)
icd_10_cases = icd_10_cases.str.replace(']', '', regex=False)
icd_10_cases = icd_10_cases.str.replace('\'', '', regex=False)
# Remove any leading or trailing whitespace
icd_10_cases = icd_10_cases.str.strip()
icd_10_cases = pd.DataFrame(icd_10_cases, columns=['GT_ICD-10'])
# Remove any empty strings from the ICD-10 codes
icd_10_cases = icd_10_cases[icd_10_cases['GT_ICD-10'] != '']
# Count the the frequency of each ICD-10 code
icd_counts = icd_10_cases['GT_ICD-10'].value_counts()

# Set style
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# Create a figure with sufficient height
plt.figure(figsize=(8, 6))

# Plot top n most frequent codes (adjust as needed)
top_n = 10
top_codes = icd_counts.head(top_n)

# Horizontal bar chart
ax = sns.barplot(x=top_codes.values, y=top_codes.index, palette='viridis')

# Add value labels to the bars
for i, v in enumerate(top_codes.values):
    ax.text(v + 0.5, i, str(v), va='center')

plt.title(f'a) Top {top_n} Most Frequent ICD-10 Codes')
plt.xlabel('Frequency')
plt.ylabel('ICD-10 Code')
plt.tight_layout()

# Save the figure
plt.savefig('plots/top_icd10_codes.png', dpi=300, bbox_inches='tight')
plt.show()

icd_o_cases = cases[cases['GT_ICD-O'].notna()]
# Before counting the frequency of ICD-O codes, we need to break down multiple ICD-O codes
# in the same cell into separate rows
icd_o_cases = icd_o_cases['GT_ICD-O'].str.split(',').explode()
# Replace any '[' or ']' characters in the ICD-O codes
icd_o_cases = icd_o_cases.str.replace('[', '', regex=False)
icd_o_cases = icd_o_cases.str.replace(']', '', regex=False)
icd_o_cases = icd_o_cases.str.replace('\'', '', regex=False)
# Remove any leading or trailing whitespace
icd_o_cases = icd_o_cases.str.strip()
icd_o_cases = pd.DataFrame(icd_o_cases, columns=['GT_ICD-O'])
# Remove any empty strings from the ICD-O codes
icd_o_cases = icd_o_cases[icd_o_cases['GT_ICD-O'] != '']
# Count the the frequency of each ICD-O code
icd_o_counts = icd_o_cases['GT_ICD-O'].value_counts()
# Create a figure with sufficient height

plt.figure(figsize=(8, 6))
# Plot top n most frequent codes (adjust as needed)
top_n = 10
top_codes = icd_o_counts.head(top_n)
# Horizontal bar chart
ax = sns.barplot(x=top_codes.values, y=top_codes.index, palette='viridis')
# Add value labels to the bars
for i, v in enumerate(top_codes.values):
    ax.text(v + 0.5, i, str(v), va='center')
plt.title(f'b) Top {top_n} Most Frequent ICD-O Codes')
plt.xlabel('Frequency')
plt.ylabel('ICD-O Code')
plt.tight_layout()
# Save the figure
plt.savefig('plots/top_icdo_codes.png', dpi=300, bbox_inches='tight')
plt.show()