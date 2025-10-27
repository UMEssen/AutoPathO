import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text

df = pd.read_csv("results/class-based/icdo_results.csv")

# add font size to 14
plt.rcParams.update({'font.size': 14})
# Set the style of seaborn
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Sort the dataframe by support in descending order and take the top 10
top10_by_support = df.sort_values('support', ascending=False).head(10)

# Also create a scatter plot to show the relationship between support and F1-score
plt.figure(figsize=(10, 6))
scatter_plot = sns.scatterplot(x='support', y='f1_score', data=top10_by_support, 
                              s=100, alpha=0.7)

# Create a list to store the text objects for adjustText
texts = []
# Add label annotations to each point
for i, row in enumerate(top10_by_support.itertuples()):
    texts.append(plt.text(row.support, row.f1_score, row.ICD_code))

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
#plt.title('e) Exact ICD-O Code Performance (Llama 3.3 70B) in Relation to Support', fontsize=16)
plt.title('f) Three Character ICD-O Code Performance (Llama 3.3 70B) in Relation to Support', fontsize=16)

plt.xlabel('Support (Frequency)', fontsize=14)
plt.ylabel('F1-Score', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()


plt.savefig('plots/support_vs_f1_scatter_icdo_three_chars.png', dpi=600, bbox_inches='tight')