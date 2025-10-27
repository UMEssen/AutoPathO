import re
import pandas as pd

# Load the dataset
df = pd.read_csv('data/patho_dataframe.csv')

# Check if title is nan
df = df[df["title"].notna()]

# Define a function to extract the identification string for grouping
def extract_id(title):
    match = re.search(r'\b[C|D|E|F|N|R]\d{2,5}/\d{2}\b', title)
    return match.group(0) if match else None

# Apply the function to create a new column for grouping
df['id'] = df['title'].apply(extract_id)
df = df[df["id"].notna()]
df["Befunde_count"] = 0

# Define custom aggregation functions
agg_funcs = {col: 'last' for col in df.columns if col != 'Befunde'}
agg_funcs['Befunde'] = lambda x: '\n\n'.join(x.astype(str))
agg_funcs['Befunde_count'] = lambda x: len(x)

# Group by the identification string and apply the aggregation functions
grouped_df = df.groupby('id').agg(agg_funcs).reset_index(drop=True)

# Save the processed dataframe to a new CSV file
grouped_df = grouped_df[grouped_df['Befunde'].notna()]
grouped_df = grouped_df[grouped_df['Befunde'].str.contains(r'[C]\d{2}\.[\d{1,2}|-]*')]
grouped_df.to_csv('aggregated_patho_documents.csv', index=False)