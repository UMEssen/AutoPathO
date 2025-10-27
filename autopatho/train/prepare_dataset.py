import pandas as pd
import re

def regex_icd_codes(doc: str):
    # remove the ICD-O code from the document
    doc = re.sub(r'[C|D]\s*\d{2}\.[\d{1,2}|-]', '', doc)
    # same for ICD-O morphology
    doc = re.sub(r'[M |\n]\d{4}\/\d{1,2}', '', doc)
    return doc

df = pd.read_csv('data/common_reports_.csv')

df["Befunde_filtered"] = df["Befunde"].apply(lambda x: regex_icd_codes(x))

df.drop(columns=["Generated_ICD-10"], inplace=True)
df.drop(columns=["Generated_ICD-O"], inplace=True)
df.drop(columns=["Generated_ICD-10_wo_locs"], inplace=True)
df.drop(columns=["Befunde"], inplace=True)

df.to_csv('data/patho_icdo_dataset.csv', index=False)
