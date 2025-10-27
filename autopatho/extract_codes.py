import re
import asyncio
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from .helpers import load_material_codes, load_localization_codes
from .llm import generate_icd_code

load_dotenv()

def regex_icd_codes(doc: str):
    # use regex to extract the ICD-1O code from the document
    match_icd_10 = re.findall(r'[C|D]\s*\d{2}\.[\d{1,2}|-]', doc)
    icd_10 = match_icd_10 if match_icd_10 else None
    # remove the ICD-O code from the document
    doc = re.sub(r'[C|D]\s*\d{2}\.[\d{1,2}|-]', '', doc)
    # same for ICD-O morphology
    match_icd_o_m = re.findall(r'[M |\n]\d{4}\/\d{1,2}', doc)
    icd_o_list = []
    for i in match_icd_o_m:
        if i.startswith(' 8') or i.startswith(' 9') or i.startswith('\n8') or i.startswith('\n9'):
            icd_o_list.append(i.replace('\n', '').replace(' ', ''))
    icd_o_m = icd_o_list if match_icd_o_m else None
    doc = re.sub(r'[M |\n]\d{4}\/\d{1,2}', '', doc)
    return icd_10, icd_o_m, doc

async def handle_code_extraction(cases: pd.DataFrame, csv_path: str):
    if 'GT_ICD-10' not in cases.columns:
        cases['GT_ICD-10'] = None
        cases['GT_ICD-O'] = None
        cases['Generated_ICD-10'] = None
        cases['Generated_ICD-10_reasoning'] = None
        cases['Generated_ICD-O'] = None
        cases['Generated_ICD-O_reasoning'] = None
        cases['Generated_ICD-10_wo_locs'] = None
        cases['Generated_ICD-10_wo_locs_reasoning'] = None
    tasks = []
    for index, row in cases.iterrows():
        icd_10, icd_o_m, doc = regex_icd_codes(row['Befunde'])
        # only process those rows that have not already been processed
        if row['Befunde'] and pd.isna(row['Generated_ICD-10']):
            if icd_10:
                cases.at[index, 'GT_ICD-10'] = str(icd_10)
                task = asyncio.create_task(generate_icd_code("icd_10", doc, loc_codes, cases, index, csv_path, use_openai_api=True))
                tasks.append(task)
                if pd.isna(row['Generated_ICD-10_wo_locs']):
                    task = asyncio.create_task(generate_icd_code("icd_10_wo_locs", doc, loc_codes, cases, index, csv_path, use_openai_api=True))
                    tasks.append(task)
        if icd_o_m and pd.isna(row['Generated_ICD-O']):
            cases.at[index, 'GT_ICD-O'] = str(icd_o_m)
            task = asyncio.create_task(generate_icd_code("icd_o", doc, loc_codes, cases, index, csv_path, use_openai_api=True))
            tasks.append(task)
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    loc_codes = load_localization_codes()
    mat_codes = load_material_codes()
    initial_csv_path = "data/initial_dataset.csv"
    csv_path = "data/model_name_results.csv"
    cases = pd.read_csv(csv_path if Path(csv_path).exists() else initial_csv_path)
    asyncio.run(handle_code_extraction(cases, csv_path))