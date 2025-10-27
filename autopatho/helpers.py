import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def load_localization_codes() -> list:
    loc_json_file = Path('data/localizations.json')
    if not loc_json_file.exists():
        loc = pd.read_excel('data/Diagnosecodeliste_ICD-O-T_DIMDI.xlsx')
        loc_codes = [{'code':row['Value'], 'localization':row['Bedeutung']} for _, row in loc.iterrows()]
        json.dump(loc_codes, open(str(loc_json_file), 'w'), indent=4)
        return loc_codes
    return json.load(open(str(loc_json_file)))

def load_material_codes() -> list:
    mat_json_file = Path('data/materials.json')
    if not mat_json_file.exists():
        mat = pd.read_excel('data/Materialcodeliste.xlsx')
        mat_codes = [{'code':row['Value'], 'material':row['Bedeutung'], 'comment':row['Bemerkung'] if not pd.isna(row['Bemerkung']) else None} for _, row in mat.iterrows()]
        json.dump(mat_codes, open(str(mat_json_file), 'w'), indent=4)
        return mat_codes
    return json.load(open(str(mat_json_file)))