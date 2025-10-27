import pandas as pd

initial_csv_path = "data/patho_icdo_results.csv"
cases = pd.read_csv(initial_csv_path)
# count number of not na values in GT_ICD-10 and GT_ICD-O columns
# and save them in a new column
total_icd10 = cases['GT_ICD-10'].notna().astype(int)
total_icdo = cases['GT_ICD-O'].notna().astype(int)
print(f"Total number of cases with ICD-10 codes: {total_icd10.sum()}")
print(f"Total number of cases with ICD-O codes: {total_icdo.sum()}")
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
total_icd10_entries = len(icd_10_cases)

# Calculate percentages for ICD-10 codes
icd_10_percentages = (icd_counts / total_icd10_entries) * 100

# Extract the first three characters to get ICD-10 groups
icd_10_cases['ICD_Group'] = icd_10_cases['GT_ICD-10'].str[:3]
# Count the frequency of each ICD-10 group
group_counts = icd_10_cases['ICD_Group'].value_counts()
# Calculate percentages for ICD-10 groups
icd_10_group_percentages = (group_counts / total_icd10_entries) * 100
# Get the top 10 most frequent ICD-10 groups
top_10_groups = group_counts.head(10)

icd_o_cases = cases[cases['GT_ICD-O'].notna()]
icd_o_cases = icd_o_cases['GT_ICD-O'].str.split(',').explode()
icd_o_cases = icd_o_cases.str.replace('[', '', regex=False)
icd_o_cases = icd_o_cases.str.replace(']', '', regex=False)
icd_o_cases = icd_o_cases.str.replace('\'', '', regex=False)
icd_o_cases = icd_o_cases.str.strip()
icd_o_cases = pd.DataFrame(icd_o_cases, columns=['GT_ICD-O'])
icd_o_cases = icd_o_cases[icd_o_cases['GT_ICD-O'] != '']
icd_o_counts = icd_o_cases['GT_ICD-O'].value_counts()
total_icdo_entries = len(icd_o_cases)

# Calculate percentages for ICD-O codes
icd_o_percentages = (icd_o_counts / total_icdo_entries) * 100

# Define ICD-O classification groups
icdo_classes = {
    "800-800": (800, 800),
    "801-804": (801, 804),
    "805-808": (805, 808),
    "809-811": (809, 811),
    "812-813": (812, 813),
    "814-838": (814, 838),
    "839-842": (839, 842),
    "843-843": (843, 843),
    "844-849": (844, 849),
    "850-854": (850, 854),
    "855-855": (855, 855),
    "856-857": (856, 857),
    "858-858": (858, 858),
    "859-867": (859, 867),
    "868-871": (868, 871),
    "872-879": (872, 879),
    "880-880": (880, 880),
    "881-883": (881, 883),
    "884-884": (884, 884),
    "885-888": (885, 888),
    "889-892": (889, 892),
    "893-899": (893, 899),
    "900-903": (900, 903),
    "904-904": (904, 904),
    "905-905": (905, 905),
    "906-909": (906, 909),
    "910-910": (910, 910),
    "911-911": (911, 911),
    "912-916": (912, 916),
    "917-917": (917, 917),
    "918-924": (918, 924),
    "925-925": (925, 925),
    "926-926": (926, 926),
    "927-934": (927, 934),
    "935-937": (935, 937),
    "938-948": (938, 948),
    "949-952": (949, 952),
    "953-953": (953, 953),
    "954-957": (954, 957),
    "958-958": (958, 958),
    "959-972": (959, 972),
    "959-959": (959, 959),
    "965-966": (965, 966),
    "967-972": (967, 972),
    "967-969": (967, 969),
    "972-972": (972, 972),
    "973-973": (973, 973),
    "974-974": (974, 974),
    "975-975": (975, 975),
    "976-976": (976, 976),
    "980-994": (980, 994),
    "980-980": (980, 980),
    "981-983": (981, 983),
    "984-993": (984, 993),
    "994-994": (994, 994),
    "995-996": (995, 996),
    "997-997": (997, 997),
    "998-999": (998, 999)
}

def classify_icdo_code(code):
    """Classify ICD-O code into official classification groups"""
    try:
        # Extract first 3 digits as integer
        code_num = int(str(code)[:3])
        
        for class_name, (min_val, max_val) in icdo_classes.items():
            if min_val <= code_num <= max_val:
                return class_name
        return "Other"
    except (ValueError, TypeError):
        return "Invalid"

# Apply classification to ICD-O codes
icd_o_cases['ICD_Class'] = icd_o_cases['GT_ICD-O'].apply(classify_icdo_code)

# Count the frequency of each ICD-O class
class_counts = icd_o_cases['ICD_Class'].value_counts()
# Calculate percentages for ICD-O classes
class_percentages = (class_counts / total_icdo_entries) * 100
# Get the top 10 most frequent ICD-O classes
top_10_icdo_classes = class_counts.head(10)

print("Top 10 most frequent ICD-10 codes:")
for code, count in icd_counts.head(10).items():
    percentage = icd_10_percentages[code]
    print(f"{code}: {count} ({percentage:.2f}%)")

print("\nTop 10 most frequent ICD-10 groups:")
for group, count in top_10_groups.items():
    percentage = icd_10_group_percentages[group]
    print(f"{group}: {count} ({percentage:.2f}%)")

print("\nTop 10 most frequent ICD-O codes:")
for code, count in icd_o_counts.head(10).items():
    percentage = icd_o_percentages[code]
    print(f"{code}: {count} ({percentage:.2f}%)")
print("\nTop 10 most frequent ICD-O classes:")
for class_name, count in top_10_icdo_classes.items():
    percentage = class_percentages[class_name]
    print(f"{class_name}: {count} ({percentage:.2f}%)")

cases['text_length'] = cases['Befunde'].str.len()
    
# Calculate statistics
median_length = cases['text_length'].median()
mean_length = cases['text_length'].mean()
q1 = cases['text_length'].quantile(0.25)
q3 = cases['text_length'].quantile(0.75)
iqr = q3 - q1

print("\nText Length Statistics for 'Befunde' column:")
print(f"Median length: {median_length:.1f} characters")
print(f"Average length: {mean_length:.1f} characters")
print(f"IQR: {iqr:.1f} characters (Q1: {q1:.1f}, Q3: {q3:.1f})")
print(f"Min length: {cases['text_length'].min()}")
print(f"Max length: {cases['text_length'].max()}")