"""
M3_postprocessing.py

This script loads and cleans raw clinical measurement data for both case and control cohorts.
Each measurement column is parsed to extract valid numeric values from free-text strings.

For each measurement:
- Commas and ranges are removed
- Units (when expected) are validated and optional
- Valid numeric values are extracted and filtered within a given range

Outputs:
- Cleaned "M3_case.csv" and "M3_control.csv" files with standardized numeric values
"""

# ------------------------------------------------------
# IMPORTS
# ------------------------------------------------------

import pandas as pd
import glob
import re

## ------------------------------------------------------
# CONFIGURATION — file paths and input pattern
# ------------------------------------------------------

prefix = "output/notes/M3_control_part_"  # or "M3_case_part_"
num_parts = 10               # Expected number of chunk files

case_file_path = "output/notes/M3_case_part_0.csv"

control_output_path = "output/notes/M3_control.csv"
case_output_path = "output/notes/M3_case.csv"

# ------------------------------------------------------
# LOAD AND CONCATENATE CONTROL PARTS
# ------------------------------------------------------

print("Loading and merging control part files...")

files = [f"{prefix}{i+1}.csv" for i in range(num_parts)]
dfs = []

for file in files:
    try:
        df = pd.read_csv(file)
        dfs.append(df)
    except Exception as e:
        print(f"Failed to load {file}: {e}")

if dfs:
    M3_control_df = pd.concat(dfs, ignore_index=True)
else:
    raise ValueError("No control part files were successfully loaded.")

# ------------------------------------------------------
# LOAD CASE DATA
# ------------------------------------------------------

print("Loading case data...")
M3_case_df = pd.read_csv(case_file_path)


# ------------------------------------------------------
# CLEANING FUNCTIONS
# ------------------------------------------------------

def clean_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans clinical measurement columns in the dataframe by extracting numeric values
    and discarding invalid or out-of-range entries.
    """
    df_cleaned = df.copy()

    def extract_valid(series, units=[""], value_range=(0, 20)):
        raw = series.astype(str).str.lower().str.strip()
        raw = raw.str.replace(',', '', regex=False)

        # Remove text ranges like "2-4", "3 to 5", etc.
        is_range = raw.str.contains(r'[-+]?\d*\.?\d+\s*(?:-+|–|to|->|→)\s*[-+]?\d*\.?\d+', na=False)
        raw = raw.where(~is_range)

        # Create unit-matching regex
        has_empty = "" in units
        units_no_empty = [u for u in units if u]
        unit_pattern = f"(?:{'|'.join(re.escape(u) for u in units_no_empty)})" if units_no_empty else ""

        if unit_pattern and has_empty:
            pattern = rf'^"value":\s*[-+]?\d*\.?\d+(?:\s*{unit_pattern})?$'
        elif unit_pattern:
            pattern = rf'^"value":\s*[-+]?\d*\.?\d+\s*{unit_pattern}$'
        else:
            pattern = r'^"value":\s*[-+]?\d*\.?\d+$'

        matches = raw.str.match(pattern, na=False, flags=re.IGNORECASE)
        raw = raw.where(matches)

        extracted = raw.str.extract(r'([-+]?\d*\.?\d+)')[0]
        numeric = pd.to_numeric(extracted, errors='coerce')
        return numeric.where(numeric.between(*value_range))

    def extract_fever(series, value_range=(90, 110)):
        series = series.astype(str).str.strip().str.lower()
        starts_with_value = series.str.match(r'^"value":', na=False)
        valid = series[starts_with_value]

        matches = valid.str.extractall(r'([-+]?\d*\.?\d+)')
        matches[0] = pd.to_numeric(matches[0], errors='coerce')

        in_range = matches[0].between(*value_range)
        filtered = matches[0][in_range]

        counts = in_range.groupby(matches.index.get_level_values(0)).sum()
        single_valid = counts[counts == 1].index

        final_values = (
            filtered.groupby(filtered.index.get_level_values(0)).first()
            .reindex(series.index)
        )

        return final_values

    # ------------------------------------------------------
    # COLUMN CLEANING MAP
    # ------------------------------------------------------

    units_mcg_min_kg = ["mcg/kg/min", "mcgs/kg", "mcg/kilo", "mic/kilo", "mcg", "mcgs/kgmin", "mcg/min", ""]
    column_configs = {
        "dopamine": (units_mcg_min_kg, (0, 50)),
        "dobutamine": (units_mcg_min_kg, (0, 50)),
        "epinephrine": (units_mcg_min_kg, (0, 50)),
        "norepinephrine": (units_mcg_min_kg, (0, 50)),
        "glasgow coma score": ([""], (3, 15)),
        "bilirubin": ([""], (0.1, 60)),
        "creatinine": ([""], (0.1, 60)),
        "partial pressure of oxygen in arterial blood": ([""], (32, 700)),
        "blood platelets": (["", "K"], (0, 2000)),
        "urine output": (["cc/24hr"], (0, 1200)),
        "mean arterial pressure ": ([""], (14, 330)),
        "tachycardia": ([""], (0, 350)),
        "tachypnea": ([""], (5, 60)),
        "leukocytosis": (["K/uL", ""], (0, 1000)),
        "systolic blood pressures": ([""], (0, 375)),
    }

    for col, (units, value_range) in column_configs.items():
        if col in df.columns:
            df_cleaned[col] = extract_valid(df[col], units=units, value_range=value_range)

    if 'fever' in df.columns:
        df_cleaned['fever'] = extract_fever(df['fever'])

    return df_cleaned

# ------------------------------------------------------
# CLEAN AND SAVE
# ------------------------------------------------------

print("Cleaning case data...")
M3_case_df_clean = clean_measurements(M3_case_df)
M3_case_df_clean.to_csv(case_output_path, index=False)
print(f"Saved cleaned case data to: {case_output_path}")

print("Cleaning control data...")
M3_control_df_clean = clean_measurements(M3_control_df)
M3_control_df_clean.to_csv(control_output_path, index=False)
print(f"Saved cleaned control data to: {control_output_path}")

# ------------------------------------------------------
# DONE
# ------------------------------------------------------

print("Finished cleaning measurements.")
