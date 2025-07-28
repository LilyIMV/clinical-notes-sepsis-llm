import pandas as pd

## ------------------------------------------------------
# CONFIGURATION — file paths and input pattern
# ------------------------------------------------------

prefix = "output/notes/M2_control_part_"  # or "M3_case_part_"
num_parts = 10               # Expected number of chunk files
control_output_path = "output/notes/M2_control.csv"


case_input_path = "output/notes/M2_case_part_0.csv"  # or "M3_case_part_"
case_output_path = "output/notes/M2_case.csv"


# ------------------------------------------------------
# LOAD AND CONCATENATE CONTROL PARTS
# ------------------------------------------------------
print("Loading case file...")

M2_case_df = pd.read_csv(case_input_path)

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
    M2_control_df = pd.concat(dfs, ignore_index=True)

else:
    print(files)
    raise ValueError("No control part files were successfully loaded.")

M2_control_df.to_csv(control_output_path, index=False)
M2_case_df.to_csv(case_output_path, index=False)

print(f"Succesfully saved files{control_output_path, case_output_path}")



