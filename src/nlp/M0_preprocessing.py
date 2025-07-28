"""
Script to deduplicate clinical notes for case and control groups based on 'note_text',
then chunk the notes into overlapping text segments for downstream processing.

Steps:
1. Load raw case/control note files
2. Remove duplicate notes based on `note_text`
3. Chunk notes using `chunk_text_df`
4. Save the deduplicated chunked note files
"""

# ------------------------------------------------------
# IMPORTS
# ------------------------------------------------------

import pandas as pd
import numpy as np
import re
import os

# ------------------------------------------------------
# PATHS
# ------------------------------------------------------

case_path = "output/query/case_55h_hourly_notes_binned.csv"
control_path = "output/query/control_55h_hourly_notes_binned.csv"
case_static_in = 'output/query/static_variables_cases.csv'
control_static_in = 'output/query/static_variables_controls.csv'

case_output_path = "output/chunks/M0_case_chunks.csv"
control_output_path = "output/chunks/M0_control_chunks.csv"

# ------------------------------------------------------
# SKIP IF ALREADY DONE
# ------------------------------------------------------

if os.path.exists(case_output_path) and os.path.exists(control_output_path):
    print("Skipping preprocessing: output files already exist.")
    exit(0)

# ------------------------------------------------------
# LOAD FILES
# ------------------------------------------------------

case_df = pd.read_csv(case_path)
control_df = pd.read_csv(control_path)

case_static = pd.read_csv(case_static_in)
control_static = pd.read_csv(control_static_in)

# ------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------
def deduplicate_notes(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Deduplicates notes by 'note_text' and saves the result.

    Args:
        df (pd.DataFrame): Raw input notes.
        label (str): Group label for logging.
        output_path (str): File path to save deduplicated CSV.

    Returns:
        pd.DataFrame: Deduplicated DataFrame.
    """
    print(f"=== Processing {label.upper()} ===")
    
    original_count = len(df)
    df_deduped = df.drop_duplicates(subset=["note_text"], keep="first")
    new_count = len(df_deduped)

    print(f"→ Removed {original_count - new_count} duplicate notes")
    print(f"→ Retained {new_count} unique notes")
    return df_deduped

def chunk_text(text, min_size, max_size, word_overlap=0):
    """Splits text into chunks:
       - Prefer ending after a sentence if chunk is at least min_size
       - Force a cut at max_size if needed
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # Sentence-ish units (split by punctuation or newlines)
    split_units = re.split(r'(?<=[.!?])\s+|\n+', text)
    chunks = []
    current_chunk = ""

    for unit in split_units:
        unit = unit.strip()
        if not unit:
            continue

        # If adding the unit exceeds max_size, force a split
        if len(current_chunk) + len(unit) > max_size:
            if current_chunk:
                chunks.append(current_chunk.strip())

                # Optional: add overlap
                words = current_chunk.strip().split()
                overlap_words = " ".join(words[-word_overlap:]) if word_overlap else ""
                current_chunk = overlap_words

            # If the unit is larger than max_size, break it into pieces
            while len(unit) > max_size:
                chunks.append(unit[:max_size].strip())
                unit = unit[max_size:]
            current_chunk += (" " if current_chunk else "") + unit

        else:
            current_chunk += (" " if current_chunk else "") + unit

            # If current_chunk is long enough (≥ min_size), and ends on a sentence, finalize it
            if len(current_chunk) >= min_size and re.search(r'[.!?]\s*$', current_chunk):
                chunks.append(current_chunk.strip())

                # Add overlap if needed
                words = current_chunk.strip().split()
                overlap_words = " ".join(words[-word_overlap:]) if word_overlap else ""
                current_chunk = overlap_words

    # Append any remaining chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks



def chunk_text_df(df, static_data, name="", min_size=216, max_size=512, word_overlap=0):
    print(f"=== CREATING DF WITH UNIQUE CHUNKS OF {name.upper()} ===")
    if name == "case":
        onset_name='sepsis_onset'
    else:
        onset_name='control_onset_time'
    df = df.copy()
    df['chart_time'] = pd.to_datetime(df['chart_time'])
    
    result_rows = []

    for icuid in df['icustay_id'].unique():
        pat_notes = df[df['icustay_id'] == icuid].copy()
        pat_notes['chart_time'] = pd.to_datetime(pat_notes['chart_time'])
        
        # Get start and end time for window
        try:
            start = pd.to_datetime(static_data.loc[static_data['icustay_id'] == icuid, 'intime'].values[0])
            end = pd.to_datetime(static_data.loc[static_data['icustay_id'] == icuid, onset_name].values[0])

        except IndexError:
            continue  # Skip if patient not found in static_data

        pat_notes = pat_notes[(pat_notes['chart_time'] >= start) & (pat_notes['chart_time'] < end)].copy()

        # Compute bin as hours before onset
        pat_notes['chart_time_bin'] = np.floor(
            (end - pat_notes['chart_time']) / pd.Timedelta(hours=1)
        ).astype(int)

        # Chunk each note
        pat_notes['note_chunks'] = pat_notes['note_text'].apply(
            lambda x: chunk_text(x, min_size, max_size, word_overlap)
        )

        # Group by bin
        def dedup_chunks(group):
            all_chunks = sum(group['note_chunks'], [])
            unique_chunks = list(dict.fromkeys(all_chunks))
            all_notes = list(group['note_text'])

            first_row = group.iloc[0].copy()
            first_row['note_chunks'] = unique_chunks
            first_row['note_text'] = all_notes
            first_row['chart_time'] = pd.to_datetime(group['chart_time']).mean()
            return first_row

        grouped = (
            pat_notes.groupby(['icustay_id', 'chart_time_bin'], group_keys=False)
            .apply(dedup_chunks)
            .reset_index(drop=True)
        )

        result_rows.append(grouped)

    # Final combined DataFrame
    final_df = pd.concat(result_rows, ignore_index=True)
    final_df = final_df.drop(columns=['chart_time_bin'], errors='ignore')

    return final_df


# ------------------------------------------------------
# DEDUPLICATE NOTES
# ------------------------------------------------------

case_deduped = deduplicate_notes(case_df, label="cases")
control_deduped = deduplicate_notes(control_df, label="controls")

# ------------------------------------------------------
# CHUNK NOTES
# ------------------------------------------------------

chunk_params = {
    "min_size": 256,
    "max_size": 512,
    "word_overlap": 5
}

case_note_df = chunk_text_df(case_deduped, case_static, name="case", **chunk_params)
control_note_df = chunk_text_df(control_deduped, control_static, name="control", **chunk_params)

print(f" \n Final case df column names and shape: \n {case_note_df.columns}  \n {case_note_df.shape}")
print(f" \n Final control df column names and shape: \n {control_note_df.columns} \n {control_note_df.shape}")

case_note_df.to_csv(case_output_path, index=False)
control_note_df.to_csv(control_output_path, index=False)

# ------------------------------------------------------
# DONE
# ------------------------------------------------------

print("\n Preprocessing complete: notes deduplicated and chunked.")
