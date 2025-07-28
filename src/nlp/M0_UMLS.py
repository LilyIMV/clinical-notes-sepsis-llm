"""
This script extracts symptom-related text chunks from clinical notes using:
1. Regex-based matching from a symptom dictionary (from a CSV)
2. UMLS concept linking via SciSpaCy and MedSpaCy for negation detection
It processes both case and control notes, and saves per-symptom chunk results to CSV.
"""

# ------------------------------------------------------
# IMPORTS
# ------------------------------------------------------

import pandas as pd
import numpy as np
import re
import os
from tqdm import tqdm

import scispacy
import spacy
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector
import medspacy

# ------------------------------------------------------
# CONFIGURATION — file paths and env
# ------------------------------------------------------

symptom_csv_path = "../../features.csv"
case_input_path = "output/chunks/M0_case_chunks.csv"
control_input_path = "output/chunks/M0_control_chunks.csv"
case_output_template = "output/chunks/M0_UMLS_case_chunks_part_{}.csv"
control_output_template = "output/chunks/M0_UMLS_control_chunks_part_{}.csv"

mode = os.getenv("M3_MODE", "case")
array_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
num_chunks = int(os.getenv("NUM_CHUNKS", 10))

# ------------------------------------------------------
# LOAD FEATURES
# ------------------------------------------------------

print("\nLoading symptom definitions...")
feat_df = pd.read_csv(symptom_csv_path, sep=";")
symptoms_dict = {}

for _, row in feat_df.iterrows():
    cui = row["CUI"]
    name = row["RAG_NAME"]
    pattern_str = str(row["M0_regex"]).strip()
    if pattern_str:
        try:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            symptoms_dict[cui] = (name, pattern)
        except re.error as e:
            print(f"Invalid regex for CUI {cui} ({name}): {e}")
    else:
        print(f"Skipping empty regex for {cui} ({name})")

# ------------------------------------------------------
# SET UP NLP PIPELINE
# ------------------------------------------------------

print("\nSetting up NLP pipeline...")
nlp = spacy.load("en_core_sci_md")
nlp.add_pipe("abbreviation_detector")
nlp.add_pipe("medspacy_context")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

# ------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------

def extract_chunks_per_symptom(chunks, symptoms_dict, nlp, max_hits=3):
    found_chunks = {symptoms_dict[cui][0]: [] for cui in symptoms_dict}
    remaining = set(symptoms_dict.keys())

    for chunk in chunks:
        doc = nlp(chunk)
        stop_checking = set()

        for cui in list(remaining):
            concept_name, pattern = symptoms_dict[cui]
            for match in pattern.finditer(chunk):
                start, end = match.start(), match.end()
                span = doc.char_span(start, end, alignment_mode="contract")
                if span and not span._.is_negated:
                    if len(found_chunks[concept_name]) < max_hits:
                        found_chunks[concept_name].append(chunk)
                    if len(found_chunks[concept_name]) >= max_hits:
                        stop_checking.add(cui)
                    break

        for ent in doc.ents:
            if ent._.kb_ents:
                for umls_ent in ent._.kb_ents:
                    cui = umls_ent[0]
                    if cui in remaining and not ent._.is_negated:
                        concept_name = symptoms_dict[cui][0]
                        if len(found_chunks[concept_name]) < max_hits:
                            found_chunks[concept_name].append(chunk)
                        if len(found_chunks[concept_name]) >= max_hits:
                            stop_checking.add(cui)
                        break

        remaining -= stop_checking
        if not remaining:
            break

    return found_chunks


def apply_feature_chunk_extraction(df, symptoms_dict, nlp, label=""):
    print(f"\nExtracting UMLS chunks for {label}...")
    all_symptoms = [symptoms_dict[cui][0] for cui in symptoms_dict]

    for name in all_symptoms:
        col = f"retrieved_chunks_for_{re.sub(r'\\W+', '_', name.lower())}"
        df[col] = None

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Scanning {label} notes"):
        chunks = eval(row['note_chunks']) if isinstance(row['note_chunks'], str) else row['note_chunks']
        found_chunks = extract_chunks_per_symptom(chunks, symptoms_dict, nlp)
        for name in all_symptoms:
            col = f"retrieved_chunks_for_{re.sub(r'\\W+', '_', name.lower())}"
            df.at[i, col] = found_chunks[name]

    return df

# ------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------


if mode == "case":
    input_path = case_input_path
    output_path = case_output_template.format(array_id)
    print(f"\nReading input file: {input_path}")
    df = pd.read_csv(input_path)

elif mode == "control":

    input_path = control_input_path
    output_path= control_output_template.format(array_id)
    print(f"\nReading input file: {input_path}")
    df = pd.read_csv(input_path)
    df_chunks = np.array_split(df, num_chunks)
    df = df_chunks[array_id - 1]  # array_id starts at 1 for control

else:
    raise ValueError("Mode must be 'case' or 'control'.")


umls_df = apply_feature_chunk_extraction(df, symptoms_dict, nlp, label=mode)
umls_df.to_csv(output_path, index=False)

print(f"\nSaved output to: {output_path}")
print(f"\nDone with array job {array_id}.")
