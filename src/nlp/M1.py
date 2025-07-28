"""
M1_symptom_extraction.py

This script detects symptom presence in clinical note chunks using:
1. Regex patterns from a symptom dictionary (CSV)
2. UMLS linking with SciSpaCy and MedSpaCy for negation detection

Steps:
- Load symptom definitions and chunked notes
- Initialize SciSpaCy pipeline with UMLS linker and negation detector
- Extract binary presence (0/1) flags for each symptom
- Save results to CSV for case and control groups
"""

# ------------------------------------------------------
# IMPORTS
# ------------------------------------------------------

import pandas as pd
import re
from tqdm import tqdm

import scispacy
import spacy
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector
import medspacy

# ------------------------------------------------------
# CONFIGURATION — file paths
# ------------------------------------------------------

symptom_csv_path = "../../features.csv"
case_chunked_path = "output/chunks/M0_case_chunks.csv"
control_chunked_path = "output/chunks/M0_control_chunks.csv"
case_output_path = "output/notes/M1_case.csv"
control_output_path = "output/notes/M1_control.csv"

# ------------------------------------------------------
# LOAD FEATURES AND NOTES
# ------------------------------------------------------

print("\n Loading symptom definitions...")
feat_df = pd.read_csv(symptom_csv_path, sep=";")

# Build {CUI: (concept_name, compiled_regex)} dictionary
symptoms_dict = {
    row["CUI"]: (row["RAG_NAME"], re.compile(row["M1_regex"], re.IGNORECASE))
    for _, row in feat_df.iterrows()
}

print(" Loading chunked note data...")
case_note_df = pd.read_csv(case_chunked_path)
control_note_df = pd.read_csv(control_chunked_path)

# ------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------

def extract_symptom_presence(chunks, symptoms_dict, nlp):
    """
    Check for presence/absence of each symptom in a list of chunks using regex + UMLS.

    Returns:
        dict: {symptom_name: 0 | 1 | NaN}
    """
    symptom_flags = {symptoms_dict[cui][0]: float('nan') for cui in symptoms_dict}
    remaining = set(symptoms_dict.keys())

    for chunk in chunks:
        doc = nlp(chunk)
        stop_checking = set()

        # --- Regex matching ---
        for cui in list(remaining):
            concept_name, pattern = symptoms_dict[cui]
            for match in pattern.finditer(chunk):
                start, end = match.start(), match.end()
                span = doc.char_span(start, end, alignment_mode="contract")
                if span:
                    symptom_flags[concept_name] = 0 if span._.is_negated else 1
                    stop_checking.add(cui)
                    break

        # --- UMLS linking ---
        for ent in doc.ents:
            if ent._.kb_ents:
                for umls_ent in ent._.kb_ents:
                    cui = umls_ent[0]
                    if cui in remaining:
                        concept_name = symptoms_dict[cui][0]
                        symptom_flags[concept_name] = 0 if ent._.is_negated else 1
                        stop_checking.add(cui)
                        break

        remaining -= stop_checking
        if not remaining:
            break

    return symptom_flags


def apply_symptom_presence_extraction(df, symptoms_dict, nlp, label="", output_path="output.csv"):
    """
    Adds binary symptom presence columns (0/1/NaN) to a new DataFrame and saves to disk.
    """
    print(f"\n Extracting symptom flags for {label} notes...")

    all_symptoms = [symptoms_dict[cui][0] for cui in symptoms_dict]
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Scanning {label} notes"):
        chunks = eval(row["note_chunks"]) if isinstance(row["note_chunks"], str) else row["note_chunks"]
        flags = extract_symptom_presence(chunks, symptoms_dict, nlp)

        row_target_col = "sepsis_target" if label == "case" else "pseudo_target"
        result = {
            "subject_id": row.get("subject_id"),
            "icustay_id": row.get("icustay_id"),
            "chart_time": row.get("chart_time"),
            "chart_hour": row.get("chart_hour"),
            row_target_col: row.get(row_target_col),
        }

        for name in all_symptoms:
            result[name] = flags[name]


        results.append(result)

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    print(f"Saved {label} symptom flags to: {output_path}")

# ------------------------------------------------------
# SET UP NLP PIPELINE
# ------------------------------------------------------

print("\n Setting up SciSpaCy NLP pipeline...")

nlp = spacy.load("en_core_sci_md")
nlp.add_pipe("abbreviation_detector")
nlp.add_pipe("medspacy_context")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

# ------------------------------------------------------
# RUN EXTRACTION AND SAVE
# ------------------------------------------------------

print("\n Running symptom extraction pipeline...")

apply_symptom_presence_extraction(case_note_df, symptoms_dict, nlp, label="case", output_path=case_output_path)
apply_symptom_presence_extraction(control_note_df, symptoms_dict, nlp, label="control", output_path=control_output_path)

# ------------------------------------------------------
# DONE
# ------------------------------------------------------

print("\n All done: symptom presence data saved for case and control notes.")
