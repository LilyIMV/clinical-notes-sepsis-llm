"""
M3.py

This script uses FLAN-T5-XXL to evaluate symptom presence in RAG-retrieved clinical note chunks.
For each note and symptom:
- A prompt is constructed using the chunk text and a symptom-specific question
- FLAN-T5-XXL scores confidence for "yes" vs "no"
- The maximum "yes" confidence across chunks is saved

Outputs:
- One row per note with symptom scores
- Includes subject_id, icustay_id, chart_time, and the target label
"""

# ------------------------------------------------------
# IMPORTS
# ------------------------------------------------------
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from tqdm import tqdm
import re
import os
import numpy as np
import torch

print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name())
else:
    print("No GPU available — exiting.")
    exit(1)


# ------------------------------------------------------
# FILE PATH CONFIGURATION
# ------------------------------------------------------

symptom_csv_path = "../../features.csv"
case_input_template = "output/chunks/M0_UMLS_case_chunks_part_{}.csv"
control_input_template = "output/chunks/M0_UMLS_control_chunks_part_{}.csv"

case_output_template = "output/notes/M3_case_part_{}.csv"
control_output_template = "output/notes/M3_control_part_{}.csv"

array_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
num_chunks = int(os.getenv("NUM_CHUNKS", 10))
mode = os.getenv("M3_MODE", "control")

# ------------------------------------------------------
# FLAN XXL - Load model with optimized settings
# ------------------------------------------------------

print("\n Loading Flan XXL")

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-xxl",
    device_map="auto",
    torch_dtype=torch.float16,
    max_memory={i: "70GiB" for i in range(torch.cuda.device_count())}
)

# ------------------------------------------------------
# FEATURES — load csv files with features and notes
# ------------------------------------------------------

df = pd.read_csv(symptom_csv_path, sep=";") 
symptom_question_list = df[["RAG_NAME", "M3_question", "M3_examples"]].dropna().values.tolist()

# ------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------

def flan_batch_extract_values(note_chunks, question, examples, model, tokenizer, max_tokens=512):
    if not note_chunks:
        return []

    prompts = [
        f"""
        {question} Only answer if you know the EXACT measurements, else return null. 
        Return your answer ONLY in this exact JSON format: {{"value": <number or null>}}
        --------------------------

        Here are some examples of text and the JSON output.
        {examples}
        
        --------------------------
        
        Input:
        {chunk}
        Output:
        """
        for chunk in note_chunks
    ]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded_outputs

def is_valid_output(output):
    if not output or not isinstance(output, str):
        return False
    return "null" not in output.lower().strip()

def extract_measurement_values_from_df(df, symptom_question_list, model, tokenizer, label=""):
    result_rows = []

    if 'case' in label:
        target_column = 'sepsis_target'
    elif 'control' in label:
        target_column = 'pseudo_target'
    else:
        raise ValueError("Label must contain 'case' or 'control'")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting values from {label}"):
        result = {
            "icustay_id": row["icustay_id"],
            "subject_id": row["subject_id"],
            "chart_time": row["chart_time"],
            "chart_hour": row["chart_hour"],
            target_column: row.get(target_column, None)
        }

        for symptom, question, examples in symptom_question_list:
            clean_symptom = re.sub(r'\W+', '_', symptom.lower())
            colname = f"retrieved_chunks_for_{clean_symptom}"

            if colname not in row or pd.isna(row[colname]):
                result[symptom] = None
                continue

            try:
                chunks = eval(row[colname]) if isinstance(row[colname], str) else row[colname]
            except Exception:
                chunks = []

            decoded_outputs = flan_batch_extract_values(chunks, question, examples, model, tokenizer)
            value_found = next((output for output in decoded_outputs if is_valid_output(output)), None)
            result[symptom] = value_found

        result_rows.append(result)

    return pd.DataFrame(result_rows)

# ------------------------------------------------------
# EXECUTE
# ------------------------------------------------------


if mode == "case":
    df = pd.read_csv(case_input_template.format(array_id))
    case_df = extract_measurement_values_from_df(df, symptom_question_list, model, tokenizer, label=mode)
    case_df.to_csv(case_output_template.format(array_id), index=False)
else:
    df = pd.read_csv(control_input_template.format(array_id))
    control_df = extract_measurement_values_from_df(df, symptom_question_list, model, tokenizer, label=mode)
    control_df.to_csv(control_output_template.format(array_id), index=False)


# ------------------------------------------------------
# DONE
# ------------------------------------------------------

print(f"\nFinished M3 array {array_id}.")