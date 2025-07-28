"""
M2.py

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
import numpy as np
import re
import os
import torch
from tqdm import tqdm
from torch import inference_mode
from transformers import T5Tokenizer, T5ForConditionalGeneration


print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name())
else:
    print("No GPU available — exiting.")
    exit(1)


# ------------------------------------------------------
# CONFIGURATION — file paths
# ------------------------------------------------------

symptom_csv_path = "../../features.csv"
case_input_template = "output/chunks/M0_RAG_case_chunks_part_{}.csv"
control_input_template = "output/chunks/M0_RAG_control_chunks_part_{}.csv"

case_output_template = "output/notes/M2_case_part_{}.csv"
control_output_template = "output/notes/M2_control_part_{}.csv"

mode = os.getenv("M2_MODE", "case")
array_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
num_chunks = int(os.getenv("NUM_CHUNKS", 10))

# ------------------------------------------------------
# LOAD FEATURES AND NOTES
# ------------------------------------------------------

print("\nLoading symptoms...")
symptom_df = pd.read_csv(symptom_csv_path, sep=";")
symptom_question_list = symptom_df[["RAG_NAME", "M2_question"]].values.tolist()

print("\nLoading notes...")
if mode == "case":
    df = pd.read_csv(case_input_template.format(array_id))
else:
    df = pd.read_csv(control_input_template.format(array_id))


# ------------------------------------------------------
# LOAD FLAN-T5-XXL
# ------------------------------------------------------

print("\nLoading FLAN-T5-XXL model...")

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-xxl",
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

# ------------------------------------------------------
# FUNCTIONS
# ------------------------------------------------------

def get_confidence_yes(prompt, model, tokenizer):
    """
    Returns the softmax confidence of answering 'yes' to the given prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True).to(model.device)
    candidate_answers = ["yes", "no"]
    log_likelihoods = []

    for answer in candidate_answers:
        target_ids = tokenizer(answer, return_tensors="pt").input_ids.to(model.device)
        with inference_mode():
            output = model(input_ids=inputs.input_ids, labels=target_ids)
        neg_log_likelihood = output.loss.item()
        log_likelihoods.append(-neg_log_likelihood)

    probs = torch.nn.functional.softmax(torch.tensor(log_likelihoods), dim=0)
    return probs[0].item(), probs[1].item()  # yes, no


def evaluate_notes_with_rag_chunks(df, symptom_question_list, model, tokenizer, label=""):
    """
    Evaluates all RAG chunks for a note and symptom, keeping the max confidence score for 'yes'.

    Args:
        df (pd.DataFrame): Notes with RAG chunks
        symptom_question_list (list): List of (symptom, question) tuples
        label (str): 'case' or 'control' — determines which target column to keep

    Returns:
        pd.DataFrame: Output summary with one row per note
    """
    prompt_template = (
        "Read the following text from a clinical note:\n"
        "————\n"
        "{chunk}\n"
        "————\n"
        "{question}"
    )

    if "case" in label:
        target_column = "sepsis_target"
    elif "control" in label:
        target_column = "pseudo_target"
    else:
        raise ValueError("Label must contain 'case' or 'control'.")

    result_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {label} notes"):
        result = {
            "icustay_id": row["icustay_id"],
            "subject_id": row["subject_id"],
            "chart_time": row["chart_time"],
            "chart_hour": row["chart_hour"],
            target_column: row.get(target_column)
        }

        for symptom, question in symptom_question_list:
            clean_symptom = re.sub(r'\W+', '_', symptom.lower())
            colname = f"retrieved_chunks_for_{clean_symptom}"

            # Handle missing or invalid columns
            if colname not in row or pd.isna(row[colname]):
                result[symptom] = 0.0
                continue

            try:
                chunks = eval(row[colname]) if isinstance(row[colname], str) else row[colname]
            except Exception:
                chunks = []

            max_confidence = 0.0
            for chunk in chunks:
                prompt = prompt_template.format(chunk=chunk, question=question)
                try:
                    confidence_yes, _ = get_confidence_yes(prompt, model, tokenizer)
                    max_confidence = max(max_confidence, confidence_yes)
                except Exception as e:
                    print(f"Error evaluating chunk for {symptom}: {e}")
                    continue

            result[symptom] = round(max_confidence, 3)

        result_rows.append(result)

    return pd.DataFrame(result_rows)

# ------------------------------------------------------
# RUN EVALUATION AND SAVE
# ------------------------------------------------------

if mode == "case":
    case_df = evaluate_notes_with_rag_chunks(df, symptom_question_list, model, tokenizer, label=mode)
    case_df.to_csv(case_output_template.format(array_id), index=False)
else:
    control_df = evaluate_notes_with_rag_chunks(df, symptom_question_list, model, tokenizer, label=mode)
    control_df.to_csv(control_output_template.format(array_id), index=False)


# ------------------------------------------------------
# DONE
# ------------------------------------------------------

print(f"\nFinished M2 array {array_id}.")
