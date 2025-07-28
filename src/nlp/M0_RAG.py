# -*- coding: utf-8 -*-
"""
This script prepares and embeds clinical note chunks for RAG (Retrieval-Augmented Generation).
Steps:
1. Load and chunk case/control notes
2. Encode chunks using the INSTRUCTOR model
3. Store embeddings in ChromaDB
4. Retrieve top chunks per symptom for each note
5. Save RAG-augmented notes to CSV
"""

# ------------------------------------------------------
# IMPORTS
# ------------------------------------------------------

from InstructorEmbedding import INSTRUCTOR

import chromadb
from chromadb.config import Settings

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import os
import re


# ------------------------------------------------------
# CONFIGURATION: FILEPATHS
# ------------------------------------------------------

# Load configuration
symptom_csv_path = "../../features.csv"
case_input_path = "output/chunks/M0_case_chunks.csv"
control_input_path = "output/chunks/M0_control_chunks.csv"
case_output_template = "output/chunks/M0_RAG_case_part_{}.csv"
control_output_template = "output/chunks/M0_RAG_control_part_{}.csv"

mode = os.getenv("M2_MODE", "case")
array_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
num_chunks = int(os.getenv("NUM_CHUNKS", 10))


# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------

case_note_df = pd.read_csv(case_input_path)
control_note_df = pd.read_csv(control_input_path)

# Load symptom list
symptoms = pd.read_csv(symptom_csv_path, sep=";")['RAG_NAME'].tolist()

input_path = case_input_path if mode == "case" else control_input_path
output_template = case_output_template if mode == "case" else control_output_template
df = pd.read_csv(input_path)

if mode == "control":
    df_chunks = np.array_split(df, num_chunks)
    df = df_chunks[array_id - 1]  # array_id starts at 1 for control

# ------------------------------------------------------
# INITIALIZE MODEL AND DB
# ------------------------------------------------------

print("Loading embedding model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = INSTRUCTOR("hkunlp/instructor-large").to(device)
instruction = "Represent the clinical text for retrieval:"

chroma_client = chromadb.Client(Settings())
collection = chroma_client.get_or_create_collection(name="rag_notes")

# Flatten chunks
all_chunks, all_ids, meta = [], [], []
for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Flattening {mode}"):
    chunks = eval(row['note_chunks']) if isinstance(row['note_chunks'], str) else row['note_chunks']
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, str) and chunk.strip():
            chunk_id = f"{row['note_id']}_{i}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            meta.append({'note_id': int(row['note_id']), 'chunk_index': i})

# Encode and add to ChromaDB
chunk_inputs = [[instruction, chunk] for chunk in all_chunks]
embeddings = model.encode(chunk_inputs, batch_size=32, show_progress_bar=True).tolist()

for i in tqdm(range(0, len(all_chunks), 500), desc=f"Storing {mode}"):
    collection.add(
        documents=all_chunks[i:i+500],
        ids=all_ids[i:i+500],
        embeddings=embeddings[i:i+500],
        metadatas=meta[i:i+500]
    )



# ------------------------------------------------------
# FUNCTION: RETRIEVE MATCHING CHUNKS BY SYMPTOM
# ------------------------------------------------------

# Retrieve chunks for symptoms
def retrieve_chunks_for_symptoms(df, meta, all_chunks):
    chunk_lookup = {row['note_id']: [] for row in meta}
    for i, m in enumerate(meta):
        chunk_lookup[m['note_id']].append(all_chunks[i])

    for symptom in symptoms:
        col = f"retrieved_chunks_for_{re.sub(r'\\W+', '_', symptom.lower())}"
        df[col] = None

        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Retrieving {symptom}"):
            note_id = int(row['note_id'])
            chunks = chunk_lookup.get(note_id, [])
            if len(chunks) <= 3:
                df.at[i, col] = chunks
                continue

            symptom_embedding = model.encode([[instruction, symptom]]).tolist()[0]
            try:
                results = collection.query(
                    query_embeddings=[symptom_embedding],
                    n_results=3,
                    where={"note_id": note_id}
                )
                df.at[i, col] = results['documents'][0] if results['documents'] else []
            except Exception as e:
                print(f"RAG failed for note {note_id}: {e}")
                df.at[i, col] = []

    return df

# ------------------------------------------------------
# RETRIEVE CHUNKS 
# ------------------------------------------------------

df_rag = retrieve_chunks_for_symptoms(df, meta, all_chunks)
df_rag.to_csv(output_template.format(array_id), index=False)

# ------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------

print(f"RAG processing done for {mode} array ID {array_id}")



