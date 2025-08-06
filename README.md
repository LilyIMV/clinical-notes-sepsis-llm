# Sepsis Prediction with Clinical Notes

This project explores interpretable sepsis prediction by combining structured Electronic Health Record (EHR) data with clinical notes from the MIMIC-III database. By comparing three methods for feature extraction: rule-based, Large Language Model (LLM)-based, and a hybrid approach. The study evaluates their impact on prediction accuracy and interpretability. It aims to enhance transparency in clinical AI by leveraging LLMs to extract meaningful, explainable features from unstructured text.

The src/query and src/model/preprocessing is largely based on [Moor et al. (2019)](https://github.com/BorgwardtLab/mgp-tcn/tree/master)

# Feature Extraction Methods

M1: Rule-Based Approach
A traditional method using regular expressions and named entity recognition to detect clinical features from notes. Prioritizes interpretability but may miss context or informal language.

M2: LLM-Based Probabilistic Classification
Uses a Large Language Model (FLAN-XXL) with retrieval-augmented prompts to classify the presence of symptoms in clinical text. Outputs a confidence score for each feature based on contextual understanding.

M3: Hybrid Rule-Guided Quantification
Combines rule-based detection with LLMs to extract exact numerical values (e.g., vitals, labs) from notes. Balances symbolic precision with contextual adaptability, enabling more interpretable and quantitative inputs for downstream modeling.

## Query and real-world experiments

This repository provides a postgresql-pipeline to extract vital time series of sepsis cases and controls from the MIMIC database following the recent SEPSIS-3 definition.
 
1. Requirements for MIMIC:
  a) Requesting Access to MIMIC (publicly available, however with permission procedure)
      https://mimic.physionet.org/gettingstarted/access/
  b) Downloading and installing the MIMIC SQL database with guidance of the mimic code github:
     https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/buildmimic


2. Run the pipeline:
    a) Once the requirements are fulfilled, open the Makefile and
        - specify your username and database name
    
    To run the query, type:

        $ make query

    To analyse the Clinical Notes, type:

        $ make M1
        $ make M2
        $ make M3
    
    To run the model, type:
    
        $ make model


## Library Versions (which were used for development)
This project uses **three separate conda environments**:

### 1. `env`: General Environment

- Used for shared utilities, preprocessing, and evaluation.
- Defined in `environment.yml`.

Create it with:

```bash
conda env create -f environment.yml
```

The project depends on the `en-core-sci-md` model from [SciSpacy](https://allenai.github.io/scispacy/), which **cannot be installed directly via `environment.yml` or `requirements.txt`** because it is a language model, not a standard Python package.

You must install it manually **after activating the conda environment**:

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
```

### 2. `env_RAG`: RAG-specific Environment

- Tailored for **Retrieval-Augmented Generation (RAG)** workflows.
- Includes dependencies like `transformers`, `chromadb`, etc.
- Defined in a separate file `environment_RAG.yml` 

Create it with:

```bash
conda env create -f environment_RAG.yml
```


### 3. `env_model`: Model-specific Environment

- Includes dependencies like '`tensorflow`,' etc.
- Defined in a separate file  `environment_LSTM.yml` 

Create it with:

```bash
conda env create -f environment_LSTM.yml
```





