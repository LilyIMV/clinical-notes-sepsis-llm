import os

# Define the output directories and file names
OUTPUT_DIR = "output"
SHAP_DIR = os.path.join(OUTPUT_DIR, "M4_shap_plots")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "M4_lstm_results.csv")
AUC_TABLE_CSV = os.path.join(OUTPUT_DIR, "M4_lstm_auc_table.csv")

# These must exist for main.py to work
SOURCES = ['no_notes', 'M1', 'M2', 'M3']
HORIZONS = [0, 6, 12]

NOTES_ONLY = False

SHAP_SAMPLE_SIZE = 100

# ------------------------------------------------------
# CONFIGURATION — SHAP computation enabled for specific (source, horizon) pairs
# ------------------------------------------------------

SHAP_CONFIGS = {}


# Make sure SHAP directory exists
os.makedirs(SHAP_DIR, exist_ok=True)
