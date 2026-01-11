import os
import pandas as pd
from config import SOURCES, HORIZONS
from train import train_and_evaluate_lstm
from preprocessing.main_preprocessing_clean import load_data

# List all (source, horizon) combinations
from itertools import product
all_configs = list(product(SOURCES, HORIZONS))

# Get SLURM array task ID
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))
if task_id < 0:
    raise ValueError("Missing SLURM_ARRAY_TASK_ID")


source, horizon = all_configs[task_id]
model_name = f"LSTM_{source}_h{horizon}"
print(f"\nTask {task_id}: Running {model_name}...")

# Load data
data = load_data(
    test_size=0.1,
    na_thres=500,
    variable_start_index=5,
    data_sources=['labs', 'vitals', source],
    min_length=7,
    max_length=200,
    split=0,
    horizon=horizon
)

# Train and evaluate
metrics = train_and_evaluate_lstm(data, model_name, source, horizon)

# Save results
results_csv = f"output/M4_lstm_results_auprc.csv"
results_df = pd.DataFrame([{**metrics, 'model': 'LSTM', 'source': source, 'horizon': horizon}])
results_df.to_csv(results_csv, mode='a', header=not os.path.exists(results_csv), index=False)