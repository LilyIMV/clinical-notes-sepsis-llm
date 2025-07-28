"""
Author: Lily Voge, 2025

This script performs binning and imputation on preprocessed ICU patient time-series data.
It expects patient-level charted data (with hourly 'chart_time' bins) and:
- Bins variables into hourly windows from hour 47 down to `horizon` (e.g., 6)
- Aggregates and imputes missing values via forward-fill and population statistics
- Returns a list of standardized patient matrices for modeling

Input: DataFrame with one row per measurement per patient
Output: List of (time, variable) matrices per patient and a list of sorted patient IDs
"""

import pandas as pd
import numpy as np

def bin_and_impute(data, variable_start_index=5, horizon=0):
    """
    Bins and imputes time-series data for each ICU stay.

    Parameters:
    - data: DataFrame with ICU time-series data, including 'icustay_id' and 'chart_time'
    - variable_start_index: Index from which variable columns start
    - horizon: Number of hours before onset to cut off (reduces window to 48 - horizon)

    Returns:
    - imputed_all: List of DataFrames, one per patient (shape: [timesteps, variables])
    - sorted_ids: List of icustay_ids in processing order
    """

    imputed_all = []
    sorted_ids = []
    variables = data.columns[variable_start_index:]
    id_s = sorted(data['icustay_id'].unique())

    # Compute population means
    population_means = data[variables].mean(skipna=True)

    for icustay_id in id_s:
        pat = data.query("icustay_id == @icustay_id").copy()
        pat = pat.sort_values("chart_time")

        grouped = []

        #Take average of variable if there are multiple values
        other_agg = pat.groupby("chart_time")[variables].mean()
        grouped.append(other_agg)

        if not grouped:
            continue  # skip if patient has no usable data

        pat_grouped = pd.concat(grouped, axis=1)

        # Define full time bin index from 47 down to `horizon`
        full_index = np.arange(47, horizon - 1, -1)  # e.g. 47→6 if horizon=6

        # Identify available bins from the original data
        available_bins = pat_grouped.index

        # Reindex to full bin range
        pat_grouped = pat_grouped.reindex(full_index)

        # Create a mask of which bins were present in the original data
        mask_present = pat_grouped.index.isin(available_bins)

        # Step 1: Backfill only within patient’s valid bins since time goes from 47 -> horizon
        for col in pat_grouped.columns:
            values = pat_grouped[col].copy()
            # Only backfill values within the patient's observed range
            values[~mask_present] = np.nan
            values = values.ffill()
            pat_grouped[col] = values

        # Step 2: Fill remaining NaNs
        for col in pat_grouped.columns:
            if pat_grouped[col].isnull().all():
                pat_grouped[col] = 0.0
            else:
                pat_grouped[col] = pat_grouped[col].fillna(population_means[col])

        # Final check
        if pat_grouped.isnull().any().any():
            print(f"[WARN] NaNs remaining for patient {icustay_id}!")

        imputed_all.append(pat_grouped)
        sorted_ids.append(icustay_id)

    return imputed_all, sorted_ids
