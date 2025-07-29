"""
Author: Lily Voge, 2025

This script performs binning and imputation on preprocessed ICU patient time-series data.
It expects patient-level charted data (with hourly 'chart_time' bins) and:
- Bins variables into hourly windows from hour 47 down to `horizon` (e.g., 6)
- Aggregates and imputes missing values via forward-fill and population statistics. Takes the mean for all measurements across training
validation, and test set per prediction label. Uses the mean for the opposite label.
- Returns a list of standardized patient matrices for modeling

Input: DataFrames for train, validation and test with one row per measurement per patient
Output: List of (time, variable) matrices per patient for train, validation, test and a list of sorted patient IDs
"""

import pandas as pd
import numpy as np

def bin_and_impute(train_df, val_df, test_df, variable_start_index=5, horizon=0, label_dicts=None):
    """
    Compute class-specific means from all combined data, then apply bin_and_impute to each split.
    
    Parameters:
    - train_df, val_df, test_df: DataFrames containing time-series data
    - variable_start_index: index where variable columns start
    - horizon: prediction horizon (used for window indexing)
    - label_dicts: dict with 'train', 'val', 'test' → {icustay_id: label} mappings

    Returns:
    - (train_imputed, val_imputed, test_imputed), (sorted_train_ids, ..., sorted_test_ids)
    """

    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    variables = all_data.columns[variable_start_index:]
    all_label_dict = {**label_dicts['train'], **label_dicts['val'], **label_dicts['test']}

    # Compute class-specific means
    label_series = all_data['icustay_id'].map(all_label_dict)
    class_0_means = all_data[label_series == 0][variables].mean(skipna=True)
    class_1_means = all_data[label_series == 1][variables].mean(skipna=True)

    # Drop variables with NaN means in either class
    nan_in_class_0 = class_0_means[class_0_means.isna()].index.tolist()
    nan_in_class_1 = class_1_means[class_1_means.isna()].index.tolist()
    columns_to_drop = list(set(nan_in_class_0 + nan_in_class_1))

    if columns_to_drop:
        print(f"[DROP] Dropping variables with undefined population means: {columns_to_drop}")
        class_0_means = class_0_means.drop(columns_to_drop)
        class_1_means = class_1_means.drop(columns_to_drop)
        variables = [v for v in variables if v not in columns_to_drop]
        all_data = all_data.drop(columns=columns_to_drop)
        train_df = train_df.drop(columns=columns_to_drop)
        val_df = val_df.drop(columns=columns_to_drop)
        test_df = test_df.drop(columns=columns_to_drop)

    def bin_single_split(data_split, label_map):
        imputed_all = []
        sorted_ids = []
        id_s = sorted(data_split['icustay_id'].unique())

        for icustay_id in id_s:
            pat = data_split.query("icustay_id == @icustay_id").copy()
            pat = pat.sort_values("chart_time")
            if pat.empty:
                continue

            label = label_map.get(icustay_id)
            population_means = class_1_means if label == 0 else class_0_means

            grouped = [pat.groupby("chart_time")[variables].mean()]
            pat_grouped = pd.concat(grouped, axis=1)

            full_index = np.arange(47, horizon - 1, -1)
            available_bins = pat_grouped.index
            pat_grouped = pat_grouped.reindex(full_index)
            mask_present = pat_grouped.index.isin(available_bins)

            for col in pat_grouped.columns:
                values = pat_grouped[col].copy()
                values[~mask_present] = np.nan
                values = values.ffill()
                pat_grouped[col] = values

            for col in pat_grouped.columns:
                if pat_grouped[col].isnull().all():
                    pat_grouped[col] = 0.0
                else:
                    mean_val = population_means.get(col)
                    if pd.isna(mean_val):
                        mean_val = 0.0
                    pat_grouped[col] = pat_grouped[col].fillna(mean_val)

            if pat_grouped.isnull().any().any():
                print(f"[WARN] NaNs remaining for patient {icustay_id}!")

            imputed_all.append(pat_grouped)
            sorted_ids.append(icustay_id)

        return imputed_all, sorted_ids

    train_imputed, train_ids = bin_single_split(train_df, label_dicts['train'])
    val_imputed, val_ids = bin_single_split(val_df, label_dicts['val'])
    test_imputed, test_ids = bin_single_split(test_df, label_dicts['test'])

    return (train_imputed, val_imputed, test_imputed), (train_ids, val_ids, test_ids), variables
