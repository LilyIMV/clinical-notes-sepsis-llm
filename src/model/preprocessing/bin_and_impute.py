"""
Author: Lily Voge, 2025

This module performs hourly binning and imputation of ICU time-series data for
early sepsis prediction.

1) Compute population-level feature means using the training set only.
2) For each patient:
   - Aggregate measurements into hourly bins based on `h_until_onset`.
   - Reindex to a fixed prediction window (e.g., 47 → horizon).
   - Impute missing values using forward filling (last observation carried
     forward).
   - Fill remaining leading missing values with population means.
   - Set features entirely missing in the training data to zero.

The output consists of patient × time × variable sequences for the train,
validation, and test splits, suitable for recurrent models.
"""

import pandas as pd
import numpy as np


def bin_and_impute(train_df, val_df, test_df, horizon=0, label_dicts=None):
    """
    Binning + imputation following Li et al. (2023)

    Strategy:
    1) Hourly binning via mean aggregation
    2) Forward filling (last observation carried forward, LOCF)
    3) Population-mean imputation for leading missing values
    4) Zero fill for entirely missing features

    IMPORTANT:
    - Population statistics are computed on TRAINING DATA ONLY to avoid information leakage.
    """

    # ---------------------------------------------------------------------
    # 1. Identify time-varying variables (exclude metadata)
    # ---------------------------------------------------------------------
    metadata = {
        'label', 'icustay_id', 'subject_id',
        'chart_hour', 'h_until_onset'
    }

    variables = [c for c in train_df.columns if c not in metadata]
    print("VARIABLES:", variables)

    # ---------------------------------------------------------------------
    # 2. Compute population means (TRAINING SET ONLY)
    # ---------------------------------------------------------------------
    # Li et al. explicitly use population means as fallback for leading missing values. 
    global_means = train_df[variables].mean(skipna=True)

    # Identify variables that are entirely missing in training (mean cannot be computed)
    drop_cols = global_means[global_means.isna()].index.tolist()

    if drop_cols:
        print(f"[WARN] Variables never observed in training data: {drop_cols}")
        print("These will be zero-filled after imputation.")

        # Remove from variable list but keep dimensionality consistent later
        variables = [v for v in variables if v not in drop_cols]
        global_means = global_means.drop(drop_cols)

    # ---------------------------------------------------------------------
    # 3. Per-patient binning + imputation
    # ---------------------------------------------------------------------
    def process_split(df):
        """
        Process one dataset split (train / val / test) independently.
        """
        sequences = []
        ids = sorted(df['icustay_id'].unique())

        for pid in ids:
            pat = df[df['icustay_id'] == pid].copy()

            # Ensure integer hour offsets and correct temporal ordering
            pat['h_until_onset'] = pat['h_until_onset'].round().astype(int)
            pat = pat.sort_values('h_until_onset', ascending=False)

            # -------------------------------------------------------------
            # a) Hourly binning using mean aggregation
            # -------------------------------------------------------------
            # One row per hour relative to onset
            pat_hourly = pat.groupby('h_until_onset')[variables].mean()

            # -------------------------------------------------------------
            # b) Reindex to full fixed-length window
            # -------------------------------------------------------------
            # Example: hours 47 ... 0 (or horizon)
            # Ensures all patients hadve the same time grid.
            full_idx = np.arange(47, horizon - 1, -1)
            pat_hourly = pat_hourly.reindex(full_idx)

            # -------------------------------------------------------------
            # c) Forward filling (LOCF)
            # -------------------------------------------------------------
            pat_hourly = pat_hourly.ffill()

            # -------------------------------------------------------------
            # d) Population-mean imputation for leading missing values
            # -------------------------------------------------------------
            # At the start of the sequence, forward fill cannot apply.
            # Li et al. explicitly use population means here. Which is 
            # 0 since the data is standardised.
            for col in variables:
                if pat_hourly[col].isnull().any():
                    pat_hourly[col] = pat_hourly[col].fillna(0.0)

            # -------------------------------------------------------------
            # e) Zero fill for entirely missing features
            # -------------------------------------------------------------
            # Variables never observed in training data are set to zero
            # to maintain consistent input dimensionality.
            for col in drop_cols:
                pat_hourly[col] = 0.0

            # Enforce column order consistency
            pat_hourly = pat_hourly.reindex(columns=variables + drop_cols)

            sequences.append(pat_hourly)

        return sequences, ids

    # ---------------------------------------------------------------------
    # 4. Process all splits independently
    # ---------------------------------------------------------------------
    train_out, train_ids = process_split(train_df)
    val_out, val_ids     = process_split(val_df)
    test_out, test_ids   = process_split(test_df)

    return (
        (train_out, val_out, test_out),
        (train_ids, val_ids, test_ids),
        variables
    )