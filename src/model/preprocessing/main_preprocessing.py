'''
---------------------------------
Author: Michael Moor, 11.10.2018
Edited: Lily Voge 2025
---------------------------------
This code is used to preprocess vital/lab data in a way that the format can be directly used for mgp-rnn baseline. Our proposed methods start out from this format too.
Edited to include notes and preprocess for the keras library.

'''

import argparse
import pandas as pd
from pandas import Series
import numpy as np

from datetime import datetime

import csv
import json

import sys #for exiting the script
import time

import os.path # check if file exists
import pickle

import matplotlib.pyplot as plt
import os

#import preprocessing scripts:
from .collect_records import *
from .bin_and_impute import bin_and_impute

BASE_DIR = os.path.expanduser('~/thesis/final')

def path(rel):
    return os.path.join(BASE_DIR, rel)

def stack_patient_data(data_list):
    return np.stack([df.values for df in data_list], axis=0)

def label_ids(id_list, case_ids, control_ids):
    labels = []
    for pid in id_list:
        if pid in case_ids:
            labels.append(1)
        elif pid in control_ids:
            labels.append(0)
        else:
            raise ValueError(f"Unknown label for icustay_id: {pid}")
    return np.array(labels)

def extract_window(data=None, static_data=None, onset_name=None, horizon=0):
    """
    Extracts a time window for each patient from ICU admission to (onset - horizon),
    and creates:
      - h_until_onset: hours before onset (0 = last hour before onset)
    it deletes old chart_time.
    """
    result = pd.DataFrame()
    ids = data['icustay_id'].unique()

    for icuid in ids:
        pat = data.query("icustay_id == @icuid").copy()
        # Set index
        pat = pat.set_index(pd.to_datetime(pat['chart_time']))

        # window boundaries
        start = static_data.loc[static_data['icustay_id']==icuid, 'intime'].values[0]
        end   = static_data.loc[static_data['icustay_id']==icuid, onset_name].values[0]
        early_end = end - pd.Timedelta(hours=horizon)

        # Extract window
        pat_window = pat[start:early_end].copy()

        # Compute h_until_onset
        h_until = np.floor((early_end - pat_window.index) / pd.Timedelta(hours=1)).astype(int)

        # Insert directly after subject_id
        pat_window.insert(
            pat_window.columns.get_loc('subject_id') + 1,
            'h_until_onset',
            h_until
            )
        
        if 'chart_time' in pat_window.columns:
            pat_window.drop(columns=['chart_time'], inplace=True)

        result = pd.concat([result, pat_window.reset_index(drop=True)], ignore_index=True)
    return result


def standardize(train=None, val=None, test=None, binary_prefixes=('M1_',)):
    # Get all variables to consider for standardization (exclude metadata)
    metadata = {'label', 'icustay_id', 'subject_id', 'h_until_onset', 'chart_time', 'chart_hour'}
    all_vars = [col for col in test.columns if col not in metadata]

    # Identify binary variables
    binary_vars = [v for v in all_vars if any(v.startswith(p) for p in binary_prefixes)]

    # Everything else is continuous
    cont_vars = [v for v in all_vars if v not in binary_vars]

    # Make deep copies
    train_z = train.copy(deep=True)
    val_z = val.copy(deep=True)
    test_z = test.copy(deep=True)

    mean = train[cont_vars].mean(axis=0, skipna=True)
    std = train[cont_vars].std(axis=0, skipna=True)

    # Identify columns with all NaNs
    all_nan_cols = mean[mean.isna()].index.tolist()

    if all_nan_cols:
        print(f'[WARN] These variables have all NaNs in training data and will not be standardized: {all_nan_cols}')
        # Optionally drop them from standardization
        cont_vars = [v for v in cont_vars if v not in all_nan_cols]
        mean = mean.drop(labels=all_nan_cols)
        std = std.drop(labels=all_nan_cols)

    # Replace any std=0 (after removing NaNs) with 1 to avoid division by zero
    std = std.replace(0, 1)

    # Standardize continuous vars
    train_z[cont_vars] = (train[cont_vars] - mean) / std
    val_z[cont_vars] = (val[cont_vars] - mean) / std
    test_z[cont_vars] = (test[cont_vars] - mean) / std

    stats = {
        'mean': mean.to_dict(),
        'std': std.to_dict(),
        'names': cont_vars + binary_vars,
        'binary_vars': binary_vars
    }
    print(f'[WARN] This column will be dropped due to missing data: {all_nan_cols}')
    train_z.drop(columns=all_nan_cols, inplace=True)
    val_z.drop(columns=all_nan_cols, inplace=True)
    test_z.drop(columns=all_nan_cols, inplace=True)


    return train_z, val_z, test_z, stats


def drop_short_series(data, case_static, control_static, min_length=7):
    # Keep only patients whose onset occurs after the minimum required hours
    long_cases = case_static[case_static['sepsis_onset_hour'] >= min_length]['icustay_id'].values
    long_controls = control_static[control_static['control_onset_hour'] >= min_length]['icustay_id'].values
    selected_patients = np.concatenate([long_cases, long_controls])

    result = pd.DataFrame()
    ids = data['icustay_id'].unique()
    cases, controls = 0, 0

    for icuid in ids:
        pat = data.query("icustay_id == @icuid")

        if icuid not in selected_patients:
            continue

        result = pd.concat([result, pat], ignore_index=True)

        if icuid in long_cases:
            cases += 1
        elif icuid in long_controls:
            controls += 1

    print(f'Using {cases} cases, {controls} controls.')
    return result


def get_onset_hour(case_static, control_static):
    # Rename both onset columns to 'onset_hour' and concat
    case_hour = case_static[['icustay_id', 'sepsis_onset_hour']].rename(columns={'sepsis_onset_hour': 'onset_hour'})
    control_hour = control_static[['icustay_id', 'control_onset_hour']].rename(columns={'control_onset_hour': 'onset_hour'})
    result = pd.concat([case_hour, control_hour], ignore_index=True)
    return result


#Main function:
def load_data(test_size=0.1, horizon=0, na_thres=500, variable_start_index=5, data_sources=['labs','vitals','M1','M2','M3'], min_length=None, max_length=None, split=0):

    #Parameters:
    rs = np.random.RandomState(split)
    
    #---------------------------------
    # 0. SET PATHS 
     #---------------------------------

    # outpath to case/control-joined and window extracted file
    labvital_outpath = path('output/full_labvitals_horizon_{}.csv'.format(horizon))

    #Case vitals and labs (input, output)
    case_vitals_in = path('output/case_55h_hourly_vitals_binned.csv')
    case_vitals_out = path('output/case_55h_hourly_vitals_binned_collected.csv')
    case_labs_in = path('output/case_55h_hourly_labs_binned.csv')
    case_labs_out = path('output/case_55h_hourly_labs_binned_collected.csv')

    #Control vitals and labs (input, output)
    control_vitals_in = path('output/control_55h_hourly_vitals_binned.csv')
    control_vitals_out = path('output/control_55h_hourly_vitals_binned_collected.csv')
    control_labs_in = path('output/control_55h_hourly_labs_binned.csv')
    control_labs_out = path('output/control_55h_hourly_labs_binned_collected.csv')

    
    # Optional: Load notes if specified
    # Check if any M-version notes are requested
    note_versions = {'M1', 'M2', 'M3'}
    use_notes = note_versions.intersection(data_sources)
    use_labvitals = 'labs' in data_sources and 'vitals' in data_sources

    if not (use_notes or use_labvitals or (use_notes and use_labvitals)):
        raise ValueError(
            f"[ERROR] Invalid combination in data_sources: {data_sources}. "
            "Use either ['labs','vitals'], a single note source like ['M1'], or all three like ['labs','vitals','M1']"
        )

    if use_notes:
        print("Collecting Notes...")

        note_type = sorted(use_notes)[0]  # choose the first if multiple (e.g., M1)

        case_notes_in = path(f'output/{note_type}_case.csv')
        control_notes_in = path(f'output/{note_type}_control.csv')

        binned_split_out = path('output/labvitals_tr_te_val_binned_note_{}_min_length_{}_max_length_{}_horizon_{}_split_{}.pkl'.format(note_type, min_length,max_length, horizon,split))
    else:
        binned_split_out = path('output/labvitals_tr_te_val_binned_min_length_{}_max_length_{}_horizon_{}_split_{}.pkl'.format(min_length,max_length, horizon,split))

    #Static data:
    case_static_in = path('output/static_variables_cases.csv')
    control_static_in = path('output/static_variables_controls.csv')

    #Load static info (with onset hour / times)
    case_static = pd.read_csv(case_static_in)
    for t in ['intime','sepsis_onset']: #convert string (of times) to datetime objects
        case_static[t] = case_static[t].apply( lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S") )

    control_static = pd.read_csv(control_static_in)
    for t in ['intime','control_onset_time']: 
        control_static[t] = control_static[t].apply( lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S") )
    

    # IF NOT ALREADY RUN, UNCOMMENT THIS SECTION:
        
    #---------------------------------
    # 1. a): Collect Data line-by-line
    #---------------------------------

    print('First run of this setting. Collecting Data...')
    # First collect the data of the sql-queried csv files (case and control vitals) in single rows for each patient & point in time:

    print('Collecting Case vitals...')
    collect_records(case_vitals_in,case_vitals_out)
    #collect_records(args.infile_cases, args.outfile_collected_cases)

    print('Collecting Case labs...')
    collect_records(case_labs_in,case_labs_out)

    print('Collecting Control vitals...')
    collect_records(control_vitals_in,control_vitals_out)

    print('Collecting Control labs...')
    collect_records(control_labs_in,control_labs_out)

    #the notes have already been grouped per charttime per patient in M0_preprocessing.py

        
    #---------------------------------
    # 1. b): Load collected data
     #---------------------------------
    merge = ['icustay_id', 'chart_time', 'chart_hour', 'subject_id']

    if use_labvitals:

        print('Loading collected case records...')
        # Read and convert case vitals
        case_vitals = pd.read_csv(case_vitals_out).drop(columns=['sepsis_target'])
        case_vitals['chart_time'] = pd.to_datetime(case_vitals['chart_time'], format="%Y-%m-%d %H:%M:%S", errors='coerce')

        # Read and convert case labs
        case_labs = pd.read_csv(case_labs_out).drop(columns=['sepsis_target'])
        case_labs['chart_time'] = pd.to_datetime(case_labs['chart_time'], format="%Y-%m-%d %H:%M:%S", errors='coerce')

        print('Loading collected control records...')

        # Read and convert control vitals
        control_vitals = pd.read_csv(control_vitals_out).drop(columns=['pseudo_target'])
        control_vitals['chart_time'] = pd.to_datetime(control_vitals['chart_time'], format="%Y-%m-%d %H:%M:%S", errors='coerce')

        # Read and convert control labs
        control_labs = pd.read_csv(control_labs_out).drop(columns=['pseudo_target'])
        control_labs['chart_time'] = pd.to_datetime(control_labs['chart_time'], format="%Y-%m-%d %H:%M:%S", errors='coerce')

        #Drop NA timeframes
        for df in [case_vitals, case_labs, control_vitals, control_labs]:
            df.dropna(subset=['chart_time'], inplace=True)
            df.reset_index(drop=True, inplace=True)


    if use_notes:
        print('Loading collected note records...')

        # Read and convert case notes
        case_notes = pd.read_csv(case_notes_in).drop(columns=['sepsis_target'])
        case_notes['chart_time'] = pd.to_datetime(case_notes['chart_time'], errors='coerce')

        # Read and convert control notes       
        control_notes = pd.read_csv(control_notes_in).drop(columns=['pseudo_target'])
        control_notes['chart_time'] = pd.to_datetime(control_notes['chart_time'], errors='coerce')

        # Drop rows with missing chart_time just in case
        case_notes = case_notes.dropna(subset=['chart_time']).reset_index(drop=True)
        control_notes = control_notes.dropna(subset=['chart_time']).reset_index(drop=True)

        # Immediately rename columns (except merge keys) with note_type prefix
        case_notes.columns = [
            f"{note_type}_{col}" if col not in merge else col
            for col in case_notes.columns
        ]
        control_notes.columns = [
            f"{note_type}_{col}" if col not in merge else col
            for col in control_notes.columns
        ]





    #---------------------------------
    # 2. a): Merge lab with vital time series (from different sql tables originally) and append case and controls to one dataframe!
     #---------------------------------
    print('Merge lab and vital time series data ..')

    if use_labvitals:

        #CASE: Merge vital and lab values into one time series:
        case_labvitals = pd.merge(case_vitals, case_labs, how='outer', left_on=merge, 
            right_on=merge, sort=True)

        #CONTROL: Merge vital and lab values into one time series:
        control_labvitals = pd.merge(control_vitals, control_labs, how='outer', left_on=merge, 
            right_on=merge, sort=True)


        if use_notes:
            print('Merge note data to labvitals...')

            # === CASE MERGE ===
            case_labvitals = pd.merge(
                case_labvitals,
                case_notes,
                how='outer',
                on=merge,
                sort=True
            )

            # === CONTROL MERGE ===
            control_labvitals = pd.merge(
                control_labvitals,
                control_notes,
                how='outer',
                on=merge,
                sort=True
            )
    else:
        #CASE: Merge vital and lab values into one time series:
        case_labvitals = case_notes
        #CONTROL: Merge vital and lab values into one time series:
        control_labvitals = control_notes

    
    #---------------------------------
    # 2. b): Extract case and control window time series
    #---------------------------------
    print('Extract time series window 48 hour before onset for prediction') 
    case_labvitals = extract_window(data=case_labvitals, static_data=case_static,
                            onset_name='sepsis_onset', horizon=horizon)

    control_labvitals = extract_window(data=control_labvitals, static_data=control_static,
                                onset_name='control_onset_time', horizon=horizon)

    
    # in the extract_window() step we drop 633 control stays from 17909 control stays to 17276, as for some controls there is no data in this window (luckily no losses on cases!)

    #---------------------------------
    # 2. c): Join Cases and Controls
     #---------------------------------
    print('Merge case and control data')
    #for joining label cases/controls with label: 1/0
    control_labvitals.insert(loc=0, column='label', value=np.repeat(0,len(control_labvitals)))
    case_labvitals.insert(loc=0, column='label', value=np.repeat(1,len(case_labvitals)))


    #append cases and controls, for spliting/standardizing:
    full_labvitals = pd.concat([case_labvitals, control_labvitals], ignore_index=True)
    full_labvitals=full_labvitals.reset_index(drop=True) #drop index, so that on-the-fly df is identical with loaded one
    full_labvitals.to_csv(labvital_outpath, index=False)

    #full_labvitals = full_labvitals.dropna(axis=1, thresh=na_thres) # drop variables that don't at least have na_thres many measurements..
    #drop too short time series samples.
    if min_length or max_length:
        #handle edge cases to prevent unexpected bugs
        if min_length is None:
            min_length=0
        if max_length is None:
            max_length=100000
        full_labvitals = drop_short_series(full_labvitals, case_static, control_static, min_length=min_length)


    #---------------------------------
    # 3. a): Create Splits 
    #---------------------------------
    print('Creating Splits..')

    #Get list of actual ids (for which there was data during extraction window)
    all_ids = full_labvitals['icustay_id'].unique() #get icustay_ids of all patients
    #case_ids = case_labvitals['icustay_id'].unique()
    case_ids = full_labvitals[full_labvitals['label']==1]['icustay_id'].unique() # use full_labvitals for case_ids as case_labvitals might no be available if second run 
    #control_ids = control_labvitals['icustay_id'].unique()
    control_ids = full_labvitals[full_labvitals['label']==0]['icustay_id'].unique()

    #Createt train/test/val split ids:
    tvt_perm = rs.permutation(len(all_ids)) #train/val/test permutation of indices
    split_size = int(test_size*len(all_ids))
    test_ind = tvt_perm[:split_size]
    test_ids= all_ids[test_ind]
    validation_ind = tvt_perm[split_size:2*split_size]
    validation_ids = all_ids[validation_ind]
    train_ind = tvt_perm[2*split_size:]
    train_ids = all_ids[train_ind]
    #write TVTsplit ids of current split to json
    tvt_ids = {'train': train_ids, 'validation': validation_ids, 'test': test_ids}
    pickle.dump(tvt_ids, open(path(f'output/tvt_info_split_{split}.pkl'), 'wb') )

    ##sanity check that split was reasonably balanced (not by chance no cases in test split)
    test_prev = len(set(test_ids).intersection(case_ids))/len(test_ids) # is it comparable to overall prevalence of 9%?
    #print('Splitting: random perm of ids: {} ... '.format(tvt_perm[:10]))

    print('Split ids set up!')
    #Currently, always process lab/vital timeseries
    #actually split timeseries data: (still labeled) 
    print('Split test time series!')
    test_data = full_labvitals[full_labvitals['icustay_id'].isin(test_ids)]
    print('Split train time series!')
    train_data = full_labvitals[full_labvitals['icustay_id'].isin(train_ids)]
    print('Split validation time series!') 
    validation_data = full_labvitals[full_labvitals['icustay_id'].isin(validation_ids)]


    #---------------------------------
    # 3. b): and Standardize (still in df format, not compact one)
    #---------------------------------
    print('Standardizing time series!')


    train_z,val_z,test_z, stats = standardize(train=train_data, val=validation_data, test=test_data)
    variables = np.array(list(train_z.iloc[:,variable_start_index:]))

    # Write used stats and variable names to json for easier later processing (temporal signature creation)
    stats['names'] = variables.tolist()
    with open(path(f'output/temporal_signature_info_split_{split}.json'), 'w') as jf:
        json.dump(stats, jf)

    #call prepro script to bin and impute (carry forward) data for simple baselines
    print('Binned time series not dumped yet, computing and dumping it...')

    y_train = label_ids(tvt_ids['train'], case_ids, control_ids)
    y_val = label_ids(tvt_ids['validation'], case_ids, control_ids)
    y_test = label_ids(tvt_ids['test'], case_ids, control_ids)

    label_dicts = {
        'train': dict(zip(tvt_ids['train'], y_train)),
        'val': dict(zip(tvt_ids['validation'], y_val)),
        'test': dict(zip(tvt_ids['test'], y_test)),
    }

    #---------------------------------
    # 3. c): imputate the data
    #---------------------------------

    (train_data, validation_data, test_data), (train_ids, val_ids, test_ids), variables = bin_and_impute(
            train_z, val_z, test_z, horizon=horizon, label_dicts=label_dicts)
    
    train_labels = [label_dicts['train'][pid] for pid in train_ids]
    val_labels   = [label_dicts['val'][pid]   for pid in val_ids]
    test_labels  = [label_dicts['test'][pid]  for pid in test_ids]


    datadump = {
        'variables': variables,
        'labels': (train_labels, val_labels, test_labels),
        'data': (train_data, validation_data, test_data)
        }

    
    return datadump
