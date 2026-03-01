#!/usr/bin/env python
# coding: utf-8

import os
import gc
import time
import pickle
import random
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings('ignore')

# =========================
# PATH SETUP (ROBUST)
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # /app/A1/3. code/3. predict
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))  # /app/A1

RAW_DATA_DIR = os.path.join(ROOT_DIR, '2. data')
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, '2. data', 'processed')
LOG_DIR = os.path.join(ROOT_DIR, '4. logs')
MODEL_DIR = os.path.join(ROOT_DIR, '5. models')
SUBMISSION_DIR = os.path.join(ROOT_DIR, '6. submissions')

print("ROOT_DIR:", ROOT_DIR)
print("RAW_DATA_DIR exists:", os.path.exists(RAW_DATA_DIR))
print("PROCESSED_DATA_DIR exists:", os.path.exists(PROCESSED_DATA_DIR))

# =========================
# LOAD SAMPLE SUBMISSION
# =========================

submission_path = os.path.join(RAW_DATA_DIR, 'sample_submission.csv')
assert os.path.exists(submission_path), f"Missing: {submission_path}"

submission = pd.read_csv(submission_path).set_index('id').iloc[30490:]
sub_id = pd.DataFrame({'id': submission.index.tolist()})

# =========================
# CONFIG
# =========================

cvs = ['private']


STORES = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
DEPTS = ['HOBBIES_1', 'HOBBIES_2', 'HOUSEHOLD_1', 'HOUSEHOLD_2', 'FOODS_1', 'FOODS_2', 'FOODS_3']


FIRST_DAY = 710

remove_feature = [
    'id', 'state_id', 'store_id',
    'dept_id', 'cat_id',
    'date', 'wm_yr_wk', 'd', 'sales'
]

grid2_colnm = [
    'sell_price', 'price_max', 'price_min', 'price_std',
    'price_mean', 'price_norm', 'price_nunique', 'item_nunique',
    'price_momentum', 'price_momentum_m', 'price_momentum_y'
]

grid3_colnm = [
    'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
    'snap_CA', 'snap_TX', 'snap_WI',
    'tm_d', 'tm_w', 'tm_m', 'tm_y', 'tm_wm', 'tm_dw', 'tm_w_end'
]

lag_colnm = [
    'sales_lag_28','sales_lag_29','sales_lag_30','sales_lag_31','sales_lag_32',
    'sales_lag_33','sales_lag_34','sales_lag_35','sales_lag_36','sales_lag_37',
    'sales_lag_38','sales_lag_39','sales_lag_40','sales_lag_41','sales_lag_42',
    'rolling_mean_7','rolling_std_7','rolling_mean_14','rolling_std_14',
    'rolling_mean_30','rolling_std_30','rolling_mean_60','rolling_std_60',
    'rolling_mean_180','rolling_std_180'
]

mean_enc_colnm = [
    'enc_item_id_store_id_mean',
    'enc_item_id_store_id_std'
]

validation = {
    'private': [1941, 1969]
}

# =========================
# UTILS
# =========================

def reduce_mem_usage(df):
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df


def p(*args):
    return os.path.join(*args)

# =========================
# DATA PREP
# =========================

def prepare_data(store, dept):

    grid_1 = pd.read_pickle(p(PROCESSED_DATA_DIR, "grid_part_1.pkl"))
    grid_2 = pd.read_pickle(p(PROCESSED_DATA_DIR, "grid_part_2.pkl"))[grid2_colnm]
    grid_3 = pd.read_pickle(p(PROCESSED_DATA_DIR, "grid_part_3.pkl"))[grid3_colnm]

    grid_df = pd.concat([grid_1, grid_2, grid_3], axis=1)
    del grid_1, grid_2, grid_3
    gc.collect()

    grid_df = grid_df[(grid_df['store_id'] == store) & (grid_df['dept_id'] == dept)]
    grid_df = grid_df[grid_df['d'] >= FIRST_DAY]

    lag = pd.read_pickle(p(PROCESSED_DATA_DIR, "lags_df_28.pkl"))[lag_colnm]
    lag = lag.loc[lag.index.isin(grid_df.index)]

    mean_enc = pd.read_pickle(p(PROCESSED_DATA_DIR, "mean_encoding_df.pkl"))[mean_enc_colnm]
    mean_enc = mean_enc.loc[mean_enc.index.isin(grid_df.index)]

    grid_df = pd.concat([grid_df, lag, mean_enc], axis=1)
    del lag, mean_enc
    gc.collect()

    return reduce_mem_usage(grid_df)

# =========================
# PREDICT
# =========================

for cv in cvs:
    print('CV:', cv, validation[cv])

    for store in STORES:
        for dept in DEPTS:

            print(store, dept, 'start')

            grid_df = prepare_data(store, dept)

            model_var = grid_df.columns[~grid_df.columns.isin(remove_feature)]

            tr_mask = (grid_df['d'] <= validation[cv][0]) & (grid_df['d'] >= FIRST_DAY)
            vl_mask = (grid_df['d'] > validation[cv][0]) & (grid_df['d'] <= validation[cv][1])

            model_path = p(MODEL_DIR, f'non_recur_model_{store}_{dept}.bin')
            assert os.path.exists(model_path), f"Missing model: {model_path}"

            with open(model_path, 'rb') as f:
                m_lgb = pickle.load(f)

            indice = grid_df[vl_mask].index.tolist()
            preds = m_lgb.predict(grid_df.loc[vl_mask, model_var])

            prediction = pd.DataFrame({'y_pred': preds}, index=indice)

            grid_1 = pd.read_pickle(p(PROCESSED_DATA_DIR, "grid_part_1.pkl"))
            out = (
                pd.concat([grid_1.loc[indice], prediction], axis=1)
                  .pivot(index='id', columns='d', values='y_pred')
                  .reset_index()
                  .set_index('id')
            )

            os.makedirs(LOG_DIR, exist_ok=True)
            out_path = p(LOG_DIR, f'submission_storeanddept_{store}_{dept}_{cv}.csv')
            out.to_csv(out_path)

            del grid_df, prediction, out, grid_1, m_lgb
            gc.collect()

# =========================
# MAKE FINAL SUBMISSION
# =========================

pri = [a for a in os.listdir(LOG_DIR) if 'storeanddept' in a]

fcol = [f'F{i}' for i in range(1, 29)]
sub_copy = submission.copy()

for file in pri:
    temp = pd.read_csv(p(LOG_DIR, file))
    temp.columns = ['id'] + fcol
    sub_copy += sub_id.merge(temp, how='left', on='id').set_index('id').fillna(0)

sub_copy.columns = fcol

final_dir = p(SUBMISSION_DIR, 'before_ensemble')
os.makedirs(final_dir, exist_ok=True)
sub_copy.to_csv(p(final_dir, 'submission_kaggle_nonrecursive_store_dept.csv'))

print("✅ DONE")