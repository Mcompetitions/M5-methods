#!/usr/bin/env python
# coding: utf-8

# =========================
# Imports & Environment
# =========================
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import lightgbm as lgb

pd.options.display.max_columns = 50

# Debug: list Kaggle inputs (safe to ignore locally)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# =========================
# Dtypes
# =========================
CAL_DTYPES = {
    "event_name_1": "category",
    "event_name_2": "category",
    "event_type_1": "category",
    "event_type_2": "category",
    "weekday": "category",
    "wm_yr_wk": "int16",
    "wday": "int16",
    "month": "int16",
    "year": "int16",
    "snap_CA": "float32",
    "snap_TX": "float32",
    "snap_WI": "float32",
}

PRICE_DTYPES = {
    "store_id": "category",
    "item_id": "category",
    "wm_yr_wk": "int16",
    "sell_price": "float32",
    "rm_diff_price_4": "float32",
    "rm_diff_price_12": "float32",
    "rm_diff_price_50": "float32",
}

PROC_PRICES_DTYPES = {
    "store_id": "category",
    "item_id": "category",
    "wm_yr_wk": "int16",
    "rm_diff_price_4": "float32",
    "rm_diff_price_12": "float32",
    "rm_diff_price_50": "float32",
}


# =========================
# Globals
# =========================
h = 28
max_lags = 70
tr_last = 1941
fday = datetime(2016, 5, 23)
FIRST_DAY = 350


# =========================
# Load Once (GLOBAL CACHE)
# =========================
_PRICES_RAW = pd.read_csv("./raw_data/sell_prices.csv", dtype=PRICE_DTYPES)
_PROC_PRICES_RAW = (
    pd.read_csv("./proc_data/prices_processed.csv", dtype=PROC_PRICES_DTYPES)
    .drop("sell_price", axis=1)
)

_CAL_RAW = pd.read_csv("./raw_data/calendar.csv", dtype=CAL_DTYPES)
_PROC_CAL_RAW = (
    pd.read_csv("./proc_data/processed_calendar.csv")
    .drop(["wm_yr_wk", "wday", "month", "year", "snap_CA", "snap_TX", "snap_WI"], axis=1)
    .rename(columns={"day": "d"})
)

_SALES_RAW = pd.read_csv("./raw_data/sales_train_evaluation.csv")


# =========================
# Data Loader (Optimized)
# =========================
def create_dt(is_train=True, nrows=None, first_day=1200, dept="HOBBIES_1"):
    prices = _PRICES_RAW.copy()
    proc_price = _PROC_PRICES_RAW.copy()

    prices = prices.merge(proc_price, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()

    cal = _CAL_RAW.copy()
    proc_cal = _PROC_CAL_RAW.copy()

    cols_events_days = [f"d_{c}" for c in range(1910, 1990)]
    ev1 = cal[cal["d"].isin(cols_events_days)]["event_name_1"].unique().tolist()
    ev2 = cal[cal["d"].isin(cols_events_days)]["event_name_2"].unique().tolist()
    evs = list(set(ev1 + ev2))

    for c in set(proc_cal.columns) - {"d"}:
        proc_cal[c] = proc_cal[c].astype(int)

    cal = cal.merge(proc_cal, on="d", how="left")
    cal["date"] = pd.to_datetime(cal["date"])

    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()

    events_to_keep = (
        [f"event_name_1_{c}" for c in evs]
        + [f"event_name_2_{c}" for c in evs]
    )
    events_to_keep = [c for c in cal.columns if c in events_to_keep]

    cal = cal[
        [
            "date",
            "wm_yr_wk",
            "weekday",
            "wday",
            "month",
            "year",
            "d",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
            "snap_CA",
            "snap_TX",
            "snap_WI",
            "group_day",
        ]
        + events_to_keep
    ]

    start_day = max(1 if is_train else tr_last - max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day, tr_last + 1)]
    catcols = ["id", "item_id", "dept_id", "store_id", "cat_id", "state_id"]

    dt = _SALES_RAW[catcols + numcols].copy()
    dt = dt[dt["dept_id"] == dept]

    for col in catcols:
        if col != "id":
            dt[col] = dt[col].astype("category").cat.codes.astype("int16")
            dt[col] -= dt[col].min()

    if not is_train:
        for day in range(tr_last + 1, tr_last + 29):
            dt[f"d_{day}"] = np.nan

    dt = pd.melt(
        dt,
        id_vars=catcols,
        value_vars=[c for c in dt.columns if c.startswith("d_")],
        var_name="d",
        value_name="sales",
    )

    dt = dt.merge(cal, on="d", copy=False)
    dt = dt.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], copy=False)

    return dt


# =========================
# Feature Engineering
# =========================
def create_fea(dt):
    lags = [7, 28]
    for lag in lags:
        dt[f"lag_{lag}"] = dt.groupby("id")["sales"].shift(lag)

    wins = [7, 28]
    for win in wins:
        for lag in lags:
            dt[f"rmean_{lag}_{win}"] = (
                dt.groupby("id")[f"lag_{lag}"]
                .transform(lambda x: x.rolling(win).mean())
            )

    dt["week"] = dt["date"].dt.isocalendar().week.astype("int16")
    dt["wday"] = dt["date"].dt.weekday.astype("int16")
    dt["month"] = dt["date"].dt.month.astype("int16")
    dt["quarter"] = dt["date"].dt.quarter.astype("int16")
    dt["year"] = dt["date"].dt.year.astype("int16")
    dt["mday"] = dt["date"].dt.day.astype("int16")


# =========================
# Training + Forecast Loop
# =========================
sub_p_total = pd.DataFrame()

for dept in ["HOBBIES_1", "HOBBIES_2", "HOUSEHOLD_1", "HOUSEHOLD_2", "FOODS_1", "FOODS_2", "FOODS_3"]:
    print(f"\nTraining department: {dept}")

    df = create_dt(is_train=True, first_day=FIRST_DAY, dept=dept)
    create_fea(df)

    for c in [c for c in df.columns if "rm_diff_price_" in c]:
        df[c] = df[c].fillna(0)

    cat_feats = (
        ["item_id", "store_id", "cat_id", "state_id"]
        + ["event_type_1", "event_type_2"]
        + ["wday", "month", "snap_CA", "snap_TX", "snap_WI"]
    )

    useless_cols = [
        "id", "date", "sales", "d", "wm_yr_wk", "weekday",
        "dept_id", "sell_price", "event_name_1", "event_name_2"
    ]

    train_cols = df.columns[~df.columns.isin(useless_cols)]

    days_val = random.choices(df["d"].unique().tolist(), k=500)
    X_train = df[~df["d"].isin(days_val)][train_cols]
    y_train = df[~df["d"].isin(days_val)]["sales"]
    X_val = df[df["d"].isin(days_val)][train_cols]
    y_val = df[df["d"].isin(days_val)]["sales"]

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_feats)
    valid_data = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_feats)

    params = {
        "objective": "poisson",
        "metric": "poisson",
        "learning_rate": 0.09,
        "sub_feature": 0.9,
        "sub_row": 0.75,
        "bagging_freq": 1,
        "lambda_l2": 0.1,
        "verbosity": -1,
        "num_iterations": 2000,
        "num_leaves": 32,
        "min_data_in_leaf": 50,
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(20)],
    )

    te = create_dt(is_train=False, dept=dept)

    for tdelta in range(28):
        day = fday + timedelta(days=tdelta)
        tst = te[(te.date >= day - timedelta(days=max_lags)) & (te.date <= day)].copy()
        create_fea(tst)
        te.loc[te.date == day, "sales"] = model.predict(tst.loc[tst.date == day, train_cols])

    sub_p = (
        pd.pivot_table(te, index="id", values="sales", columns="d")
        .iloc[:, -28:]
        .reset_index()
    )

    sub_p_total = pd.concat([sub_p_total, sub_p], ignore_index=True)


# =========================
# Submission
# =========================
sub = pd.read_csv("./raw_data/sample_submission.csv", usecols=["id"])
sub = sub.merge(sub_p_total, on="id", how="left").dropna()

sub2 = sub.copy()
sub2["id"] = sub2["id"].str.replace("validation$", "evaluation", regex=True)

sub = pd.concat([sub, sub2], axis=0)
sub.columns = ["id"] + [f"F{i}" for i in range(1, 29)]
sub.to_csv("./proc_data/partial_submission.csv", index=False)

print("✅ partial_submission.csv written to ./proc_data/")