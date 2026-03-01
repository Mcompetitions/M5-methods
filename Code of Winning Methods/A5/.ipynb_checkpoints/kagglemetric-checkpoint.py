#!/usr/bin/env python3
# coding: utf-8

import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "./raw_data"
PRED_PATH = "./proc_data/partial_submission.csv"  # adjust if needed

SALES_EVAL_PATH = os.path.join(DATA_DIR, "sales_train_evaluation.csv")
CALENDAR_PATH = os.path.join(DATA_DIR, "calendar.csv")
PRICES_PATH = os.path.join(DATA_DIR, "sell_prices.csv")

# ============================================================
# LOAD DATA
# ============================================================
hist_cols = [f"d_{i}" for i in range(1, 1914)]
true_cols = [f"d_{i}" for i in range(1914, 1942)]

sales = pd.read_csv(
    SALES_EVAL_PATH,
    usecols=[
        "id", "item_id", "dept_id", "cat_id",
        "store_id", "state_id"
    ] + hist_cols + true_cols
)

calendar = pd.read_csv(CALENDAR_PATH, usecols=["d", "wm_yr_wk"])
prices = pd.read_csv(PRICES_PATH)

pred = pd.read_csv(PRED_PATH)

# ============================================================
# DETECT PRED COLS
# ============================================================
if set(true_cols).issubset(pred.columns):
    pred_cols = true_cols
else:
    pred_cols = [c for c in pred.columns if c != "id"]

# ============================================================
# FIX DTYPES
# ============================================================
sales[hist_cols + true_cols] = sales[hist_cols + true_cols].apply(pd.to_numeric, errors="coerce")
pred[pred_cols] = pred[pred_cols].apply(pd.to_numeric, errors="coerce")

# ============================================================
# MERGE
# ============================================================
df = sales.merge(pred, on="id", how="inner")

# ============================================================
# GLOBAL WMAPE + ITEM RMSSE
# ============================================================
y_true = df[true_cols].values
y_pred = df[pred_cols].values

wmape_global = np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()

diffs_item = df[hist_cols].values[:, 1:] - df[hist_cols].values[:, :-1]
scale_item = np.mean(diffs_item ** 2, axis=1)
scale_item[scale_item == 0] = np.nan

mse_item = np.mean((y_true - y_pred) ** 2, axis=1)
df["rmsse"] = np.sqrt(mse_item / scale_item)

# ============================================================
# WEIGHTS (DOLLAR SALES LAST 28 DAYS OF TRAIN)
# ============================================================
d_to_week = calendar.set_index("d")["wm_yr_wk"].to_dict()
last_28_train = [f"d_{i}" for i in range(1886, 1914)]

w_df = df[["id", "item_id", "store_id"] + last_28_train].melt(
    id_vars=["id", "item_id", "store_id"],
    var_name="d",
    value_name="units"
)

w_df["wm_yr_wk"] = w_df["d"].map(d_to_week)
w_df = w_df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
w_df["dollar_sales"] = w_df["units"] * w_df["sell_price"]

weights = (
    w_df.groupby("id")["dollar_sales"]
    .sum()
    .fillna(0)
)
weights = weights / weights.sum()
df = df.merge(weights.rename("weight"), on="id", how="left")

# ============================================================
# WRMSSE (CORRECT)
# ============================================================
df["__total__"] = "TOTAL"

levels = [
    ["__total__"],
    ["state_id"],
    ["store_id"],
    ["cat_id"],
    ["dept_id"],
    ["item_id"],
    ["state_id", "cat_id"],
    ["state_id", "dept_id"],
    ["store_id", "cat_id"],
    ["store_id", "dept_id"],
    ["item_id", "state_id"],
    ["item_id", "store_id"],
]

wrmsse = 0.0
num_levels = len(levels)

for lvl in levels:
    agg_true = df.groupby(lvl)[true_cols].sum()
    agg_pred = df.groupby(lvl)[pred_cols].sum()
    agg_hist = df.groupby(lvl)[hist_cols].sum()

    diffs = agg_hist.values[:, 1:] - agg_hist.values[:, :-1]
    scale_lvl = np.mean(diffs ** 2, axis=1)
    scale_lvl[scale_lvl == 0] = np.nan

    mse_lvl = np.mean((agg_true.values - agg_pred.values) ** 2, axis=1)
    rmsse_lvl = np.sqrt(mse_lvl / scale_lvl)

    w_lvl = df.groupby(lvl)["weight"].sum().values
    w_lvl = w_lvl / np.nansum(w_lvl)   # ✅ normalize per level

    wrmsse += np.nansum(w_lvl * rmsse_lvl)

wrmsse = wrmsse / num_levels   # ✅ average across levels
# ============================================================
# OUTPUT
# ============================================================
print("=" * 60)
print("WMAPE (global):", wmape_global)
print("Mean RMSSE (per series):", np.nanmean(df["rmsse"]))
print("WRMSSE (Kaggle metric):", wrmsse)
print("=" * 60)