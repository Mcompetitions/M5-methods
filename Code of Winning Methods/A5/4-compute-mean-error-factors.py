#!/usr/bin/env python3
# coding: utf-8

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# -----------------------------
# Config
# -----------------------------
RAW_PATH = "./raw_data/sales_train_evaluation.csv"
PRED_PATH = "./proc_data/partial_submission.csv"
OUT_PATH = "./proc_data/factor.csv"

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

# -----------------------------
# Load data
# -----------------------------
cols_days = [f"d_{c}" for c in range(1914, 1942)]

df_eval = pd.read_csv(
    RAW_PATH,
    usecols=["id", "dept_id", "store_id"] + cols_days
)

df_model = pd.read_csv(PRED_PATH)

# -----------------------------
# Fix dtypes (important!)
# -----------------------------
pred_cols = [c for c in df_model.columns if c != "id"]
df_model[pred_cols] = df_model[pred_cols].apply(pd.to_numeric, errors="coerce")

# -----------------------------
# Merge
# -----------------------------
df_comparison = df_eval.merge(df_model, on="id", how="inner")

# -----------------------------
# Group means (numeric only)
# -----------------------------
df_comp_mean = (
    df_comparison
    .groupby(["dept_id", "store_id"])
    .mean(numeric_only=True)
)

# -----------------------------
# Plot true vs pred (raw)
# -----------------------------
for i in range(len(df_comp_mean)):
    x = df_comp_mean.columns[:28]
    y_true = df_comp_mean.iloc[i, :28].values
    y_pred = df_comp_mean.iloc[i, 28:].values
    dept, store = df_comp_mean.index[i]

    fig, ax = plt.subplots(figsize=(10, 3), dpi=150)
    fig.suptitle(f"Dept: {dept}     Store: {store}")

    ax.plot(x, y_true, label="True")
    ax.plot(x, y_pred, label="Pred")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.show()

# -----------------------------
# Compute correction factors + plot adjusted preds
# -----------------------------
depts, stores, factors = [], [], []

for i in range(len(df_comp_mean)):
    x = df_comp_mean.columns[:28]
    y_true = df_comp_mean.iloc[i, :28].values
    y_pred = df_comp_mean.iloc[i, 28:].values
    dept, store = df_comp_mean.index[i]

    denom = y_pred.mean()
    factor = y_true.mean() / denom if denom != 0 else 1.0

    depts.append(dept)
    stores.append(store)
    factors.append(factor)

    fig, ax = plt.subplots(figsize=(10, 3), dpi=150)
    fig.suptitle(f"Dept: {dept}     Store: {store}     Factor: {factor:.3f}")

    ax.plot(x, y_true, label="True")
    ax.plot(x, y_pred * factor, label="Pred (Scaled)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.show()

# -----------------------------
# Save factors
# -----------------------------
df_factors = pd.DataFrame({
    "dept_id": depts,
    "store_id": stores,
    "factor": factors
})

df_factors.to_csv(OUT_PATH, index=False)

print(f"Saved factors to: {OUT_PATH}")