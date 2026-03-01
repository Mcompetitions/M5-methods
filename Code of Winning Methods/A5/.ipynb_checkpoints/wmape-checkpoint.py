#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd

# -----------------------------
# Config
# -----------------------------
TRUE_PATH = "./raw_data/sales_train_evaluation.csv"
PRED_PATH = "./proc_data/team_submission.csv"   # <- troque para seu arquivo
OUT_PATH = "./proc_data/metrics_by_id.csv"

# -----------------------------
# Load data
# -----------------------------
# Ground truth (history + evaluation horizon)
df_true = pd.read_csv(TRUE_PATH)

# Submission / predictions
df_pred = pd.read_csv(PRED_PATH)

# -----------------------------
# Identify columns
# -----------------------------
# Horizon: d_1914 ... d_1941 (28 dias)
true_cols = [f"d_{c}" for c in range(1914, 1942)]

# Prediction columns: assume F1...F28 OU d_1914...d_1941
if set(true_cols).issubset(df_pred.columns):
    pred_cols = true_cols
else:
    pred_cols = [c for c in df_pred.columns if c != "id"]

# -----------------------------
# Fix dtypes
# -----------------------------
df_pred[pred_cols] = df_pred[pred_cols].apply(pd.to_numeric, errors="coerce")
df_true[true_cols] = df_true[true_cols].apply(pd.to_numeric, errors="coerce")

# -----------------------------
# Merge
# -----------------------------
df_comp = df_true.merge(df_pred, on="id", how="inner")

# -----------------------------
# WMAPE (global)
# -----------------------------
y_true = df_comp[true_cols].to_numpy()
y_pred = df_comp[pred_cols].to_numpy()

wmape_global = np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()

# -----------------------------
# RMSSE scale (per id)
# -----------------------------
hist_cols = [f"d_{c}" for c in range(1, 1914)]

def rmsse_scale(x):
    diffs = np.diff(x)
    return np.mean(diffs ** 2)

df_scale = (
    df_comp[["id"] + hist_cols]
    .set_index("id")
    .apply(lambda r: rmsse_scale(r.values.astype(float)), axis=1)
    .rename("scale")
    .reset_index()
)

df_comp = df_comp.merge(df_scale, on="id", how="left")

# -----------------------------
# RMSSE per series
# -----------------------------
def rmsse_row(row):
    yt = row[true_cols].values.astype(float)
    yp = row[pred_cols].values.astype(float)

    mse = np.mean((yt - yp) ** 2)
    scale = row["scale"]

    return np.sqrt(mse / scale) if scale and scale > 0 else np.nan

df_comp["rmsse"] = df_comp.apply(rmsse_row, axis=1)

# -----------------------------
# Save per-id metrics
# -----------------------------
df_out = df_comp[["id", "rmsse"]].copy()
df_out.to_csv(OUT_PATH, index=False)

# -----------------------------
# Print summary
# -----------------------------
print("=" * 50)
print("WMAPE (global):", wmape_global)
print("Mean RMSSE:", df_comp["rmsse"].mean())
print("Saved per-id RMSSE to:", OUT_PATH)
print("=" * 50)