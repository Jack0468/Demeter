"""
Derives danforth_growth.csv from Bellwether SnapshotInfo.csv.

SnapshotInfo contains 33,496 watering records with:
  - weight before/after (grams) — mass of plant+pot
  - water amount (ml) — water applied
  - plant barcode — encodes genotype, population, treatment group
  - timestamp — time of measurement

We derive features equivalent to a growth/environment dataset and a
Growth_Milestone target (binned weight_after quintile).
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

snap_path = PROJECT_ROOT / 'data/raw/vision/Bellwether/SnapshotInfo.csv'
out_path   = PROJECT_ROOT / 'data/raw/enviroment/danforth_growth.csv'

print(f"Loading {snap_path}...")
snap = pd.read_csv(snap_path)
snap.columns = snap.columns.str.strip()

# Parse timestamp features
snap['timestamp'] = pd.to_datetime(snap['timestamp'])
snap['hour_of_day'] = snap['timestamp'].dt.hour
snap['day_of_experiment'] = (snap['timestamp'] - snap['timestamp'].min()).dt.days

# Decode plant barcode structure into features
# e.g. Dp1AA01002 → genotype=Dp, population=1, treatment=AA
snap['genotype_code'] = snap['plant barcode'].str[:2]    # Dp or Dr
snap['population']    = snap['plant barcode'].str[2]     # 1,2,...8
snap['treatment_code']= snap['plant barcode'].str[3:5]   # AA, AD, AE

# Derived growth proxy: weight delta (water absorbed / growth delta)
snap['weight_delta'] = snap['weight after'] - snap['weight before']

# Growth_Milestone: binned weight_after into up to 5 ordinal quintiles.
# duplicates='drop' removes ties at bin edges, so the actual bin count
# may be less than 5 — we derive labels dynamically to match.
milestone_bins = pd.qcut(snap['weight after'], q=5, duplicates='drop')
n_bins = milestone_bins.cat.categories.shape[0]
snap['Growth_Milestone'] = pd.qcut(
    snap['weight after'], q=5,
    labels=list(range(1, n_bins + 1)),
    duplicates='drop'
).astype(float)

# Select final feature set (mirrors what tabular_models.py expects)
feature_cols = [
    'weight before',
    'water amount',
    'weight_delta',
    'hour_of_day',
    'day_of_experiment',
    'genotype_code',     # categorical
    'population',        # categorical
    'treatment_code',    # categorical
    'Growth_Milestone'   # target
]

df_out = snap[feature_cols].dropna().copy()

print(f"Derived dataset shape: {df_out.shape}")
print(f"Growth_Milestone distribution:\n{df_out['Growth_Milestone'].value_counts().sort_index()}")
print(f"Null count: {df_out.isnull().sum().sum()}")

df_out.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
