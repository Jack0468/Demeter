import pickle
import pandas as pd
import os

pkl_path = "data/metadata_cache.pkl"
csv_path = "data/metadata_cache.csv"

if os.path.exists(pkl_path):
    with open(pkl_path, 'rb') as f:
        df = pickle.load(f)
    df.to_csv(csv_path, index=False)
    print(f"Exported to {csv_path}")
else:
    print(f"{pkl_path} not found.")
