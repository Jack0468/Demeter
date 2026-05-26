import os
import pandas as pd
import joblib
import json
from pathlib import Path
import sys

_current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _current_dir.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.inference_engine import predict_growth_milestone

def process_tabular_data():
    print("Loading tabular models and manifest...")
    rf_path = PROJECT_ROOT / "models" / "demeter_rf_danforth.joblib"
    rf_model = joblib.load(str(rf_path)) if rf_path.exists() else None
    
    manifest_path = PROJECT_ROOT / "data" / "processed" / "data_split_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    else:
        print("[!] ERROR: No data split manifest found.")
        return
        
    danforth_test_indices = manifest.get("demeter_rf_danforth", {}).get("test", [])
    if not danforth_test_indices:
        print("[!] ERROR: No test indices found in manifest for Danforth.")
        return
        
    danforth_csv = PROJECT_ROOT / "data" / "raw" / "enviroment" / "danforth_growth.csv"
    if not danforth_csv.exists():
        return
        
    df = pd.read_csv(danforth_csv)
    df_clean = df.dropna()
    
    # Convert string indices back to int if necessary
    danforth_test_indices = [int(i) for i in danforth_test_indices]
    df_test = df_clean.loc[danforth_test_indices]
    
    print(f"Loaded {len(df_test)} test rows from {danforth_csv}")
    
    results = []
    if rf_model:
        for idx, row in df_test.iterrows():
            env_data = row.to_dict()
            res = predict_growth_milestone(env_data, rf_model)
            
            record = env_data.copy()
            record["predicted_growth_milestone"] = res["Predicted_Growth_Milestone"]
            results.append(record)
        
    if results:
        out_df = pd.DataFrame(results)
        out_path = PROJECT_ROOT / "data" / "processed" / "tabular_features_test.csv"
        os.makedirs(out_path.parent, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"Saved {len(out_df)} tabular records to {out_path}")

if __name__ == "__main__":
    process_tabular_data()
