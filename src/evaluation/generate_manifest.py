import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

_current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _current_dir.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.training.split_tracker import update_manifest

def generate_miniature_manifest():
    print("Generating miniature data split manifest...")
    
    # 1. PlantVillage Split
    pv_dir = PROJECT_ROOT / "data" / "raw" / "vision" / "PlantVillage"
    if pv_dir.exists():
        image_paths = []
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            image_paths.extend(glob.glob(str(pv_dir / "**" / ext), recursive=True))
        
        if image_paths:
            # We don't need stratify for this mock manifest, just random split
            train_paths, test_paths = train_test_split(image_paths, test_size=0.2, random_state=123)
            # Ensure paths are saved consistently
            train_paths = [Path(p).resolve().as_posix() for p in train_paths]
            test_paths = [Path(p).resolve().as_posix() for p in test_paths]
            update_manifest("demeter_cnn_plantvillage", train_paths, test_paths)
            print(f"PlantVillage: {len(train_paths)} train, {len(test_paths)} test")

    # 2. Danforth Tabular Split
    danforth_csv = PROJECT_ROOT / "data" / "raw" / "enviroment" / "danforth_growth.csv"
    if danforth_csv.exists():
        df = pd.read_csv(danforth_csv)
        df_clean = df.dropna()
        # In tabular_models.py, we split and recorded the index
        train_idx, test_idx = train_test_split(df_clean.index, test_size=0.2, random_state=42)
        update_manifest("demeter_rf_danforth", train_idx.tolist(), test_idx.tolist())
        print(f"Danforth RF: {len(train_idx)} train, {len(test_idx)} test")

if __name__ == "__main__":
    generate_miniature_manifest()
