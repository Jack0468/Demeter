import os
import shutil
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path
import json

# --- DYNAMIC PROJECT ROOT RESOLUTION ---
_current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _current_dir.parent if _current_dir.name == "scripts" else _current_dir

def main():
    config_path = PROJECT_ROOT / 'config.json'
    if not config_path.exists():
        config_path = PROJECT_ROOT / 'config' / 'config.json'
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    base_dir = config['paths']['bellwether_images_dir']
    cache_path = PROJECT_ROOT / 'data' / 'metadata_cache.pkl'
    
    if not cache_path.exists():
        print(f"[!] Error: {cache_path} not found. Have you run train_pipeline.py yet?")
        sys.exit(1)
        
    with open(cache_path, 'rb') as f:
        df = pickle.load(f)
        
    print("Preparing Bellwether Test Set...")
    
    # --- 1. Prepare CNN Test Split ---
    df_vis = df[df['spectrum'].str.contains('Visible', na=False, case=False)].copy()
    df_vis['filepath'] = df_vis.apply(
        lambda row: os.path.join(base_dir, f"snapshot{row['parent snapshot id']}", f"{row['name']}.jpg"),
        axis=1
    )
    df_vis = df_vis[df_vis['filepath'].apply(lambda p: os.path.exists(p) and os.path.getsize(p) > 0)]
    
    median_water = df_vis['water amount'].median()
    df_vis['label_int'] = (df_vis['water amount'] >= median_water).astype(int)
    class_names = ['Water_Stressed', 'Well_Watered']
    
    # Exact same split as train_and_save_cnn
    _, val_df_cnn = train_test_split(df_vis, test_size=0.2, random_state=123)
    
    out_dir_cnn = PROJECT_ROOT / 'data' / 'bellwether_test_images'
    for c in class_names:
        os.makedirs(out_dir_cnn / c, exist_ok=True)
        
    print(f"Creating CNN class directories at {out_dir_cnn} (Copying {len(val_df_cnn)} images)...")
    for _, row in val_df_cnn.iterrows():
        src = row['filepath']
        cls_name = class_names[row['label_int']]
        dst = out_dir_cnn / cls_name / os.path.basename(src)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            
    # --- 2. Prepare RF Test Split ---
    df_clean = df.dropna(subset=['weight before', 'water amount', 'weight after']).copy()
    # Exact same split as train_and_save_rf
    _, val_df_rf = train_test_split(df_clean, test_size=0.2, random_state=42)
    
    out_csv = PROJECT_ROOT / 'data' / 'bellwether_rf_test.csv'
    
    # evaluate_rf.py expects all columns except the target to be features.
    # The Bellwether model only uses 'weight before' and 'water amount'.
    # We place the target 'weight after' as the last column so evaluate_rf.py auto-detects it.
    val_df_rf[['weight before', 'water amount', 'weight after']].to_csv(out_csv, index=False)
    print(f"Saved RF tabular test set to {out_csv}")
    
    print("\n[!] Setup complete. You can now run the evaluation scripts using these paths!")

if __name__ == "__main__":
    main()