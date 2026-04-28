import time
import os
import glob
import json
import pickle
import pandas as pd
from inference_engine import load_models, analyze_plant_status, log_to_csv
from model_builder import train_and_save_cnn, train_and_save_rf, train_and_save_cnn_plantvillage, train_and_save_rf_danforth # Now active!

# --- CONFIGURATION LOADING ---
try:
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    print("[!] ERROR: config.json not found.")
    exit(1)

TRAIN_MODEL = config['training'].get('force_retrain', False)  # Legacy support

# New granular model retraining flags
train_plantvillage_cnn = config['training']['models'].get('plantvillage_cnn', False)
train_danforth_rf = config['training']['models'].get('danforth_rf', False)
train_bellwether_cnn = config['training']['models'].get('bellwether_cnn', False)
train_bellwether_rf = config['training']['models'].get('bellwether_rf', False)

cnn_model_path = config['paths']['cnn_model']
rf_model_path = config['paths']['rf_model']
csv_log_path = config['paths']['csv_log']
bellwether_dir = config['paths']['bellwether_images_dir']
metadata_cache_path = "data/metadata_cache.pkl"

# Paths for new dataset-specific models
plantvillage_dir = "data/layer2_health_rgb/PlantVillage"
danforth_csv_path = "data/layer3_environment/plant_growth_data.csv"
plantvillage_cnn_model_path = "models/demeter_cnn_plantvillage.keras"
danforth_rf_model_path = "models/demeter_rf_danforth.joblib"


def load_bellwether_metadata(base_dir):
    """
    Reads and merges the SnapshotInfo and TileInfo CSVs to link 
    image filenames with their physical plant measurements.
    """
    print("Loading Bellwether metadata CSVs...")
    snapshot_path = os.path.join(base_dir, "SnapshotInfo.csv")
    tile_path = os.path.join(base_dir, "TileInfo.csv")
    
    if not os.path.exists(snapshot_path) or not os.path.exists(tile_path):
        print(f"[!] ERROR: Missing CSV files in {base_dir}")
        return None

    # Load the CSVs
    df_snap = pd.read_csv(snapshot_path)
    df_tile = pd.read_csv(tile_path)

    # Clean up column names to avoid trailing spaces common in CSV exports
    df_snap.columns = df_snap.columns.str.strip()
    df_tile.columns = df_tile.columns.str.strip()

    # Merge the dataframes
    merged_df = pd.merge(
        df_tile, 
        df_snap, 
        left_on='parent snapshot id', 
        right_on='id', 
        suffixes=('_tile', '_snap')
    )
    
    return merged_df


def save_metadata_cache(metadata_df, cache_path):
    """Saves processed metadata dataframe to pickle for quick reuse."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(metadata_df, f)
    print(f"Metadata cache saved to {cache_path}")


def load_metadata_cache(cache_path):
    """Loads cached metadata dataframe if available."""
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            metadata_df = pickle.load(f)
        print(f"Loaded metadata from cache: {cache_path}")
        return metadata_df
    return None


def main():
    print("Starting Demeter Software Pipeline...")
    
    # 1. LOAD METADATA (with caching)
    metadata_df = load_metadata_cache(metadata_cache_path)
    if metadata_df is None:
        metadata_df = load_bellwether_metadata(bellwether_dir)
        if metadata_df is None or metadata_df.empty:
            print("Failed to load metadata. Exiting.")
            return
        save_metadata_cache(metadata_df, metadata_cache_path)
    
    print(f"Successfully linked {len(metadata_df)} image records with plant data.\n")

    # --- TRAINING CHECK ---
    # Train PlantVillage CNN for disease classification
    if not os.path.exists(plantvillage_cnn_model_path) or train_plantvillage_cnn:
        print("\n[!] PlantVillage CNN not found or retrain forced. Training...")
        train_and_save_cnn_plantvillage(
            plantvillage_dir, 
            plantvillage_cnn_model_path, 
            epochs=config['training']['epochs']
        )
        print("[!] PlantVillage CNN training complete.\n")
    
    # Train Danforth RF for growth prediction
    if not os.path.exists(danforth_rf_model_path) or train_danforth_rf:
        print("\n[!] Danforth RF not found or retrain forced. Training...")
        train_and_save_rf_danforth(danforth_csv_path, danforth_rf_model_path)
        print("[!] Danforth RF training complete.\n")
    
    # Train original Bellwether CNN for water stress (optional/legacy)
    if not os.path.exists(cnn_model_path) or train_bellwether_cnn:
        print("\n[!] Bellwether CNN not found or retrain forced. Training...")
        train_and_save_cnn(metadata_df, bellwether_dir, cnn_model_path, epochs=config['training']['epochs'])
        print("[!] Bellwether CNN training complete.\n")
    
    # Train original Bellwether RF (optional/legacy)
    if not os.path.exists(rf_model_path) or train_bellwether_rf:
        print("\n[!] Bellwether RF not found or retrain forced. Training...")
        train_and_save_rf(metadata_df, rf_model_path)
        print("[!] Bellwether RF training complete.\n")

    # --- INFERENCE PIPELINE ---
    print("Loading AI Models...")
    # Load the primary production models (PlantVillage CNN + Danforth RF)
    cnn_model, rf_model = load_models(plantvillage_cnn_model_path, danforth_rf_model_path) 
    print("System Online. Generating diagnoses...\n")
    
    # TODO: Inference pipeline needs to be refactored for PlantVillage disease detection
    # Current Bellwether images are not compatible with PlantVillage disease classification
    # and the Danforth RF expects environmental sensor data, not just weight/water
    print("[!] Note: Inference pipeline requires refactoring for PlantVillage disease classification")
    print("[!] Skipping inference for now. Models trained and ready for deployment.\n")

if __name__ == "__main__":
    main()