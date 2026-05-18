import os
import json
import pickle
import pandas as pd
import sys
from pathlib import Path

'''
Train all models as required based on config.json in the 
config directory, which will run sequentially and save
the trained models to the models directory.

Use 2>&1 to log all output to a file for debugging.
Example: python src/training/train_pipeline.py > logs/train_log.txt 2>&1
'''

# --- DYNAMIC PROJECT ROOT RESOLUTION ---
_current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _current_dir.parent.parent if _current_dir.parent.name == "src" else _current_dir.parent

# Add project root to path for imports from src.*
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.training.vision_models import train_and_save_cnn, train_and_save_cnn_plantvillage, train_tiller_cnn_regressor, train_biomass_cnn_regressor
    from src.training.tabular_models import train_and_save_rf, train_and_save_rf_danforth
    from src.training.train_kmeans_cluster import train_health_clusters
except ModuleNotFoundError:
    from vision_models import train_and_save_cnn, train_and_save_cnn_plantvillage, train_tiller_cnn_regressor, train_biomass_cnn_regressor
    from tabular_models import train_and_save_rf, train_and_save_rf_danforth
    from train_kmeans_cluster import train_health_clusters

try:
    from scripts.setup_tiller_data import load_manual_tiller_data
except ModuleNotFoundError:
    print("[!] Warning: scripts.setup_tiller_data not found. Tiller CNN training will be skipped.")
    load_manual_tiller_data = None

try:
    from scripts.setup_biomass_data import load_biomass_data
except ModuleNotFoundError:
    print("[!] Warning: scripts.setup_biomass_data not found. Biomass CNN training will be skipped.")
    load_biomass_data = None

# --- CONFIGURATION LOADING ---
config_path = PROJECT_ROOT / 'config.json'
if not config_path.exists():
    config_path = PROJECT_ROOT / 'config' / 'config.json'

try:
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    print(f"[!] ERROR: config.json not found at {config_path}.")
    exit(1)

TRAIN_MODEL = config['training'].get('force_retrain', False)  # Legacy support

# New granular model retraining flags
train_plantvillage_cnn = config['training']['models'].get('plantvillage_cnn', False)
train_danforth_rf = config['training']['models'].get('danforth_rf', False)
train_bellwether_cnn = config['training']['models'].get('bellwether_cnn', False)
train_bellwether_rf = config['training']['models'].get('bellwether_rf', False)
train_tiller_cnn = config['training']['models'].get('tiller_cnn', False)
train_kmeans = config['training']['models'].get('kmeans', False)
train_biomass_cnn = config['training']['models'].get('biomass_cnn', False)

cnn_model_path = str(PROJECT_ROOT / config['paths']['cnn_model'])
rf_model_path = str(PROJECT_ROOT / config['paths']['rf_model'])
bellwether_dir = str(PROJECT_ROOT / config['paths']['bellwether_images_dir'])
metadata_cache_path = str(PROJECT_ROOT / "data/metadata_cache.pkl")

# Paths for new dataset-specific models
plantvillage_dir = str(PROJECT_ROOT / config['paths']['plantvillage_dir'])
danforth_csv_path = str(PROJECT_ROOT / config['paths']['danforth_csv_path'])
plantvillage_cnn_model_path = str(PROJECT_ROOT / config['paths']['plantvillage_cnn_model_path'])
danforth_rf_model_path = str(PROJECT_ROOT / config['paths']['danforth_rf_model_path'])

tiller_txt_path = str(PROJECT_ROOT / config['paths'].get('tiller_data_path', ''))
tiller_img_dir = str(PROJECT_ROOT / config['paths'].get('tiller_data_image_path', ''))
tiller_cnn_model_path = str(PROJECT_ROOT / config['paths'].get('tiller_cnn_model_path', 'models/demeter_cnn_tiller.keras'))

biomass_csv_path = str(PROJECT_ROOT / config['paths'].get('biomass_data_csv', 'data/raw/vision/Manual_ biomass_measurements/manual_biomass_samples.csv'))
biomass_img_dir = str(PROJECT_ROOT / config['paths'].get('biomass_data_image_path', 'data/raw/vision/Manual_ biomass_measurements/image'))
biomass_cnn_model_path = str(PROJECT_ROOT / config['paths'].get('biomass_cnn_model_path', 'models/demeter_cnn_biomass.keras'))

kmeans_model_path = str(PROJECT_ROOT / config['paths'].get('kmeans_model_path', 'models/health_clusters.joblib'))


def load_bellwether_metadata(images_dir):
    """
    Reads and merges the SnapshotInfo and TileInfo CSVs to link 
    image filenames with their physical plant measurements.

    The CSVs live in the Bellwether root (parent of images_dir),
    while images_dir points to the images/ subdirectory.
    """
    print("Loading Bellwether metadata CSVs...")
    # CSVs sit one level above the images directory
    csv_dir = str(Path(images_dir).parent)
    snapshot_path = os.path.join(csv_dir, "SnapshotInfo.csv")
    tile_path = os.path.join(csv_dir, "TileInfo.csv")
    
    if not os.path.exists(snapshot_path) or not os.path.exists(tile_path):
        print(f"[!] ERROR: Missing CSV files in {csv_dir}")
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
    print("Starting Model Training Pipeline...")
    
    # 1. LOAD METADATA (with caching)
    metadata_df = load_metadata_cache(metadata_cache_path)
    if metadata_df is None:
        metadata_df = load_bellwether_metadata(bellwether_dir)
        if metadata_df is None or metadata_df.empty:
            print("Failed to load metadata. Skipping Bellwether training.")
        else:
            save_metadata_cache(metadata_df, metadata_cache_path)
            
    if metadata_df is not None:
        print(f"Successfully linked {len(metadata_df)} image records with plant data.\n")

    # --- TRAINING CHECK ---
    if not os.path.exists(plantvillage_cnn_model_path) or train_plantvillage_cnn:
        print("\n[!] PlantVillage CNN not found or retrain forced. Training...")
        train_and_save_cnn_plantvillage(plantvillage_dir, plantvillage_cnn_model_path, epochs=config['training']['epochs'])
        print("[!] PlantVillage CNN training complete.\n")
    
    if not os.path.exists(danforth_rf_model_path) or train_danforth_rf:
        print("\n[!] Danforth RF not found or retrain forced. Training...")
        train_and_save_rf_danforth(danforth_csv_path, danforth_rf_model_path)
        print("[!] Danforth RF training complete.\n")
    
    if metadata_df is not None and (not os.path.exists(cnn_model_path) or train_bellwether_cnn):
        print("\n[!] Bellwether CNN not found or retrain forced. Training...")
        train_and_save_cnn(metadata_df, bellwether_dir, cnn_model_path, epochs=config['training']['epochs'])
        print("[!] Bellwether CNN training complete.\n")
    
    if metadata_df is not None and (not os.path.exists(rf_model_path) or train_bellwether_rf):
        print("\n[!] Bellwether RF not found or retrain forced. Training...")
        train_and_save_rf(metadata_df, rf_model_path)
        print("[!] Bellwether RF training complete.\n")

    if train_tiller_cnn or not os.path.exists(tiller_cnn_model_path):
        print("\n[!] Tiller CNN Regressor not found or retrain forced. Training...")
        if load_manual_tiller_data is None:
            print("[!] Tiller data loader unavailable. Skipping Tiller CNN training.")
        else:
            tiller_df = load_manual_tiller_data(tiller_txt_path, tiller_img_dir)
            if tiller_df is not None and not tiller_df.empty:
                train_tiller_cnn_regressor(tiller_df, tiller_cnn_model_path, epochs=config['training']['epochs'])
                print("[!] Tiller CNN Regressor training complete.\n")
            else:
                print("[!] Failed to load Tiller dataset. Skipping Tiller CNN training.")

    if train_biomass_cnn or not os.path.exists(biomass_cnn_model_path):
        print("\n[!] Biomass CNN Regressor not found or retrain forced. Training...")
        if load_biomass_data is None:
            print("[!] Biomass data loader unavailable. Skipping Biomass CNN training.")
        else:
            biomass_df = load_biomass_data(biomass_csv_path, biomass_img_dir, multi_angle=True)
            if biomass_df is not None and not biomass_df.empty:
                train_biomass_cnn_regressor(biomass_df, biomass_cnn_model_path, target='fresh_weight', epochs=config['training']['epochs'])
                print("[!] Biomass CNN Regressor training complete.\n")
            else:
                print("[!] Failed to load Biomass dataset. Skipping Biomass CNN training.")

    if train_kmeans or not os.path.exists(kmeans_model_path):
        print("\n[!] K-Means health cluster model not found or retrain forced. Training...")
        # Use Danforth CSV as the primary clustering data source.
        # train_health_clusters gracefully handles missing/mismatched columns.
        if os.path.exists(danforth_csv_path):
            train_health_clusters(danforth_csv_path, kmeans_model_path, n_clusters=3)
            print("[!] K-Means clustering complete.\n")
        else:
            print("[!] Danforth CSV not found — K-Means training skipped. Provide data at:")
            print(f"    {danforth_csv_path}")


if __name__ == "__main__":
    main()