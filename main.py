import time
import os
import glob
import json
import pandas as pd
from inference_engine import load_models, analyze_plant_status, log_to_csv
from model_builder import train_and_save_cnn, train_and_save_rf # Now active!

# --- CONFIGURATION LOADING ---
try:
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    print("[!] ERROR: config.json not found.")
    exit(1)

TRAIN_MODEL = config['training']['force_retrain']

cnn_model_path = config['paths']['cnn_model']
rf_model_path = config['paths']['rf_model']
csv_log_path = config['paths']['csv_log']
bellwether_dir = config['paths']['bellwether_images_dir']


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


def main():
    print("Starting Demeter Software Pipeline...")
    
    # 1. LOAD METADATA
    metadata_df = load_bellwether_metadata(bellwether_dir)
    if metadata_df is None or metadata_df.empty:
        print("Failed to load metadata. Exiting.")
        return
        
    print(f"Successfully linked {len(metadata_df)} image records with plant data.\n")

    # --- TRAINING CHECK ---
    # Now correctly passes the pandas dataframe and directory to the updated builder functions
    if not os.path.exists(cnn_model_path) or not os.path.exists(rf_model_path) or TRAIN_MODEL:
        print("\n[!] Models not found or retrain forced. Initiating training sequence...")
        
        if not os.path.exists(cnn_model_path) or TRAIN_MODEL:
            train_and_save_cnn(metadata_df, bellwether_dir, cnn_model_path, epochs=config['training']['epochs'])
            
        if not os.path.exists(rf_model_path) or TRAIN_MODEL:
            train_and_save_rf(metadata_df, rf_model_path)
            
        print("[!] Training complete.\n")

    # --- INFERENCE PIPELINE ---
    print("Loading AI Models...")
    cnn_model, rf_model = load_models(cnn_model_path, rf_model_path) 
    print("System Online. Generating diagnoses...\n")
    
    # Let's test the first 5 visible spectrum images from the dataframe
    vis_images = metadata_df[metadata_df['spectrum'].str.contains('Visible', na=False, case=False)].head(5)
    
    for index, row in vis_images.iterrows():
        # Construct the absolute path to the image
        snapshot_folder = f"snapshot{row['parent snapshot id']}"
        image_filename = f"{row['name']}.jpg"
        
        img_path = os.path.join(bellwether_dir, snapshot_folder, image_filename)
        
        # Check if the constructed path actually exists on the drive
        if not os.path.exists(img_path):
            print(f"[!] Warning: Image not found at {img_path}")
            continue
            
        # Extract the real data from the CSV
        water_amount = row['water amount']
        weight_before = row['weight before']
        plant_barcode = row['plant barcode']
        
        print(f"Test Plant: {plant_barcode} | Image: {image_filename}")
        print(f"Actual Data -> Water Applied: {water_amount}ml | Weight Before: {weight_before}g")
        
        # --- PASS TO INFERENCE ENGINE ---
        # Note: I updated the class_names to match the new binary setup in model_builder.py
        try:
            diagnosis = analyze_plant_status(
                image_path=img_path, 
                water_amount=water_amount,      
                weight=weight_before,           
                cnn_model=cnn_model, 
                rf_model=rf_model, 
                class_names=['Water_Stressed', 'Well_Watered'] 
            )
            
            # Print output (Assuming your inference_engine still returns a dictionary like this)
            print(f"Predicted Status -> Vision: {diagnosis.get('Species', 'N/A')} | "
                  f"Needs Water (RF): {diagnosis.get('Needs_Water', 'N/A')}")
            
            log_to_csv(diagnosis, filepath=csv_log_path)
            
        except Exception as e:
            print(f"[!] Inference engine encountered an error: {e}")
            print("You may need to update 'inference_engine.py' to accept 'water_amount' and 'weight' kwargs instead of 'temp', 'moisture', 'light'.")
            
        print("-" * 40)
        time.sleep(0.5)

if __name__ == "__main__":
    main()