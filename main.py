import time
import os
import glob
import json
import pickle
import pandas as pd
from inference_engine import load_models, analyze_plant_status, log_to_csv, diagnose_plant_disease, predict_growth_milestone
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
    
    # ==========================================
    # 1. PlantVillage Disease Classification Demo
    # ==========================================
    print("=" * 60)
    print("PLANTVILLAGE DISEASE CLASSIFICATION TEST")
    print("=" * 60)
    
    # Scan PlantVillage for test images
    test_count = 0
    max_tests = 5
    
    for disease_class in sorted(os.listdir(plantvillage_dir)):
        if test_count >= max_tests:
            break
            
        class_path = os.path.join(plantvillage_dir, disease_class)
        if not os.path.isdir(class_path):
            continue
        
        # Get first image from this disease class
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if image_files:
            image_file = image_files[0]
            image_path = os.path.join(class_path, image_file)
            
            try:
                # Get class names from PlantVillage model
                from model_builder import train_and_save_cnn_plantvillage
                class_dirs = sorted([d for d in os.listdir(plantvillage_dir) 
                                   if os.path.isdir(os.path.join(plantvillage_dir, d))])
                
                diagnosis = diagnose_plant_disease(image_path, cnn_model, class_dirs)
                
                print(f"\nTest {test_count + 1}: {disease_class}")
                print(f"Image: {image_file}")
                print(f"Detected Disease: {diagnosis['Detected_Disease']} (Confidence: {diagnosis['Disease_Confidence']:.2%})")
                
                log_to_csv(diagnosis, filepath=csv_log_path)
                test_count += 1
                
            except Exception as e:
                print(f"[!] Error processing {image_file}: {e}")
    
    print("\n" + "=" * 60)
    print("DANFORTH GROWTH PREDICTION TEST")
    print("=" * 60)
    
    # ==========================================
    # 2. Danforth Growth Prediction Demo
    # ==========================================
    # Sample environmental conditions for testing
    sample_conditions = [
        {
            "Soil_Type": 1,  # encoded
            "Sunlight_Hours": 6.0,
            "Water_Frequency": 2,  # encoded
            "Fertilizer_Type": 1,  # encoded
            "Temperature": 25.0,
            "Humidity": 65.0
        },
        {
            "Soil_Type": 0,  # encoded
            "Sunlight_Hours": 4.0,
            "Water_Frequency": 0,  # encoded
            "Fertilizer_Type": 0,  # encoded
            "Temperature": 20.0,
            "Humidity": 55.0
        }
    ]
    
    for idx, env_data in enumerate(sample_conditions):
        try:
            growth_pred = predict_growth_milestone(env_data, rf_model)
            
            print(f"\nCondition Set {idx + 1}:")
            print(f"  Temperature: {env_data['Temperature']}°C, Humidity: {env_data['Humidity']}%")
            print(f"  Sunlight: {env_data['Sunlight_Hours']} hours")
            print(f"  Predicted Growth Milestone: {growth_pred['Predicted_Growth_Milestone']:.4f}")
            
            log_to_csv(growth_pred, filepath=csv_log_path)
            
        except Exception as e:
            print(f"[!] Error with growth prediction: {e}")
    
    print("\n" + "=" * 60)
    print("[✓] Inference pipeline complete. Results logged to CSV.")
    print("=" * 60)

if __name__ == "__main__":
    main()