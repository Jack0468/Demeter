import os

# Suppress TensorFlow GPU/CPU info warnings for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from pathlib import Path

# --- DYNAMIC PROJECT ROOT RESOLUTION ---
_current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _current_dir.parent if _current_dir.name == "scripts" else _current_dir

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def generate_bootstrap_data(num_samples=5000):
    """
    Bootstraps SVM training data using actual predictions from the 
    trained CNN and Random Forest models on our local datasets.
    """
    print(f"Generating {num_samples} bootstrap data points from actual model outputs...")
    
    # 1. Load config
    config_path = PROJECT_ROOT / 'config.json'
    if not config_path.exists():
        config_path = PROJECT_ROOT / 'config' / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # 2. Extract Real Environmental Distributions (Danforth + RF Model)
    danforth_csv = PROJECT_ROOT / config['paths']['danforth_csv_path']
    danforth_model_path = PROJECT_ROOT / config['paths']['danforth_rf_model_path']
    
    print(f"Loading Danforth data and predicting growth via RF...")
    df_env = pd.read_csv(danforth_csv).dropna()
    rf_model = joblib.load(danforth_model_path)
    
    X_rf = df_env.drop(columns=['Growth_Milestone'], errors='ignore')
    real_growth_preds = rf_model.predict(X_rf)
    
    real_temps = df_env.get('Temperature', df_env.get('Temp')).values
    real_moistures = df_env.get('Soil_Moisture', df_env.get('Moisture')).values
    real_lights = df_env.get('Sunlight_Hours', df_env.get('Light')).values

    # 3. Extract Real Visual Confidences (PlantVillage + CNN Model)
    pv_dir = PROJECT_ROOT / config['paths']['plantvillage_dir']
    pv_model_path = PROJECT_ROOT / config['paths']['plantvillage_cnn_model_path']
    
    print(f"Sampling images from PlantVillage and extracting CNN confidences...")
    cnn_model = tf.keras.models.load_model(pv_model_path)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        pv_dir, image_size=(150, 150), batch_size=32, shuffle=True, seed=42
    )
    class_names = test_ds.class_names
    
    real_confidences = []
    is_healthy_list = []
    
    batches_to_sample = min(60, len(test_ds)) # Sample ~2000 images to get a solid distribution
    for i, (imgs, labels) in enumerate(test_ds.take(batches_to_sample)):
        print(f"Processing CNN batch {i+1}/{batches_to_sample}...", end='\r')
        preds = cnn_model.predict(imgs, verbose=0)
        real_confidences.extend(np.max(preds, axis=1)) # Max probability is the confidence
        # Extract the ACTUAL ground-truth label from the dataset!
        for label in labels:
            is_healthy_list.append("healthy" in class_names[label.numpy()].lower())
    print("\nCNN extraction complete.")
    real_confidences = np.array(real_confidences)
    is_healthy_list = np.array(is_healthy_list)
    
    np.random.seed(42)
    
    # 4. Cross-sample to simulate holistic plant profiles
    print("Merging real distributions into plant profiles...")
    idx_env = np.random.choice(len(real_growth_preds), size=num_samples, replace=True)
    idx_cnn = np.random.choice(len(real_confidences), size=num_samples, replace=True)
    
    disease_confidence = real_confidences[idx_cnn]
    is_healthy = is_healthy_list[idx_cnn]
    predicted_growth = real_growth_preds[idx_env]
    temperature = real_temps[idx_env]
    soil_moisture = real_moistures[idx_env]
    sunlight_hours = real_lights[idx_env]
    
    median_growth = np.median(real_growth_preds)
    
    health_scores = []
    overall_statuses = []
    
    print("Labeling data using actual dataset ground-truths...")
    for i in range(num_samples):
        # Ground truth matrix
        healthy_vis = is_healthy[i]
        good_env = predicted_growth[i] >= median_growth
        
        if healthy_vis and good_env:
            status = "Thriving"
            score = np.random.uniform(80, 100)
        elif healthy_vis and not good_env:
            status = "Struggling"
            score = np.random.uniform(50, 79)
        elif not healthy_vis and good_env:
            status = "Struggling"
            score = np.random.uniform(40, 69)
        else:  # Visually diseased and poor environment
            status = "Critical"
            score = np.random.uniform(10, 39)
            
        health_scores.append(int(score))
        overall_statuses.append(status)
        
    df = pd.DataFrame({
        'Disease_Confidence': disease_confidence,
        'Predicted_Growth': predicted_growth,
        'Temperature': temperature,
        'Soil_Moisture': soil_moisture,
        'Sunlight_Hours': sunlight_hours,
        'Health_Score': health_scores,
        'Overall_Status': overall_statuses
    })
    
    return df

def main():
    out_dir = PROJECT_ROOT / "data" / "processed" / "svm_training"
    os.makedirs(out_dir, exist_ok=True)
    
    out_file = out_dir / "svm_bootstrap_data.csv"
    
    df = generate_bootstrap_data(5000)
    
    # Check if we have real historical data to mix in
    history_file = PROJECT_ROOT / "data" / "logs" / "inference_logs.csv"
    if history_file.exists():
        try:
            print(f"Found historical data at {history_file}. Merging...")
            history_df = pd.read_csv(history_file)
            
            required_cols = ['Disease_Confidence', 'Predicted_Growth', 'Temperature', 'Soil_Moisture', 'Sunlight_Hours', 'Overall_Status']
            
            if all(col in history_df.columns for col in required_cols):
                history_subset = history_df[required_cols].dropna()
                df = pd.concat([df, history_subset], ignore_index=True)
                print(f"Added {len(history_subset)} real historical records.")
            else:
                print("Historical data missing required columns. Skipping merge.")
        except Exception as e:
            print(f"Error reading historical data: {e}")
    
    df.to_csv(out_file, index=False)
    print(f"Successfully saved {len(df)} records to {out_file}")
    print("\n[!] Phase 1 Complete. We are now ready to train the SVM classifier!")

if __name__ == "__main__":
    main()