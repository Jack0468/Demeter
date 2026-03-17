import os
import csv
import numpy as np
import tensorflow as tf
import joblib
from datetime import datetime

def load_models(cnn_path, rf_path):
    """Loads both the pre-trained CNN and Random Forest models."""
    if not os.path.exists(cnn_path) or not os.path.exists(rf_path):
        raise FileNotFoundError("Models not found. Training required.")
    
    cnn_model = tf.keras.models.load_model(cnn_path)
    rf_model = joblib.load(rf_path)
    return cnn_model, rf_model

def analyze_plant_status(image_path, temp, moisture, light, cnn_model, rf_model, class_names):
    """Runs the full pipeline: Image classification + Random Forest logic."""
    
    # 1. Vision Classification (CNN)
    img = tf.keras.utils.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = cnn_model.predict(img_array, verbose=0)
    species_idx = np.argmax(predictions[0])
    detected_species = class_names[species_idx]
    confidence = float(predictions[0][species_idx])
    
    # 2. Determine Needs (Random Forest)
    # We format the input exactly as the RF model saw it during training
    # Note: species_idx represents the 'Species_Code' here
    sensor_data = np.array([[species_idx, temp, moisture, light]])
    
    # The RF outputs a 2D array of predictions (e.g., [1, 0, 1] for Yes, No, Yes)
    rf_predictions = rf_model.predict(sensor_data)[0] 
    
    # Map binary output back to Yes/No
    action_plan = {
        "Needs_Water": "Yes" if rf_predictions[0] == 1 else "No",
        "Needs_Fertilizer": "Yes" if rf_predictions[1] == 1 else "No",
        "Needs_More_Sunlight": "Yes" if rf_predictions[2] == 1 else "No"
    }
    
    # 3. Compile the final data package
    result = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Species": detected_species,
        "Confidence": round(confidence, 3),
        "Temp_C": temp,
        "Moisture_%": moisture,
        "Light_lux": light
    }
    result.update(action_plan) 
    
    return result

def log_to_csv(data_dict, filepath="data/demeter_logs.csv"):
    """Appends the result dictionary to a CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data_dict)
