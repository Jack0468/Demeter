"""
Inference Engine for Demeter

This module provides the core inference pipeline for the Demeter project.
It includes helper functions to:
- load pre-trained CNN and Random Forest models,
- classify plant disease from uploaded images,
- predict plant growth milestones from environmental data,
- generate a complete diagnosis payload for dashboard integration,
- log inference outputs to CSV for history tracking.

The file also contains a legacy analysis routine for a previous water-stress
prediction workflow.
"""

import os
import csv
import json
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from datetime import datetime
from pathlib import Path

# --- DYNAMIC PROJECT ROOT RESOLUTION ---
# Ensures paths like 'data/...' always resolve to the Demeter root folder,
# regardless of whether this script is currently in the root or moved to src/core/
_current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _current_dir if _current_dir.name == "Demeter" else _current_dir.parent.parent

# Import our new modules
try:
    from src.core.output_formatter import OutputFormatter
    from src.core.status_engine import StatusEngine
except ModuleNotFoundError:
    from output_formatter import OutputFormatter
    from status_engine import StatusEngine

def load_models(cnn_path, rf_path):
    """Loads both the pre-trained CNN and Random Forest models."""
    if not os.path.exists(cnn_path) or not os.path.exists(rf_path):
        raise FileNotFoundError("Models not found. Training required.")
    
    cnn_model = tf.keras.models.load_model(cnn_path)
    rf_model = joblib.load(rf_path)
    return cnn_model, rf_model

# ==========================================
# PLANTVILLAGE DISEASE CLASSIFICATION
# ==========================================
def diagnose_plant_disease(image_path, cnn_model, class_names):
    """
    Classifies plant disease from image using PlantVillage CNN.
    
    Args:
        image_path: Path to plant image
        cnn_model: Trained CNN model
        class_names: List of disease class names
        
    Returns:
        Dictionary with disease prediction and confidence
    """
    img = tf.keras.utils.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = cnn_model.predict(img_array, verbose=0)
    disease_idx = np.argmax(predictions[0])
    detected_disease = class_names[disease_idx]
    confidence = float(predictions[0][disease_idx])
    
    result = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Image_Path": image_path,
        "Detected_Disease": detected_disease,
        "Disease_Confidence": round(confidence, 4),
        "All_Predictions": {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
    }
    
    return result

# ==========================================
# DANFORTH GROWTH PREDICTION
# ==========================================
def predict_growth_milestone(environmental_data, rf_model):
    """
    Predicts plant growth milestone from environmental sensor data using Danforth RF.
    
    Args:
        environmental_data: Dict with keys: Soil_Type, Sunlight_Hours, Water_Frequency,
                           Fertilizer_Type, Temperature, Humidity (encoded as integers)
        rf_model: Trained Random Forest Regressor
        
    Returns:
        Dictionary with growth prediction
    """
    # Expected feature order from training: 
    # ['Soil_Type', 'Sunlight_Hours', 'Water_Frequency', 'Fertilizer_Type', 'Temperature', 'Humidity']
    feature_order = ['Soil_Type', 'Sunlight_Hours', 'Water_Frequency', 'Fertilizer_Type', 'Temperature', 'Humidity']
    features = pd.DataFrame(
        [[environmental_data.get(feat, 0) for feat in feature_order]],
        columns=feature_order
    )
    
    growth_prediction = float(rf_model.predict(features)[0])
    
    result = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Predicted_Growth_Milestone": round(growth_prediction, 4),
        "Environmental_Input": environmental_data
    }
    
    return result

# ==========================================
# ENHANCED: INTEGRATED DIAGNOSIS WITH STATUS ENGINE
# ==========================================
def generate_complete_diagnosis(
    image_path,
    detected_disease,
    disease_confidence,
    all_predictions,
    predicted_growth,
    temperature,
    soil_moisture,
    sunlight_hours,
    humidity=None
):
    """
    Generate a complete, dashboard-ready diagnosis combining all components.
    
    Args:
        image_path: Path to plant image
        detected_disease: Disease detected by CNN
        disease_confidence: Confidence score (0-1)
        all_predictions: Dict of all class predictions
        predicted_growth: Growth prediction from RF
        temperature: Temperature in Celsius
        soil_moisture: Soil moisture percentage (0-100)
        sunlight_hours: Sunlight hours per day
        humidity: Optional humidity percentage
        
    Returns:
        Complete diagnosis dict with status, recommendations, and system commands
    """
    # Initialize components
    formatter = OutputFormatter()
    status_engine = StatusEngine()
    
    # Format disease detection
    disease_result = OutputFormatter.format_disease_detection(
        image_path, detected_disease, disease_confidence, all_predictions
    )
    
    # Format growth prediction
    growth_result = OutputFormatter.format_growth_prediction(
        predicted_growth, 
        {"temperature": temperature, "humidity": humidity}
    )
    
    # Format sensors
    sensor_data = OutputFormatter.format_sensor_data(
        temperature, soil_moisture, sunlight_hours, humidity
    )
    
    # Merge formatted data
    merged = formatter.merge_diagnosis(
        disease_result, growth_result, sensor_data
    )
    
    # Generate status and recommendations
    full_diagnosis = status_engine.generate_full_diagnosis(
        disease_confidence, detected_disease, soil_moisture,
        temperature, sunlight_hours, predicted_growth, humidity
    )
    
    # Merge everything
    diagnosis = {**merged, **full_diagnosis}
    
    # Save outputs
    formatter.save_latest(diagnosis)
    formatter.append_history(diagnosis)
    
    return diagnosis

# ==========================================
# LEGACY: BELLWETHER WATER STRESS ANALYSIS
# ==========================================
def analyze_plant_status(image_path, water_amount, weight, cnn_model, rf_model, class_names):
    """Runs the full pipeline: Image classification + Random Forest logic."""
    
    # ==========================================
    # 1. Vision Classification (CNN)
    # ==========================================
    img = tf.keras.utils.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = cnn_model.predict(img_array, verbose=0)
    species_idx = np.argmax(predictions[0])
    detected_status_vision = class_names[species_idx] # e.g., 'Water_Stressed'
    confidence = float(predictions[0][species_idx])
    
    # ==========================================
    # 2. Determine Needs (Random Forest)
    # ==========================================
    # The RF was trained in model_builder.py on ['weight before', 'water amount']
    # to predict future growth (weight after).
    sensor_data = np.array([[weight, water_amount]])
    
    rf_prediction = rf_model.predict(sensor_data)[0] 
    
    # Generate the Action Plan
    action_plan = {
        "Needs_Fertilizer": "Yes" if rf_prediction < 0.9 * weight else "No",
        "Vision_Status": detected_status_vision,
        "Predicted_Future_Weight": round(float(rf_prediction), 2)
    }
    
    # ==========================================
    # 3. Compile the final data package
    # ==========================================
    result = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Species": "Setaria", # We know the species for this dataset
        "Confidence": round(confidence, 3),
        "Weight_g": weight,
        "Water_Applied_ml": water_amount
    }
    result.update(action_plan) 
    return result

def log_to_csv(data_dict, filepath="data/demeter_logs.csv"):
    """
    Appends the result dictionary to a CSV file.
    
    Enhanced to handle nested dictionaries (e.g., All_Predictions, Environmental_Input)
    by serializing them as JSON strings.
    """
    abs_filepath = PROJECT_ROOT / filepath
    os.makedirs(abs_filepath.parent, exist_ok=True)
    
    # Flatten nested dictionaries for CSV
    flat_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, dict):
            flat_dict[key] = json.dumps(value)
        elif isinstance(value, list):
            flat_dict[key] = json.dumps(value)
        else:
            flat_dict[key] = value
    
    file_exists = abs_filepath.is_file()
    with open(abs_filepath, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=flat_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(flat_dict)