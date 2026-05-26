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
except ImportError:
    from output_formatter import OutputFormatter

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
def diagnose_plant_disease(img_array, image_path, cnn_model, class_names):
    """
    Classifies plant disease from image array using PlantVillage CNN.
    
    Args:
        img_array: Pre-processed image tensor
        image_path: Original image path for logging
        cnn_model: Trained CNN model
        class_names: List of disease class names
        
    Returns:
        Dictionary with disease prediction and confidence
    """
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
    
    try:
        growth_prediction = float(rf_model.predict(features)[0])
    except Exception as e:
        print(f"[Warning] RF prediction failed in predict_growth_milestone: {e}")
        # Fallback prediction if the model file is corrupted or expects wrong features
        growth_prediction = environmental_data.get("Sunlight_Hours", 6.0) * 2.5    
    result = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Predicted_Growth_Milestone": round(growth_prediction, 4),
        "Environmental_Input": environmental_data
    }
    
    return result

# ==========================================
# ADDITIONAL PREDICTIONS: BIOMASS, TILLER, BELLWETHER
# ==========================================
def predict_biomass(img_array, cnn_model):
    """Predict continuous biomass weight from an image array."""
    prediction = float(cnn_model.predict(img_array, verbose=0)[0][0])
    return round(prediction, 2)

def predict_tiller_count(img_array, cnn_model):
    """Predict continuous tiller count from an image array."""
    prediction = float(cnn_model.predict(img_array, verbose=0)[0][0])
    # Tiller counts are usually integers, but we return a float for precision
    return round(prediction, 2)

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
        all_predictions: Dict of all class predictions (can include biomass, tiller keys)
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
    # Format disease detection
    # Note: we filter out multi-model results from all_predictions for the legacy formatter
    filtered_preds = {k: v for k, v in all_predictions.items() if not k.endswith("_result")}
    disease_result = OutputFormatter.format_disease_detection(
        image_path, detected_disease, disease_confidence, filtered_preds
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
    
    # If we have extra multi-model results provided via kwargs (biomass, tiller, bellwether)
    # we can append them to the diagnosis.
    if "biomass_result" in all_predictions and all_predictions["biomass_result"]:
        merged["biomass_prediction"] = all_predictions["biomass_result"]
    if "tiller_result" in all_predictions and all_predictions["tiller_result"]:
        merged["tiller_prediction"] = all_predictions["tiller_result"]
    if "bellwether_result" in all_predictions and all_predictions["bellwether_result"]:
        merged["bellwether_water_stress"] = all_predictions["bellwether_result"]
    if "hybrid_result" in all_predictions and all_predictions["hybrid_result"]:
        merged["hybrid_prediction"] = all_predictions["hybrid_result"]

    if "visual_cluster" in all_predictions:
        merged["visual_cluster"] = all_predictions["visual_cluster"]
    if "tabular_cluster" in all_predictions:
        merged["tabular_cluster"] = all_predictions["tabular_cluster"]
    if "master_cluster" in all_predictions:
        merged["master_cluster"] = all_predictions["master_cluster"]

    diagnosis = merged
    
    # Save outputs
    formatter.save_latest(diagnosis)
    formatter.append_history(diagnosis)
    
    return diagnosis

# ==========================================
# LEGACY: BELLWETHER WATER STRESS ANALYSIS
# ==========================================
def analyze_plant_status(img_array, water_amount, weight, cnn_model, rf_model, class_names):
    """Runs the full pipeline: Image classification + Random Forest logic."""
    
    # ==========================================
    # 1. Vision Classification (CNN)
    # ==========================================
    predictions = cnn_model.predict(img_array, verbose=0)
    species_idx = np.argmax(predictions[0])
    detected_status_vision = class_names[species_idx] # e.g., 'Water_Stressed'
    confidence = float(predictions[0][species_idx])
    
    # ==========================================
    # 2. Determine Needs (Random Forest)
    # ==========================================
    import pandas as pd
    
    # Create a DataFrame with all columns expected by the newer RF model
    input_data = pd.DataFrame({
        'weight before': [weight],
        'treatment_code': [0],
        'genotype_code': [0],
        'day_of_experiment': [14],
        'population': [0],
        'hour_of_day': [12],
        'water amount': [water_amount],
        'weight_delta': [0.0]
    })
    
    try:
        rf_prediction = rf_model.predict(input_data)[0] 
    except Exception as e:
        print(f"[Warning] RF prediction failed in analyze_plant_status: {e}")
        # Fallback to a generic estimation if model features changed again
        rf_prediction = weight * 1.05
    
    # Generate the Action Plan
    action_plan = {
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

def log_to_csv(data_dict, filepath="data/logs/inference_logs.csv"):
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


# ==========================================
# HYBRID SVM DISEASE CLASSIFICATION
# ==========================================
def extract_hybrid_fft_features(img_path):
    """Isolate the leaf, extract raw Grayscale FFT magnitude and HSV Color Histogram."""
    import cv2
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Could not load {img_path}")
    
    IMG_SIZE = 64
    img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    
    # Otsu thresholding
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 2D FFT Magnitude Spectrum
    fft_gray = np.fft.fft2(gray)
    mag_gray = 20 * np.log(np.abs(np.fft.fftshift(fft_gray)) + 1).flatten()
    
    # HSV Histogram on isolated leaf region
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], mask, [32], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], mask, [32], [0, 256]).flatten()
    h_hist /= h_hist.sum() + 1e-6
    s_hist /= s_hist.sum() + 1e-6
    color_hist = np.concatenate([h_hist, s_hist])
    
    return mag_gray, color_hist

def predict_hybrid_disease(img_path, scaler_fft, scaler_hist, pca_fft, svm, class_names):
    """
    Classifies plant disease using the high-performance Hybrid (Raw FFT + HSV Histogram) SVM.
    
    Returns:
        Dict: predicted disease name, confidence score, and all probability scores.
    """
    mag_gray, color_hist = extract_hybrid_fft_features(img_path)
    
    X_fft_scaled = scaler_fft.transform([mag_gray])
    X_fft_pca = pca_fft.transform(X_fft_scaled)
    X_hist_scaled = scaler_hist.transform([color_hist])
    X_hybrid = np.hstack([X_fft_pca, X_hist_scaled])
    
    # Predict probability
    probs = svm.predict_proba(X_hybrid)[0]
    pred_idx = np.argmax(probs)
    detected_disease = class_names[pred_idx]
    confidence = float(probs[pred_idx])
    
    return {
        "primary_disease": detected_disease,
        "confidence": round(confidence, 4),
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    }