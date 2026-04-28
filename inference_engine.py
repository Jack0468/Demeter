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
    features = np.array([[environmental_data.get(feat, 0) for feat in feature_order]])
    
    growth_prediction = float(rf_model.predict(features)[0])
    
    result = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Predicted_Growth_Milestone": round(growth_prediction, 4),
        "Environmental_Input": environmental_data
    }
    
    return result

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
        "Needs_Water": "Yes" if rf_prediction == 1 else "No",
        # We can still derive secondary inferences based on the data
        "Needs_Fertilizer": "Yes" if weight < 450 else "No", # Arbitrary threshold for Demeter prototype
        "Vision_Status": detected_status_vision
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
    """Appends the result dictionary to a CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data_dict)