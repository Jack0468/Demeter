import os
import csv
import numpy as np
import tensorflow as tf
from datetime import datetime

# Define plant classes matching your dataset
CLASS_NAMES = ['Tomato', 'Succulent', 'Fern', 'Orchid'] 

def load_vision_model(model_path):
    """Loads the pre-trained CNN model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}.")
    return tf.keras.models.load_model(model_path)

def determine_needs(species, temp, moisture, light):
    """
    Evaluates environmental data against species-specific requirements.
    Returns a dictionary of Yes/No actions.
    """
    # Initialize defaults
    needs = {
        "Needs_Water": "No",
        "Needs_Fertilizer": "No", 
        "Needs_More_Sunlight": "No"
    }
    
    # Conceptual logic (Replace with your Random Forest model later if desired)
    if species == "Tomato":
        if moisture < 40.0: needs["Needs_Water"] = "Yes"
        if light < 5000:    needs["Needs_More_Sunlight"] = "Yes"
        # Example condition for fertilizer
        if temp > 20.0 and moisture > 50.0: needs["Needs_Fertilizer"] = "Yes" 
            
    elif species == "Succulent":
        if moisture < 10.0: needs["Needs_Water"] = "Yes"
        if light < 2000:    needs["Needs_More_Sunlight"] = "Yes"

    return needs

def analyze_plant_status(image_path, temp, moisture, light, cnn_model):
    """Runs the full pipeline: Image classification + Environmental logic."""
    # 1. Vision Classification
    img = tf.keras.utils.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = cnn_model.predict(img_array, verbose=0)
    species_idx = np.argmax(predictions[0])
    detected_species = CLASS_NAMES[species_idx]
    confidence = float(predictions[0][species_idx])
    
    # 2. Determine Needs (Yes/No variables)
    action_plan = determine_needs(detected_species, temp, moisture, light)
    
    # 3. Compile the final data package
    result = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Species": detected_species,
        "Confidence": round(confidence, 3),
        "Temp_C": temp,
        "Moisture_%": moisture,
        "Light_lux": light
    }
    # Merge the action plan into the result dictionary
    result.update(action_plan) 
    
    return result

def log_to_csv(data_dict, filepath="demeter_logs.csv"):
    """Appends the result dictionary to a CSV file."""
    file_exists = os.path.isfile(filepath)
    
    with open(filepath, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data_dict.keys())
        
        # Write the header only if the file is being created for the first time
        if not file_exists:
            writer.writeheader()
            
        writer.writerow(data_dict)
