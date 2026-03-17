def generate_system_recommendation(image_path, current_temp, current_moisture, cnn_model, class_names):
    """
    1. Identifies the plant.
    2. Applies species-specific logic to sensor data.
    """
    
    # --- STEP A: IDENTIFY THE PLANT ---
    img = tf.keras.utils.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = cnn_model.predict(img_array)
    species_idx = np.argmax(predictions[0])
    detected_species = class_names[species_idx]
    
    # --- STEP B: SPECIES-SPECIFIC SENSOR LOGIC ---
    # In a full system, this logic would come from your trained Random Forest 
    # or a lookup database. Here is the conceptual routing:
    
    action = "Optimal"
    
    if detected_species == "Tomato":
        if current_moisture < 40.0:
            action = "ACTIVATE_WATER_PUMP"
        elif current_temp > 30.0:
            action = "ACTIVATE_COOLING_FAN"
            
    elif detected_species == "Succulent":
        if current_moisture < 10.0:
            action = "ACTIVATE_WATER_PUMP"
            
    # --- STEP C: OUTPUT ---
    print(f"[{detected_species} Detected] Confidence: {predictions[0][species_idx]:.2f}")
    print(f"Current Sensors -> Temp: {current_temp}C | Moisture: {current_moisture}%")
    print(f"System Command  -> {action}")
    
    return action
