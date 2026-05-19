import sys

with open('src/api/api_server.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update predict() route
target_predict = """    try:
        from src.core.inference_engine import (
            diagnose_plant_disease, predict_growth_milestone, generate_complete_diagnosis,
            predict_biomass, predict_tiller_count, analyze_plant_status
        )
        
        # 1. Main PlantVillage + Danforth Diagnosis
        disease = diagnose_plant_disease(
            image_path, _model_state["cnn"], _model_state["class_dirs"]
        )
        env_data = {
            "Soil_Type": 1,
            "Sunlight_Hours": sunlight_hours,
            "Water_Frequency": 2,
            "Fertilizer_Type": 1,
            "Temperature": temperature,
            "Humidity": humidity
        }
        growth = predict_growth_milestone(env_data, _model_state["rf"])
        
        all_preds = disease.get("All_Predictions", {})
        
        # 2. Biomass
        if _model_state["cnn_biomass"]:
            biomass_pred = predict_biomass(image_path, _model_state["cnn_biomass"])
            all_preds["biomass_result"] = biomass_pred
            
        # 3. Tiller
        if _model_state["cnn_tiller"]:
            tiller_pred = predict_tiller_count(image_path, _model_state["cnn_tiller"])
            all_preds["tiller_result"] = tiller_pred
            
        # 4. Bellwether Water Stress
        if _model_state["cnn_water"] and _model_state["rf_water"]:
            bellwether_result = analyze_plant_status(
                image_path, 
                water_amount=soil_moisture,
                weight=all_preds.get("biomass_result", 50.0), 
                cnn_model=_model_state["cnn_water"],
                rf_model=_model_state["rf_water"],
                class_names=["Water_Stressed", "Well_Watered"]
            )
            all_preds["bellwether_result"] = bellwether_result

        diagnosis = generate_complete_diagnosis(
            image_path=image_path,
            detected_disease=disease["Detected_Disease"],
            disease_confidence=disease["Disease_Confidence"],
            all_predictions=all_preds,
            predicted_growth=growth["Predicted_Growth_Milestone"],
            temperature=temperature,
            soil_moisture=soil_moisture,
            sunlight_hours=sunlight_hours,
            humidity=humidity
        )
        return jsonify(diagnosis), 200"""

repl_predict = """    try:
        from src.core.inference_engine import (
            diagnose_plant_disease, predict_growth_milestone, generate_complete_diagnosis,
            predict_biomass, predict_tiller_count, analyze_plant_status
        )
        import tensorflow as tf
        from concurrent.futures import ThreadPoolExecutor

        # Process image ONCE into a NumPy array
        img = tf.keras.utils.load_img(image_path, target_size=(150, 150))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        
        env_data = {
            "Soil_Type": 1,
            "Sunlight_Hours": sunlight_hours,
            "Water_Frequency": 2,
            "Fertilizer_Type": 1,
            "Temperature": temperature,
            "Humidity": humidity
        }

        # Setup parallel inference tasks
        with ThreadPoolExecutor() as executor:
            future_disease = executor.submit(diagnose_plant_disease, img_array, image_path, _model_state["cnn"], _model_state["class_dirs"])
            future_growth = executor.submit(predict_growth_milestone, env_data, _model_state["rf"])
            
            future_biomass = None
            if _model_state["cnn_biomass"]:
                future_biomass = executor.submit(predict_biomass, img_array, _model_state["cnn_biomass"])
                
            future_tiller = None
            if _model_state["cnn_tiller"]:
                future_tiller = executor.submit(predict_tiller_count, img_array, _model_state["cnn_tiller"])
            
            # Retrieve basic results first
            disease = future_disease.result()
            growth = future_growth.result()
            
            all_preds = disease.get("All_Predictions", {})
            
            # Biomass
            biomass_val = 50.0
            if future_biomass:
                biomass_val = future_biomass.result()
                all_preds["biomass_result"] = biomass_val
                
            # Tiller
            if future_tiller:
                all_preds["tiller_result"] = future_tiller.result()

            # Water Stress (needs biomass_val, so we run it after biomass completes, or wait for biomass)
            # Alternatively, we could have run it concurrently if we estimated biomass_val, but waiting is safer.
            if _model_state["cnn_water"] and _model_state["rf_water"]:
                bellwether_result = analyze_plant_status(
                    img_array, 
                    water_amount=soil_moisture,
                    weight=biomass_val, 
                    cnn_model=_model_state["cnn_water"],
                    rf_model=_model_state["rf_water"],
                    class_names=["Water_Stressed", "Well_Watered"]
                )
                all_preds["bellwether_result"] = bellwether_result

        diagnosis = generate_complete_diagnosis(
            image_path=image_path,
            detected_disease=disease["Detected_Disease"],
            disease_confidence=disease["Disease_Confidence"],
            all_predictions=all_preds,
            predicted_growth=growth["Predicted_Growth_Milestone"],
            temperature=temperature,
            soil_moisture=soil_moisture,
            sunlight_hours=sunlight_hours,
            humidity=humidity
        )
        return jsonify(diagnosis), 200"""

if target_predict in text:
    text = text.replace(target_predict, repl_predict)
else:
    print('Failed to find predict route target')

# 2. Update to call _ensure_models_loaded() on startup
target_startup = """    os.makedirs(OUTPUT_DIR, exist_ok=True)
    app.run(debug=False, host="0.0.0.0", port=5000)"""

repl_startup = """    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\\n[Demeter] Pre-loading models for faster inference... This will take a moment.")
    _ensure_models_loaded()
    print("[Demeter] Models loaded! Ready for traffic.")
    app.run(debug=False, host="0.0.0.0", port=5000)"""

if target_startup in text:
    text = text.replace(target_startup, repl_startup)
else:
    print('Failed to find startup target')

with open('src/api/api_server.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('api_server.py updated successfully!')
