import sys

with open('src/api/api_server.py', 'r', encoding='utf-8') as f:
    text = f.read()

target1 = """_model_state = {
    "loaded": False,
    "cnn": None,
    "rf": None,
    "class_dirs": [],
    "error": None
}"""
repl1 = """_model_state = {
    "loaded": False,
    "cnn": None,
    "rf": None,
    "cnn_biomass": None,
    "cnn_tiller": None,
    "cnn_water": None,
    "rf_water": None,
    "class_dirs": [],
    "error": None
}"""

if target1 in text:
    text = text.replace(target1, repl1)
else:
    print('Failed to find target1')

target2 = """    try:
        import tensorflow as tf
        import joblib
        cnn_path = str(PROJECT_ROOT / "models/demeter_cnn_plantvillage.keras")
        rf_path = str(PROJECT_ROOT / "models/demeter_rf_danforth.joblib")
        if os.path.exists(cnn_path):
            _model_state["cnn"] = tf.keras.models.load_model(cnn_path)
        if os.path.exists(rf_path):
            _model_state["rf"] = joblib.load(rf_path)
        plantvillage_dir = str(PROJECT_ROOT / "data/layer2_health_rgb/PlantVillage")
        if os.path.exists(plantvillage_dir):
            _model_state["class_dirs"] = sorted([
                d for d in os.listdir(plantvillage_dir)
                if os.path.isdir(os.path.join(plantvillage_dir, d))
            ])
        _model_state["loaded"] = True
        _model_state["error"] = None"""
repl2 = """    try:
        import tensorflow as tf
        import joblib
        from src.core.plantvillage_classes import PLANTVILLAGE_CLASSES
        
        cnn_path = str(PROJECT_ROOT / "models/demeter_cnn_plantvillage.keras")
        rf_path = str(PROJECT_ROOT / "models/demeter_rf_danforth.joblib")
        cnn_biomass_path = str(PROJECT_ROOT / "models/demeter_cnn_biomass.keras")
        cnn_tiller_path = str(PROJECT_ROOT / "models/demeter_cnn_tiller.keras")
        cnn_water_path = str(PROJECT_ROOT / "models/demeter_cnn.keras")
        rf_water_path = str(PROJECT_ROOT / "models/demeter_rf.joblib")
        
        if os.path.exists(cnn_path):
            _model_state["cnn"] = tf.keras.models.load_model(cnn_path)
        if os.path.exists(rf_path):
            _model_state["rf"] = joblib.load(rf_path)
            
        if os.path.exists(cnn_biomass_path):
            _model_state["cnn_biomass"] = tf.keras.models.load_model(cnn_biomass_path)
        if os.path.exists(cnn_tiller_path):
            _model_state["cnn_tiller"] = tf.keras.models.load_model(cnn_tiller_path)
        if os.path.exists(cnn_water_path):
            _model_state["cnn_water"] = tf.keras.models.load_model(cnn_water_path)
        if os.path.exists(rf_water_path):
            _model_state["rf_water"] = joblib.load(rf_water_path)

        plantvillage_dir = str(PROJECT_ROOT / "data/layer2_health_rgb/PlantVillage")
        if os.path.exists(plantvillage_dir):
            _model_state["class_dirs"] = sorted([
                d for d in os.listdir(plantvillage_dir)
                if os.path.isdir(os.path.join(plantvillage_dir, d))
            ])
        if not _model_state["class_dirs"]:
            _model_state["class_dirs"] = PLANTVILLAGE_CLASSES
            
        _model_state["loaded"] = True
        _model_state["error"] = None"""

if target2 in text:
    text = text.replace(target2, repl2)
else:
    print('Failed to find target2')

target3 = """    try:
        from src.core.inference_engine import (
            diagnose_plant_disease, predict_growth_milestone, generate_complete_diagnosis
        )
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
        
        diagnosis = generate_complete_diagnosis(
            image_path=image_path,
            detected_disease=disease["Detected_Disease"],
            disease_confidence=disease["Disease_Confidence"],
            all_predictions=disease.get("All_Predictions", {}),
            predicted_growth=growth["Predicted_Growth_Milestone"],
            temperature=temperature,
            soil_moisture=soil_moisture,
            sunlight_hours=sunlight_hours,
            humidity=humidity
        )"""

repl3 = """    try:
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
        )"""

if target3 in text:
    text = text.replace(target3, repl3)
else:
    # Try alternate target3 because of the `get("All_Predictions")` difference
    target3_alt = """    try:
        from src.core.inference_engine import (
            diagnose_plant_disease, predict_growth_milestone, generate_complete_diagnosis
        )
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
        
        diagnosis = generate_complete_diagnosis(
            image_path=image_path,
            detected_disease=disease["Detected_Disease"],
            disease_confidence=disease["Disease_Confidence"],
            all_predictions=disease["All_Predictions"],
            predicted_growth=growth["Predicted_Growth_Milestone"],
            temperature=temperature,
            soil_moisture=soil_moisture,
            sunlight_hours=sunlight_hours,
            humidity=humidity
        )"""
    if target3_alt in text:
        text = text.replace(target3_alt, repl3)
    else:
        print('Failed to find target3 or target3_alt')

with open('src/api/api_server.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("api_server.py updated")
