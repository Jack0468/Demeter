import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path

# --- DYNAMIC PROJECT ROOT RESOLUTION ---
_current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _current_dir.parent.parent if _current_dir.parent.name == "src" else _current_dir.parent

# Add project root to path for imports from src.*
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.core.inference_engine import (
        load_models, 
        generate_complete_diagnosis, 
        diagnose_plant_disease, 
        predict_growth_milestone
    )
except ModuleNotFoundError:
    from inference_engine import (
        load_models, 
        generate_complete_diagnosis, 
        diagnose_plant_disease, 
        predict_growth_milestone
    )

app = Flask(__name__)
CORS(app)

# Configuration for models and data
plantvillage_cnn_model_path = str(PROJECT_ROOT / "models/demeter_cnn_plantvillage.keras")
danforth_rf_model_path = str(PROJECT_ROOT / "models/demeter_rf_danforth.joblib")
plantvillage_dir = str(PROJECT_ROOT / "data/raw/vision/PlantVillage")

print("Loading AI Models for Web Inference...")
try:
    cnn_model, rf_model = load_models(plantvillage_cnn_model_path, danforth_rf_model_path)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    cnn_model, rf_model = None, None

# Get class directories for CNN
try:
    class_dirs = sorted([d for d in os.listdir(plantvillage_dir) 
                       if os.path.isdir(os.path.join(plantvillage_dir, d))])
except FileNotFoundError:
    class_dirs = []

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Handles inputs from the web interface and provides model outputs.
    Expects a JSON payload with image path and sensor readings.
    """
    if cnn_model is None or rf_model is None:
        return jsonify({"error": "Models are not loaded on the server."}), 500

    # Support both multipart/form-data (browser uploads) and JSON (programmatic usage)
    if 'image' in request.files:
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        uploads_dir = str(PROJECT_ROOT / "data/uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        image_path = os.path.join(uploads_dir, file.filename)
        file.save(image_path)
        
        temperature = float(request.form.get("temperature", 25.0))
        soil_moisture = float(request.form.get("soil_moisture", 50.0))
        sunlight_hours = float(request.form.get("sunlight_hours", 6.0))
        humidity = float(request.form.get("humidity", 50.0))
    elif request.is_json:
        data = request.json
        image_path = data.get("image_path")
        temperature = float(data.get("temperature", 25.0))
        soil_moisture = float(data.get("soil_moisture", 50.0))
        sunlight_hours = float(data.get("sunlight_hours", 6.0))
        humidity = float(data.get("humidity", 50.0))
        if not image_path or not os.path.exists(image_path):
            return jsonify({"error": "Valid image_path is required"}), 400
    else:
        return jsonify({"error": "No image file or JSON payload provided"}), 400

    try:
        # 1. Image Classification
        disease_diagnosis = diagnose_plant_disease(image_path, cnn_model, class_dirs)
        
        # 2. Growth Prediction
        env_data = {
            "Soil_Type": 1,  # Default/Placeholder encoded value
            "Sunlight_Hours": sunlight_hours,
            "Water_Frequency": 2,  # Default/Placeholder encoded value
            "Fertilizer_Type": 1,  # Default/Placeholder encoded value
            "Temperature": temperature,
            "Humidity": humidity
        }
        growth_pred = predict_growth_milestone(env_data, rf_model)
        
        # 3. Generate Complete Diagnosis
        diagnosis = generate_complete_diagnosis(
            image_path=image_path,
            detected_disease=disease_diagnosis['Detected_Disease'],
            disease_confidence=disease_diagnosis['Disease_Confidence'],
            all_predictions=disease_diagnosis['All_Predictions'],
            predicted_growth=growth_pred['Predicted_Growth_Milestone'],
            temperature=temperature,
            soil_moisture=soil_moisture,
            sunlight_hours=sunlight_hours,
            humidity=humidity
        )
        
        return jsonify(diagnosis), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    """Provides a helpful redirect message if someone visits the inference server directly."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Demeter Web Inference Server</title></head>
    <body style="font-family: sans-serif; padding: 20px; background: #f4f6f2; color: #1a1a1a;">
        <h2 style="color: #2d6a4f;">Demeter Web Inference Server (Port 5001)</h2>
        <p>This server handles machine learning predictions natively via <code>POST /api/predict</code>.</p>
        <p><strong>To view the user interface, please open the main dashboard on Port 5000:</strong></p>
        <p style="margin-top: 20px;"><a href="http://localhost:5000/dashboard" style="background: #2d6a4f; color: white; padding: 10px 15px; text-decoration: none; border-radius: 5px; font-weight: bold;">Go to Dashboard</a></p>
    </body>
    </html>
    """, 200

if __name__ == "__main__":
    print("=" * 60)
    print("Demeter Web Inference Server")
    print("=" * 60)
    print("Starting server on http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=True)