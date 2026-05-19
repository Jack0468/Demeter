"""
API Server: Unified Flask server serving the dashboard, inference API,
static assets, and historical diagnosis data.
Merges web_inference.py (previously port 5001) into a single server.
"""

import os
import json
import sys
import csv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from datetime import datetime
from pathlib import Path

# --- DYNAMIC PROJECT ROOT RESOLUTION ---
_current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _current_dir.parent.parent if _current_dir.parent.name == "src" else _current_dir.parent

# Add project root to path for imports from src.*
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

frontend_dir = str(PROJECT_ROOT / "src" / "frontend")
app = Flask(__name__, static_folder=frontend_dir, static_url_path='/static')
CORS(app)  # Enable CORS for all routes

# Configuration
OUTPUT_DIR = str(PROJECT_ROOT / "data" / "outputs")
LATEST_DIAGNOSIS_FILE = str(PROJECT_ROOT / "data" / "outputs" / "latest_diagnosis.json")
HISTORY_FILE = str(PROJECT_ROOT / "data" / "outputs" / "diagnosis_history.json")
CSV_LOG_FILE = str(PROJECT_ROOT / "data" / "plant_diagnostics.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# LAZY MODEL LOADING STATE
# ==========================================
_model_state = {
    "loaded": False,
    "cnn": None,
    "rf": None,
    "cnn_biomass": None,
    "cnn_tiller": None,
    "cnn_water": None,
    "rf_water": None,
    "class_dirs": [],
    "error": None
}

def _ensure_models_loaded():
    """Lazy-load ML models on first /api/predict call."""
    if _model_state["loaded"]:
        return
    try:
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
        _model_state["error"] = None
    except Exception as e:
        _model_state["error"] = str(e)
        print(f"[Demeter] Model loading error: {e}")


def _load_csv_as_dict(filepath) -> dict:
    """Load a single-row CSV into a dict."""
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                return {k: _try_float(v) for k, v in row.items()}
    except Exception:
        return {}
    return {}


def _load_csv_as_list(filepath) -> list:
    """Load a multi-row CSV into a list of dicts."""
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", newline="") as f:
            reader = csv.DictReader(f)
            return [{k: _try_float(v) for k, v in row.items()} for row in reader]
    except Exception:
        return []


def _try_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return val


def load_json_file(filepath: str) -> dict:
    """Safely load JSON file, return empty dict if not found."""
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def load_json_array_file(filepath: str) -> list:
    """Safely load JSON array file, return empty list if not found."""
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON: { status: "online", timestamp: ISO }
    """
    return jsonify({
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "service": "Demeter API"
    }), 200


@app.route("/api/latest", methods=["GET"])
def get_latest_diagnosis():
    """
    Get the latest plant diagnosis.
    
    Returns:
        JSON: Latest diagnosis object or { error: message } if not available
    """
    latest = load_json_file(LATEST_DIAGNOSIS_FILE)
    
    if not latest:
        return jsonify({
            "error": "No diagnosis available yet",
            "message": "Run main.py to generate diagnoses"
        }), 404
    
    return jsonify(latest), 200


@app.route("/api/history", methods=["GET"])
def get_history():
    """
    Get diagnosis history with optional filtering.
    
    Query Parameters:
        limit (int): Maximum records to return (default: 50)
        offset (int): Number of records to skip (default: 0)
    
    Returns:
        JSON: Array of diagnosis objects
    """
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)
    
    history = load_json_array_file(HISTORY_FILE)
    
    # Apply pagination
    paginated = history[offset:offset + limit]
    
    return jsonify({
        "total": len(history),
        "returned": len(paginated),
        "offset": offset,
        "limit": limit,
        "records": paginated
    }), 200


@app.route("/api/summary", methods=["GET"])
def get_summary():
    """
    Get summary statistics from history.
    
    Returns:
        JSON: { total_diagnoses, avg_health_score, status_distribution, etc. }
    """
    history = load_json_array_file(HISTORY_FILE)
    
    if not history:
        return jsonify({
            "total_diagnoses": 0,
            "message": "No diagnosis history available"
        }), 200
    
    # Extract health scores and statuses
    health_scores = []
    status_counts = {"Thriving": 0, "Struggling": 0, "Critical": 0}
    disease_counts = {}
    
    for diagnosis in history:
        if "health_score" in diagnosis:
            health_scores.append(diagnosis["health_score"])
        
        if "overall_status" in diagnosis:
            status = diagnosis["overall_status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        if "cnn_result" in diagnosis and "primary_disease" in diagnosis["cnn_result"]:
            disease = diagnosis["cnn_result"]["primary_disease"]
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
    
    # Calculate statistics
    avg_health = sum(health_scores) / len(health_scores) if health_scores else 0
    
    return jsonify({
        "total_diagnoses": len(history),
        "average_health_score": round(avg_health, 1),
        "status_distribution": status_counts,
        "disease_frequency": disease_counts,
        "last_updated": history[-1].get("timestamp", "unknown") if history else None
    }), 200


@app.route("/api/latest/export", methods=["GET"])
def export_latest():
    """
    Export latest diagnosis as CSV-compatible format.
    
    Returns:
        JSON: Flattened diagnosis object for CSV export
    """
    latest = load_json_file(LATEST_DIAGNOSIS_FILE)
    
    if not latest:
        return jsonify({"error": "No diagnosis available"}), 404
    
    # Flatten nested JSON for CSV compatibility
    flattened = {
        "Timestamp": latest.get("timestamp", ""),
        "Image_Path": latest.get("image_path", ""),
        "Detected_Disease": latest.get("cnn_result", {}).get("primary_disease", ""),
        "Disease_Confidence": latest.get("cnn_result", {}).get("confidence", ""),
        "Predicted_Growth": latest.get("rf_result", {}).get("predicted_growth", ""),
        "Health_Score": latest.get("health_score", ""),
        "Overall_Status": latest.get("overall_status", ""),
        "Temperature": latest.get("sensors", {}).get("temperature", ""),
        "Soil_Moisture": latest.get("sensors", {}).get("soil_moisture", ""),
        "Sunlight_Hours": latest.get("sensors", {}).get("sunlight_hours", ""),
        "Moisture_Stress": latest.get("stress_diagnosis", {}).get("moisture_stress", ""),
        "System_Command": latest.get("system_command", "")
    }
    
    return jsonify(flattened), 200


@app.route("/api/status/thresholds", methods=["GET"])
def get_thresholds():
    """
    Get current status engine thresholds.
    
    Returns:
        JSON: Current threshold configuration
    """
    try:
        from src.core.status_engine import StressThresholds
    except ModuleNotFoundError:
        from status_engine import StressThresholds
    
    thresholds = StressThresholds()
    
    return jsonify({
        "moisture": {
            "critical": thresholds.moisture_critical,
            "warning": thresholds.moisture_warning,
            "healthy": thresholds.moisture_healthy
        },
        "temperature": {
            "too_cold": thresholds.temp_too_cold,
            "too_hot": thresholds.temp_too_hot,
            "optimal_min": thresholds.temp_optimal_min,
            "optimal_max": thresholds.temp_optimal_max
        },
        "sunlight": {
            "minimal": thresholds.sunlight_minimal,
            "moderate": thresholds.sunlight_moderate,
            "optimal": thresholds.sunlight_optimal
        },
        "disease": {
            "critical": thresholds.disease_critical,
            "warning": thresholds.disease_warning
        }
    }), 200


@app.route("/api/config", methods=["GET"])
def get_config():
    """
    Get current system configuration.
    
    Returns:
        JSON: Configuration from config.json
    """
    config_path = str(PROJECT_ROOT / "config.json")
    if not os.path.exists(config_path):
        config_path = str(PROJECT_ROOT / "config" / "config.json")
    config = load_json_file(config_path)
    
    if not config:
        return jsonify({"error": "config.json not found"}), 404
    
    return jsonify(config), 200


@app.route("/api/status", methods=["GET"])
def get_system_status():
    """
    Get comprehensive system status.
    
    Returns:
        JSON: { models_available, latest_diagnosis, api_uptime, etc. }
    """
    # Check model availability
    models = {
        "cnn_plantvillage": (PROJECT_ROOT / "models/demeter_cnn_plantvillage.keras").exists(),
        "rf_danforth": (PROJECT_ROOT / "models/demeter_rf_danforth.joblib").exists(),
        "cnn_bellwether": (PROJECT_ROOT / "models/demeter_cnn.keras").exists(),
        "rf_bellwether": (PROJECT_ROOT / "models/demeter_rf.joblib").exists()
    }
    
    # Check data availability
    data_available = {
        "plantvillage": (PROJECT_ROOT / "data/layer2_health_rgb/PlantVillage").exists(),
        "danforth": (PROJECT_ROOT / "data/layer3_environment/plant_growth_data.csv").exists(),
        "bellwether": (PROJECT_ROOT / "data/bellwether_images_dir").exists()
    }
    
    # Get latest diagnosis timestamp
    latest = load_json_file(LATEST_DIAGNOSIS_FILE)
    latest_timestamp = latest.get("timestamp", "never")
    
    return jsonify({
        "service": "Demeter API",
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "models_available": models,
        "data_available": data_available,
        "latest_diagnosis": latest_timestamp,
        "diagnosis_count": len(load_json_array_file(HISTORY_FILE))
    }), 200


@app.route("/dashboard", methods=["GET"])
def serve_dashboard():
    """Serve the dashboard HTML file."""
    dashboard_path = str(PROJECT_ROOT / "src" / "frontend" / "dashboard.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            return f.read(), 200, {'Content-Type': 'text/html; charset=utf-8'}
    else:
        return jsonify({"error": "dashboard.html not found"}), 404



@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Unified inference endpoint (previously on port 5001).
    Accepts multipart/form-data with 'image' file + optional sensor fields,
    or JSON with 'image_path' + sensor fields.

    Returns:
        JSON: Complete diagnosis payload
    """
    _ensure_models_loaded()

    if _model_state["cnn"] is None or _model_state["rf"] is None:
        msg = _model_state.get("error") or "Models not available — train or place models in /models/"
        return jsonify({"error": msg}), 503

    # --- Parse request ---
    if 'image' in request.files:
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        uploads_dir = str(PROJECT_ROOT / "data/uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        image_path = os.path.join(uploads_dir, file.filename)
        file.save(image_path)
        temperature   = float(request.form.get("temperature",   25.0))
        soil_moisture = float(request.form.get("soil_moisture", 50.0))
        sunlight_hours = float(request.form.get("sunlight_hours", 6.0))
        humidity      = float(request.form.get("humidity",      50.0))
    elif request.is_json:
        data = request.json
        image_path = data.get("image_path")
        if not image_path or not os.path.exists(image_path):
            return jsonify({"error": "Valid image_path required"}), 400
        temperature   = float(data.get("temperature",   25.0))
        soil_moisture = float(data.get("soil_moisture", 50.0))
        sunlight_hours = float(data.get("sunlight_hours", 6.0))
        humidity      = float(data.get("humidity",      50.0))
    else:
        return jsonify({"error": "Provide multipart image file or JSON payload"}), 400

    try:
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
        return jsonify(diagnosis), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """
    Return parsed evaluation metrics from evaluation_outputs/.

    Returns:
        JSON: { plantvillage_cnn, danforth_rf, eval_run_1 }
    """
    ev = PROJECT_ROOT / "evaluation_outputs"
    result = {}

    # PlantVillage CNN
    pv = ev / "plantvillage"
    if pv.exists():
        result["plantvillage_cnn"] = {
            "name": "PlantVillage CNN",
            "type": "Image Classification",
            "model_file": "demeter_cnn_plantvillage.keras",
            "overall": _load_csv_as_dict(pv / "cnn_overall_metrics.csv"),
            "per_class": _load_csv_as_list(pv / "cnn_per_class_metrics.csv"),
            "per_class_accuracy": _load_csv_as_list(pv / "cnn_per_class_accuracy.csv")
        }

    # Danforth RF
    df = ev / "danforth"
    if df.exists():
        result["danforth_rf"] = {
            "name": "Danforth Random Forest",
            "type": "Growth Regression",
            "model_file": "demeter_rf_danforth.joblib",
            "metrics": _load_csv_as_dict(df / "rf_regression_metrics.csv")
        }

    # Eval run 1 summary
    e1 = ev / "eval_run_1"
    if e1.exists():
        result["eval_run_1"] = {
            "name": "Eval Run 1 Summary",
            "records": _load_csv_as_list(e1 / "summary.csv")
        }

    return jsonify(result), 200


@app.route("/", methods=["GET"])
def index():
    """
    Root endpoint redirects to dashboard or API docs.
    
    Returns:
        HTML: API documentation or redirect to dashboard
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Demeter API</title>
        <style>
            body { font-family: sans-serif; background: #f4f6f2; padding: 20px; }
            h1 { color: #2d6a4f; }
            .endpoint { background: white; padding: 12px; margin: 10px 0; border-radius: 6px; border-left: 4px solid #2d6a4f; }
        </style>
    </head>
    <body>
        <h1>Demeter Plant Health API</h1>
        <p>Available endpoints:</p>
        
        <div class="endpoint"><strong>GET /api/health</strong> - Service health check</div>
        <div class="endpoint"><strong>GET /api/latest</strong> - Latest diagnosis</div>
        <div class="endpoint"><strong>GET /api/history</strong> - Diagnosis history (with pagination)</div>
        <div class="endpoint"><strong>GET /api/summary</strong> - Summary statistics</div>
        <div class="endpoint"><strong>GET /api/latest/export</strong> - Latest diagnosis as CSV format</div>
        <div class="endpoint"><strong>GET /api/status</strong> - System status</div>
        <div class="endpoint"><strong>GET /api/status/thresholds</strong> - Status engine thresholds</div>
        <div class="endpoint"><strong>GET /api/config</strong> - System configuration</div>
        <div class="endpoint"><strong>GET /dashboard</strong> - Dashboard HTML</div>
        
        <hr>
        <p><a href="/dashboard">→ Open Dashboard</a></p>
    </body>
    </html>
    """, 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "error": "Endpoint not found",
        "path": request.path,
        "message": "See GET / for available endpoints"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        "error": "Internal server error",
        "message": str(error)
    }), 500


if __name__ == "__main__":
    print("=" * 60)
    print("Demeter — Unified API Server")
    print("=" * 60)
    print(f"Output directory : {OUTPUT_DIR}")
    print(f"Latest diagnosis : {LATEST_DIAGNOSIS_FILE}")
    print(f"History file     : {HISTORY_FILE}")
    print("\nEndpoints:")
    print("  GET  /dashboard        → Dashboard UI")
    print("  POST /api/predict      → Inference (merged from port 5001)")
    print("  GET  /api/latest       → Latest diagnosis")
    print("  GET  /api/history      → Diagnosis history")
    print("  GET  /api/metrics      → Evaluation metrics")
    print("  GET  /api/status       → System status")
    print("  GET  /api/health       → Health check")
    print("\nStarting on http://localhost:5000")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    app.run(debug=False, host="0.0.0.0", port=5000)
