"""
API Server: Lightweight Flask endpoint serving latest and historical diagnoses
to the dashboard frontend.
"""

import os
import json
import sys
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
OUTPUT_DIR = os.path.join(os.getcwd(), "data", "outputs")
LATEST_DIAGNOSIS_FILE = os.path.join(OUTPUT_DIR, "latest_diagnosis.json")
HISTORY_FILE = os.path.join(OUTPUT_DIR, "diagnosis_history.json")
CSV_LOG_FILE = os.path.join(os.getcwd(), "data", "plant_diagnostics.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


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
    config_path = os.path.join(os.getcwd(), "config.json")
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
        "cnn_plantvillage": os.path.exists("models/demeter_cnn_plantvillage.keras"),
        "rf_danforth": os.path.exists("models/demeter_rf_danforth.joblib"),
        "cnn_bellwether": os.path.exists("models/demeter_cnn.keras"),
        "rf_bellwether": os.path.exists("models/demeter_rf.joblib")
    }
    
    # Check data availability
    data_available = {
        "plantvillage": os.path.exists("data/layer2_health_rgb/PlantVillage"),
        "danforth": os.path.exists("data/layer3_environment/plant_growth_data.csv"),
        "bellwether": os.path.exists("data/bellwether_images_dir")
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
    """
    Serve the dashboard HTML file.
    
    Returns:
        HTML: dashboard.html
    """
    dashboard_path = os.path.join(os.getcwd(), "dashboard.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, 'r') as f:
            return f.read(), 200
    else:
        return jsonify({"error": "dashboard.html not found"}), 404


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
    print("Demeter API Server")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Latest diagnosis: {LATEST_DIAGNOSIS_FILE}")
    print(f"History file: {HISTORY_FILE}")
    print("\nStarting server on http://localhost:5000")
    print("Dashboard: http://localhost:5000/dashboard")
    print("API Docs: http://localhost:5000/")
    print("=" * 60)
    
    # Create required directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Start Flask app
    app.run(debug=False, host="0.0.0.0", port=5000)
