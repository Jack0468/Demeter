# Demeter Dashboard Integration Setup

## Overview

The Demeter system now includes three new components for real-time dashboard visualization:

1. **output_formatter.py** - Converts raw model outputs to standardized JSON
2. **status_engine.py** - Rule-based health scoring and recommendations
3. **api_server.py** - Flask API endpoint for dashboard data
4. **dashboard.html** (enhanced) - Live data visualization with auto-refresh

## Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Ensures you have Flask, Flask-CORS, and all ML dependencies.

### Step 2: Run the API Server

In one terminal:

```bash
python api_server.py
```

Expected output:
```
============================================================
Demeter API Server
============================================================
Output directory: /path/to/data/outputs
Latest diagnosis: /path/to/data/outputs/latest_diagnosis.json
History file: /path/to/data/outputs/diagnosis_history.json

Starting server on http://localhost:5000
Dashboard: http://localhost:5000/dashboard
API Docs: http://localhost:5000/
============================================================
```

### Step 3: Generate Diagnoses

In another terminal, run the inference pipeline:

```bash
python main.py
```

This will:
- Load trained models
- Run disease detection on test images
- Run growth predictions on environmental data
- Generate formatted JSON outputs
- Save to `data/outputs/latest_diagnosis.json`
- Append to `data/outputs/diagnosis_history.json`

### Step 4: View the Dashboard

Open your browser and navigate to:
```
http://localhost:5000/dashboard
```

The dashboard will:
- Fetch latest diagnosis every 5 seconds
- Display disease detection with confidence bars
- Show stress diagnosis matrix (moisture, temperature, light, nutrients)
- Display 7-day growth trajectory
- Show actionable recommendations
- List diagnostic log with latest 10 results

---

## Architecture & Data Flow

### JSON Schema

Each diagnosis is stored as:

```json
{
  "timestamp": "2026-05-01T14:30:22.123456",
  "image_path": "data/layer2_health_rgb/PlantVillage/Tomato__Early_blight/image.jpg",
  "cnn_result": {
    "primary_disease": "Early Blight",
    "confidence": 0.842,
    "top_3": [
      {"class": "Early Blight", "confidence": 0.842},
      {"class": "Late Blight", "confidence": 0.089},
      {"class": "Healthy", "confidence": 0.069}
    ]
  },
  "rf_result": {
    "predicted_growth": 45.3,
    "trajectory": "stable"
  },
  "sensors": {
    "temperature": 24.5,
    "soil_moisture": 32.0,
    "sunlight_hours": 4.8,
    "humidity": 65.0
  },
  "stress_diagnosis": {
    "moisture_stress": "High",
    "temperature_stress": "Low",
    "light_deficit": "Moderate",
    "nutrient_status": "Normal",
    "disease_severity": "Critical"
  },
  "health_score": 45,
  "overall_status": "Struggling",
  "recommendations": [
    {
      "priority": 1,
      "action": "Increase watering frequency immediately",
      "icon": "💧",
      "urgency": "critical"
    },
    {
      "priority": 2,
      "action": "Apply fungicide",
      "icon": "🧪",
      "urgency": "high"
    }
  ],
  "system_command": "ACTIVATE_WATER_PUMP | SEND_WARNING_ALERT",
  "trajectory_7day": {
    "1": "Struggling",
    "3": "Fair",
    "5": "Poor",
    "7": "Poor"
  }
}
```

### API Endpoints

All endpoints available at `http://localhost:5000`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Service health check |
| `/api/latest` | GET | Latest diagnosis |
| `/api/history` | GET | Diagnosis history (paginated) |
| `/api/summary` | GET | Summary statistics |
| `/api/latest/export` | GET | Latest diagnosis as CSV format |
| `/api/status` | GET | System status (models, data, uptime) |
| `/api/status/thresholds` | GET | Current status engine thresholds |
| `/api/config` | GET | System configuration |
| `/dashboard` | GET | Dashboard HTML |
| `/` | GET | API documentation |

### Query Parameters

**`/api/history`**
```
GET /api/history?limit=50&offset=0
```
- `limit` (int): Max records to return (default: 50)
- `offset` (int): Skip N records (default: 0)

---

## Configuration & Thresholds

Edit `status_engine.py` to customize stress level thresholds:

```python
from status_engine import StressThresholds

thresholds = StressThresholds(
    moisture_critical=25.0,      # High stress below 25%
    moisture_warning=40.0,       # Medium stress below 40%
    temp_too_cold=15.0,          # Temperature too cold
    temp_too_hot=30.0,           # Temperature too hot
    disease_critical=0.75,       # Disease confidence > 75% = critical
    # ... more settings
)
```

Or fetch current thresholds from API:

```bash
curl http://localhost:5000/api/status/thresholds
```

---

## Integration with main.py

The main.py pipeline has been enhanced to generate JSON outputs:

```python
from inference_engine import generate_complete_diagnosis

# After running CNN and RF predictions:
diagnosis = generate_complete_diagnosis(
    image_path="/path/to/image.jpg",
    detected_disease="Early Blight",
    disease_confidence=0.84,
    all_predictions={"Early Blight": 0.84, "Late Blight": 0.10, ...},
    predicted_growth=45.3,
    temperature=24.5,
    soil_moisture=32.0,
    sunlight_hours=4.8,
    humidity=65.0
)

# diagnosis dict now includes:
# - stress_diagnosis (moisture, temperature, light, nutrient)
# - health_score (0-100)
# - overall_status (Thriving/Struggling/Critical)
# - recommendations (prioritized action list)
# - system_command (automation triggers)
# - trajectory_7day (7-day forecast)
```

---

## File Structure

```
Demeter/
├── api_server.py              # Flask API endpoint (NEW)
├── output_formatter.py        # JSON serialization (NEW)
├── status_engine.py           # Health scoring & recommendations (NEW)
├── dashboard.html             # Enhanced with JavaScript (UPDATED)
├── main.py                    # Inference pipeline (UPDATED)
├── inference_engine.py        # Enhanced with new functions (UPDATED)
├── requirements.txt           # Dependencies (NEW)
├── data/
│   ├── outputs/               # Created automatically
│   │   ├── latest_diagnosis.json
│   │   └── diagnosis_history.json
│   └── plant_diagnostics.csv  # Enhanced CSV logging
├── models/
│   ├── demeter_cnn_plantvillage.keras
│   └── demeter_rf_danforth.joblib
└── ... (other files)
```

---

## Workflow

### For Development / Testing

1. **Start API Server**
   ```bash
   python api_server.py
   ```

2. **In another terminal, run inference**
   ```bash
   python main.py
   ```

3. **Open dashboard**
   ```
   http://localhost:5000/dashboard
   ```

4. **Monitor API logs** to see requests from dashboard

### For Production

- Run `api_server.py` as a service (systemd, supervisor, etc.)
- Configure Flask with a production WSGI server (Gunicorn, uWSGI)
- Set `debug=False` in api_server.py (already done)
- Add authentication if needed
- Use reverse proxy (nginx) if running on different ports

---

## Troubleshooting

### Dashboard shows "● Offline"
- Check if api_server.py is running
- Verify Flask is listening on http://localhost:5000
- Check browser console (F12) for CORS errors
- Ensure Firefox/Chrome allows CORS requests

### No diagnosis data appearing
- Run `main.py` first to generate diagnoses
- Check `data/outputs/latest_diagnosis.json` exists
- Verify API returns data: `curl http://localhost:5000/api/latest`

### Port 5000 already in use
- Change port in api_server.py: `app.run(port=5001)`
- Or kill existing process: `lsof -i :5000 | kill -9 <PID>`

### Import errors (output_formatter, status_engine)
- Ensure you're running from the Demeter root directory
- Check that files are in the same directory as main.py
- Add to Python path: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`

---

## Next Steps

1. **Enhance main.py** to call `generate_complete_diagnosis()` for all test cases
2. **Add database persistence** (SQLite/PostgreSQL) for long-term history
3. **Add image serving** to display analyzed plant images in dashboard
4. **Add live sensor integration** for real-time IoT data
5. **Add export functionality** (CSV, PDF reports)
6. **Add user authentication** for multi-user scenarios

---

## Team Integration

- **Jack (Data Pipeline)**: Integrate JSON outputs into main.py
- **Aman (RF)**: Customize threshold logic in status_engine.py
- **Edward (CNN)**: Ensure output format matches disease detection needs
- **Aneesh (Evaluation)**: Add metrics export to `/api/summary`

All components are modular and can be customized independently.
