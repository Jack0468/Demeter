Demeter: Intelligent Plant Growth Optimisation System

Demeter is a modular machine learning framework designed to bridge the "accessibility gap" in precision agriculture. By combining visual diagnostics with environmental data, the system provides home gardeners and small-scale farmers with high-fidelity insights into plant health, disease detection, and growth trajectories without the need for expensive industrial hardware.

**USYD 2026 SC1 - ENGG2112**

**Contributors:** Aman, Edward, Aneesh, Jack, The Honourable 5th Lab Partner


Project Overview
Effective plant care is often hampered by a lack of real-time diagnostic tools. Demeter addresses this by utilizing a dual-stream machine learning architecture:

Visual Health Monitoring: Using Convolutional Neural Networks (CNN) to detect leaf discoloration, spotting, and tissue degradation.

Environmental Analytics: Using Random Forest Regression to map sensor data (temperature, soil moisture, water use) to future growth metrics.

Key Features
* **Disease Classification:** Automatically identify plant diseases from leaf imagery using CNNs.
* **Growth Trajectory Prediction:** Forecast future biomass and health milestones based on longitudinal environmental data using Random Forests.
* **Rule-Based Status Engine:** Translate raw ML predictions into composite health scores, 7-day trajectories, and system automation commands (e.g., `ACTIVATE_WATER_PUMP`).
* **Actionable Insights:** Generate data-driven, prioritized recommendations for intervention.
* **Real-time Dashboard:** A responsive web interface powered by a Flask API to visualize crop health globally.

Technical Architecture
The system is built on a modular Python-based pipeline:

### 1. Image Processing Stream (CNN)
**Model:** Fine-tuned Convolutional Neural Network (MobileNetV2 backbone).
**Function:** Extracts high-dimensional features from images to classify categories of healthy and diseased leaves.
**Dataset:** PlantVillage Dataset (~54,000 images).

### 2. Growth Prediction Stream (Random Forest)
**Model:** Random Forest Regressor.
**Function:** Maps temporal environmental variables to continuous growth metrics.
**Dataset:** Donald Danforth Plant Science Center (Multi-modal hyperspectral and longitudinal data).

### Directory Structure
```text
Demeter/
├── src/
│   ├── api/          # Flask API servers (api_server.py, web_inference.py)
│   ├── core/         # Core business logic (inference_engine, status_engine)
│   ├── training/     # Model builders and training pipelines
│   └── evaluation/   # Formal ML metrics and comparison suites
├── frontend/         # Web dashboard UI (dashboard.html)
├── notebooks/        # Interactive Jupyter testing environments
├── scripts/          # Utility scripts (data expansion, Kaggle push)
├── data/             # Datasets, model outputs, and logs (Git ignored)
├── models/           # Saved .keras and .joblib files (Git ignored)
└── config.json       # Central configuration and path routing
```

Evaluation Strategy
We define success through the following metrics:

Classification: Accuracy, Precision, Recall, and F1-Score (Target: >85% accuracy).

Regression: Root Mean Square Error (RMSE) for growth trajectory predictions.
## ⚙️ Prerequisites

* Python 3.9+
* Anaconda or Miniconda (Recommended for dependency management, especially if running via WSL or Linux)
* A Kaggle account (for API access)

## 🚀 Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/Jack0468/Demeter.git](https://github.com/Jack0468/Demeter.git)
cd Demeter
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

## 💻 Usage Guide

### 1. Train the Models
To initialize the models (if they don't exist yet) or force a retrain based on `config.json`:
```bash
python src/training/train_pipeline.py
```

### 2. Start the API Server
The API server provides endpoints to fetch historical diagnoses, overall system status, and thresholds.
```bash
python src/api/api_server.py
```
*The server will start on `http://localhost:5000`.*

### 3. Start the Web Inference Server (Optional)
If you want to actively upload images and generate *new* predictions from the dashboard, start the inference worker:
```bash
python src/api/web_inference.py
```
*The inference worker listens on `http://localhost:5001`.*

### 4. Open the Dashboard
Navigate to your API server's dashboard route in any modern web browser:
```text
http://localhost:5000/dashboard
```

### 5. Run the Evaluation Suite
To generate deep analytical metrics, confusion matrices, and formal classification reports on unseen test data:
```bash
python src/evaluation/run_evaluation_suite.py \
  --cnn_test_dir data/layer2_health_rgb/PlantVillage \
  --rf_csv data/layer3_environment/plant_growth_data.csv \
  --run_name my_eval_run
