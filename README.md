Demeter: Intelligent Plant Growth Optimisation System

Demeter is a modular machine learning framework designed to bridge the "accessibility gap" in precision agriculture. By combining visual diagnostics with environmental data, the system provides home gardeners and small-scale farmers with high-fidelity insights into plant health, disease detection, and growth trajectories without the need for expensive industrial hardware.

**USYD 2026 SC1 - ENGG2112**

**Contributors:** Aman, Edward, Aneesh, Jack, The Honourable 5th Lab Partner


Project Overview
Effective plant care is often hampered by a lack of real-time diagnostic tools. Demeter addresses this by utilizing a dual-stream machine learning architecture:

Visual Health Monitoring: Using Convolutional Neural Networks (CNN) to detect leaf discoloration, spotting, and tissue degradation.

Environmental Analytics: Using Random Forest Regression to map sensor data (temperature, soil moisture, water use) to future growth metrics.

Key Features
Species Classification: Automatically identify plant species from leaf imagery.

Stress Diagnosis: Estimate stress levels and identify diseases in real-time.

Growth Trajectory Prediction: Forecast future biomass and health based on longitudinal environmental data.

Actionable Insights: Generate data-driven recommendations, such as optimized watering schedules.

Technical Architecture
The system is built on a modular Python-based pipeline:

1. Image Processing Stream (CNN)
Model: Fine-tuned Convolutional Neural Network.

Function: Extracts high-dimensional features from images to classify 38 different categories of healthy and diseased leaves.

Dataset: PlantVillage Dataset (~54,000 images).

2. Growth Prediction Stream (Random Forest)
Model: Random Forest Regressor.

Function: Maps temporal environmental variables to continuous growth metrics (e.g., biomass area).

Dataset: Donald Danforth Plant Science Center (Multi-modal hyperspectral and longitudinal data).

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
