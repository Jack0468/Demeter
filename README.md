# Demeter 🌾

Demeter AI is an intelligent plant growth optimization system that uses machine learning to classify plant species and analyze real-time environmental sensor data (temperature, humidity, light, soil moisture, CO₂) to generate personalized growth recommendations and automate environmental controls for maximum yield.

**USYD 2026 SC1 - ENGG2112**

**Contributors:** Aman, Edward, Aneesh, Jack, The Honourable 5th Lab Partner

---

## 🏗️ Project Architecture

To ensure separation of concerns and maintainable code, the software pipeline is modularized into the following components:

* `setup_data.py`: Automatically interfaces with the Kaggle API to download and organize the required image and tabular datasets into the local repository.
* `model_builder.py`: Contains the logic to train the Convolutional Neural Network (CNN) for computer vision using Transfer Learning (MobileNetV2), and the Random Forest for environmental data analysis.
* `inference_engine.py`: The core logic module. It loads the trained models, processes new sensor/image inputs, and generates actionable Yes/No recommendations.
* `main.py`: The central orchestrator. It verifies that models exist (triggering training if not), runs the continuous monitoring loop, and logs all outputs to a CSV file.

## ⚙️ Prerequisites

* Python 3.9+
* Anaconda or Miniconda (Recommended for dependency management, especially if running via WSL or Linux)
* A Kaggle account (for API access)

## 🚀 Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/Jack0468/Demeter.git](https://github.com/Jack0468/Demeter.git)
cd Demeter
