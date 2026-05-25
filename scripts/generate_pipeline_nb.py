import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

text_intro = """\
# Demeter Data Pipeline Verification & Benchmarking
This notebook verifies the data pipeline from start to finish and provides computational benchmarks comparing the deep CNN model (MobileNetV2) against our lightweight Hybrid SVM model (FFT + HSV).

The goal is to provide hard evidence that the Hybrid SVM significantly improves computation times, making it highly suitable for edge-device deployment."""

code_imports = """\
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import joblib

# Add src to path
sys.path.append(os.path.abspath('..'))

from src.core.inference_engine import extract_hybrid_fft_features, predict_hybrid_disease
from src.utils.data_loader import load_image_for_cnn
"""

text_data = """\
## 1. Load Subset of Data & Models
We load a small subset of PlantVillage images to run our latency tests."""

code_data = """\
# Setup paths
DATA_DIR = os.path.abspath('../data/raw/plantvillage')
if not os.path.exists(DATA_DIR):
    print("Please ensure raw PlantVillage data is in data/raw/plantvillage")
    
# Find a few sample images
sample_images = []
for root, _, files in os.walk(DATA_DIR):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            sample_images.append(os.path.join(root, f))
            if len(sample_images) >= 50: # Limit to 50 for quick benchmarking
                break
    if len(sample_images) >= 50:
        break
        
print(f"Loaded {len(sample_images)} images for benchmarking.")

# Load CNN
cnn_path = '../models/demeter_cnn_plantvillage.keras'
if os.path.exists(cnn_path):
    print("Loading CNN...")
    cnn_model = tf.keras.models.load_model(cnn_path)
else:
    print(f"CNN model not found at {cnn_path}")

# Load SVM models
svm_dir = '../models/experimentation'
try:
    svm_model = joblib.load(os.path.join(svm_dir, 'hybrid_fft_hsv_svm.joblib'))
    scaler_fft = joblib.load(os.path.join(svm_dir, 'hybrid_scaler_fft.joblib'))
    pca_fft = joblib.load(os.path.join(svm_dir, 'hybrid_pca_fft.joblib'))
    scaler_hist = joblib.load(os.path.join(svm_dir, 'hybrid_scaler_hist.joblib'))
    with open(os.path.join(svm_dir, 'hybrid_classes.json'), 'r') as f:
        import json
        class_names = json.load(f)
    print("Loaded SVM pipeline components.")
except Exception as e:
    print(f"SVM pipeline components missing: {e}")
"""

text_cnn = """\
## 2. CNN Inference Benchmark
We measure the total time taken to preprocess an image for the CNN and run inference."""

code_cnn = """\
cnn_times = []

# Warm-up (TensorFlow initialization can be slow on first run)
if len(sample_images) > 0 and 'cnn_model' in locals():
    dummy_img = np.zeros((1, 224, 224, 3))
    cnn_model.predict(dummy_img, verbose=0)

for img_path in sample_images:
    start_time = time.time()
    
    # Preprocessing
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)
    
    # Inference
    cnn_model.predict(img_array, verbose=0)
    
    end_time = time.time()
    cnn_times.append((end_time - start_time) * 1000) # milliseconds

cnn_avg_time = np.mean(cnn_times) if cnn_times else 0
print(f"Average CNN Inference Time: {cnn_avg_time:.2f} ms per image")
"""

text_svm = """\
## 3. SVM (FFT+HSV) Inference Benchmark
We measure the total time taken for Otsu segmentation, 2D FFT, HSV histograms, PCA, and SVM inference."""

code_svm = """\
svm_times = []

for img_path in sample_images:
    start_time = time.time()
    
    # Preprocessing & Inference (all contained in the helper)
    # Re-implementing manually to ensure we trace the pipeline correctly
    try:
        mag_gray, color_hist = extract_hybrid_fft_features(img_path)
        
        X_fft_scaled = scaler_fft.transform([mag_gray])
        X_fft_pca = pca_fft.transform(X_fft_scaled)
        X_hist_scaled = scaler_hist.transform([color_hist])
        X_hybrid = np.hstack([X_fft_pca, X_hist_scaled])
        
        svm_model.predict(X_hybrid)
        
        end_time = time.time()
        svm_times.append((end_time - start_time) * 1000) # milliseconds
    except Exception as e:
        print(f"Failed on {img_path}: {e}")

svm_avg_time = np.mean(svm_times) if svm_times else 0
print(f"Average SVM Pipeline Time: {svm_avg_time:.2f} ms per image")
"""

text_compare = """\
## 4. Visual Comparison
Let's generate the comparison charts to paste into the presentation."""

code_compare = """\
if cnn_times and svm_times:
    labels = ['CNN (MobileNetV2)', 'SVM (FFT + HSV)']
    times = [cnn_avg_time, svm_avg_time]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, times, color=['#e74c3c', '#2ecc71'])
    
    ax.set_ylabel('Inference Latency (ms / image)')
    ax.set_title('Computational Latency Comparison: CNN vs SVM')
    
    # Add text labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f} ms', va='bottom', ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('latency_comparison.png', dpi=300)
    plt.show()
    
    speedup = cnn_avg_time / svm_avg_time
    print(f"CONCLUSION: The SVM pipeline is {speedup:.2f}x faster than the CNN.")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(text_data),
    nbf.v4.new_code_cell(code_data),
    nbf.v4.new_markdown_cell(text_cnn),
    nbf.v4.new_code_cell(code_cnn),
    nbf.v4.new_markdown_cell(text_svm),
    nbf.v4.new_code_cell(code_svm),
    nbf.v4.new_markdown_cell(text_compare),
    nbf.v4.new_code_cell(code_compare)
]

os.makedirs('../notebooks', exist_ok=True)
with open('../notebooks/pipeline_verification.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated at notebooks/pipeline_verification.ipynb")
