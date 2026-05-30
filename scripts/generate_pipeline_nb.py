import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

text_intro = """\
# Demeter Data Pipeline Verification & Benchmarking
This notebook verifies the data pipeline from start to finish and provides computational benchmarks comparing the deep CNN models against our lightweight Hybrid SVM models (FFT + HSV).

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
import warnings

# Suppress slow TensorFlow GPU initialization/logging for much faster import times on CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.abspath('..'))

from src.core.inference_engine import extract_hybrid_fft_features, predict_hybrid_disease, predict_hierarchical_cnn, predict_hybrid_hierarchical
from src.utils.data_loader import load_image_for_cnn
"""

text_data = """\
## 1. Load Subset of Data & Models
We load a small subset of PlantVillage images to run our latency tests."""

code_data = """\
# Setup paths
DATA_DIR = os.path.abspath('../data/raw/vision/PlantVillage')
if not os.path.exists(DATA_DIR):
    print("Please ensure raw PlantVillage data is in data/raw/vision/PlantVillage")
    
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
    print("Loading Base CNN...")
    cnn_model = tf.keras.models.load_model(cnn_path)
else:
    print(f"Base CNN model not found at {cnn_path}")

print("Loading Hierarchical CNN models...")
id_model_path = '../models/demeter_cnn_plantvillage_species_identifier.keras'
hier_models = {}
if os.path.exists(id_model_path):
    identifier_model = tf.keras.models.load_model(id_model_path)
    for species in ['Pepper', 'Potato', 'Tomato']:
        model_path = f'../models/demeter_cnn_plantvillage_{species.lower()}.keras'
        if os.path.exists(model_path):
            hier_models[species.lower()] = tf.keras.models.load_model(model_path)
else:
    print("Hierarchical CNN models not found.")

# Load SVM models
svm_dir = '../models/experimentation'
try:
    svm_model = joblib.load(os.path.join(svm_dir, 'hybrid_svm.joblib'))
    scaler_fft = joblib.load(os.path.join(svm_dir, 'hybrid_fft_scaler.joblib'))
    pca_fft = joblib.load(os.path.join(svm_dir, 'hybrid_fft_pca.joblib'))
    scaler_hist = joblib.load(os.path.join(svm_dir, 'hybrid_hist_scaler.joblib'))
    print("Loaded Base SVM pipeline components.")
except Exception as e:
    print(f"Base SVM pipeline components missing: {e}")

print("Loading Hierarchical SVM models...")
try:
    identifier_components = {
        "svm": joblib.load(os.path.join(svm_dir, 'hybrid_svm_species_identifier.joblib')),
        "classes": joblib.load(os.path.join(svm_dir, 'hybrid_svm_species_identifier_classes.joblib')),
        "fft_pca": joblib.load(os.path.join(svm_dir, 'hybrid_svm_species_identifier_fft_pca.joblib')),
        "fft_scaler": joblib.load(os.path.join(svm_dir, 'hybrid_svm_species_identifier_fft_scaler.joblib')),
        "hist_scaler": joblib.load(os.path.join(svm_dir, 'hybrid_svm_species_identifier_hist_scaler.joblib'))
    }
    
    species_svms_cache = {}
    for species in ['pepper', 'potato', 'tomato']:
        spec_dir = os.path.join(svm_dir, 'species_svms')
        species_svms_cache[species] = {
            "svm": joblib.load(os.path.join(spec_dir, f'hybrid_svm_{species}.joblib')),
            "classes": joblib.load(os.path.join(spec_dir, f'hybrid_svm_{species}_classes.joblib')),
            "fft_pca": joblib.load(os.path.join(spec_dir, f'hybrid_svm_{species}_fft_pca.joblib')),
            "fft_scaler": joblib.load(os.path.join(spec_dir, f'hybrid_svm_{species}_fft_scaler.joblib')),
            "hist_scaler": joblib.load(os.path.join(spec_dir, f'hybrid_svm_{species}_hist_scaler.joblib'))
        }
    print("Loaded Hierarchical SVM pipeline components.")
except Exception as e:
    print(f"Hierarchical SVM pipeline components missing: {e}")
"""

text_cnn = """\
## 2. Base CNN Inference Benchmark
We measure the total time taken to preprocess an image for the Base CNN and run inference."""

code_cnn = """\
cnn_times = []
cnn_prep_times = []
cnn_inf_times = []

# Warm-up (TensorFlow initialization can be slow on first run)
if len(sample_images) > 0 and 'cnn_model' in locals():
    dummy_img = np.zeros((1, 150, 150, 3))
    cnn_model.predict(dummy_img, verbose=0)

for img_path in sample_images:
    start_time = time.time()
    
    # Preprocessing
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (150, 150))
    img_array = np.expand_dims(img_resized, axis=0)
    
    prep_time = time.time()
    # Inference
    cnn_model.predict(img_array, verbose=0)
    
    end_time = time.time()
    cnn_times.append((end_time - start_time) * 1000) # milliseconds
    cnn_prep_times.append((prep_time - start_time) * 1000)
    cnn_inf_times.append((end_time - prep_time) * 1000)

cnn_avg_time = np.mean(cnn_times) if cnn_times else 0
cnn_avg_prep = np.mean(cnn_prep_times) if cnn_prep_times else 0
cnn_avg_inf = np.mean(cnn_inf_times) if cnn_inf_times else 0
print(f"Average CNN Total Time: {cnn_avg_time:.2f} ms per image")
print(f"  - Preprocessing Time: {cnn_avg_prep:.2f} ms")
print(f"  - Inference Time: {cnn_avg_inf:.2f} ms")
"""

text_hier_cnn = """\
## 3. Hierarchical CNN Inference Benchmark
We measure the total time taken to run the primary species identifier CNN and the species-specific disease CNN."""

code_hier_cnn = """\
species_names = ['Pepper', 'Potato', 'Tomato']
class_dirs = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

hier_cnn_times = []
hier_cnn_prep_times = []
hier_cnn_inf_times = []

# Warm-up
if len(sample_images) > 0 and 'identifier_model' in locals():
    dummy_img = np.zeros((1, 224, 224, 3))
    predict_hierarchical_cnn(dummy_img, identifier_model, hier_models, species_names, class_dirs)

for img_path in sample_images:
    start_time = time.time()
    
    # Preprocessing
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)
    
    prep_time = time.time()
    # Inference
    try:
        predict_hierarchical_cnn(img_array, identifier_model, hier_models, species_names, class_dirs)
    except Exception as e:
        print(f"Failed on {img_path}: {e}")
        
    end_time = time.time()
    hier_cnn_times.append((end_time - start_time) * 1000)
    hier_cnn_prep_times.append((prep_time - start_time) * 1000)
    hier_cnn_inf_times.append((end_time - prep_time) * 1000)

hier_cnn_avg_time = np.mean(hier_cnn_times) if hier_cnn_times else 0
hier_cnn_avg_prep = np.mean(hier_cnn_prep_times) if hier_cnn_prep_times else 0
hier_cnn_avg_inf = np.mean(hier_cnn_inf_times) if hier_cnn_inf_times else 0
print(f"Average Hierarchical CNN Total Time: {hier_cnn_avg_time:.2f} ms per image")
print(f"  - Preprocessing Time: {hier_cnn_avg_prep:.2f} ms")
print(f"  - Inference Time: {hier_cnn_avg_inf:.2f} ms")
"""

text_svm = """\
## 4. Base SVM (FFT+HSV) Inference Benchmark
We measure the total time taken for Otsu segmentation, 2D FFT, HSV histograms, PCA, and Base SVM inference."""

code_svm = """\
svm_times = []
svm_prep_times = []
svm_inf_times = []

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
        
        prep_time = time.time()
        svm_model.predict(X_hybrid)
        
        end_time = time.time()
        svm_times.append((end_time - start_time) * 1000) # milliseconds
        svm_prep_times.append((prep_time - start_time) * 1000)
        svm_inf_times.append((end_time - prep_time) * 1000)
    except Exception as e:
        print(f"Failed on {img_path}: {e}")

svm_avg_time = np.mean(svm_times) if svm_times else 0
svm_avg_prep = np.mean(svm_prep_times) if svm_prep_times else 0
svm_avg_inf = np.mean(svm_inf_times) if svm_inf_times else 0
print(f"Average Base SVM Total Time: {svm_avg_time:.2f} ms per image")
print(f"  - Preprocessing Time (Otsu + FFT + HSV): {svm_avg_prep:.2f} ms")
print(f"  - Inference Time: {svm_avg_inf:.2f} ms")
"""

text_hier_svm = """\
## 5. Hierarchical SVM Inference Benchmark
We measure the total time taken for feature extraction, primary species identification, and species-specific disease classification."""

code_hier_svm = """\
hier_svm_times = []

# Warm up
if len(sample_images) > 0 and 'identifier_components' in locals():
    predict_hybrid_hierarchical(sample_images[0], identifier_components, species_svms_cache)

for img_path in sample_images:
    start_time = time.time()
    try:
        predict_hybrid_hierarchical(img_path, identifier_components, species_svms_cache)
    except Exception as e:
        print(f"Failed on {img_path}: {e}")
        
    end_time = time.time()
    hier_svm_times.append((end_time - start_time) * 1000) # milliseconds

hier_svm_avg_time = np.mean(hier_svm_times) if hier_svm_times else 0
print(f"Average Hierarchical SVM Total Time: {hier_svm_avg_time:.2f} ms per image")
"""

text_compare = """\
## 6. Visual Comparison
Let's generate the comparison charts to paste into the presentation."""

code_compare = """\
if cnn_times and svm_times and hier_cnn_times and hier_svm_times:
    labels = ['CNN (Base)', 'CNN (Hierarchical)', 'SVM (Base)', 'SVM (Hierarchical)']
    times = [cnn_avg_time, hier_cnn_avg_time, svm_avg_time, hier_svm_avg_time]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, times, color=['#e74c3c', '#f39c12', '#2ecc71', '#3498db'])
    
    ax.set_ylabel('Inference Latency (ms / image)')
    ax.set_title('Computational Latency Comparison: CNN vs SVM')
    
    # Add text labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f} ms', va='bottom', ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('latency_comparison.png', dpi=300)
    plt.show()
    
    speedup_base = cnn_avg_time / svm_avg_time
    speedup_hier = hier_cnn_avg_time / hier_svm_avg_time
    print(f"CONCLUSION: The Base SVM is {speedup_base:.2f}x faster than the Base CNN and the Hierarchical SVM is {speedup_hier:.2f}x faster than the Hierarchical CNN.")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(text_data),
    nbf.v4.new_code_cell(code_data),
    nbf.v4.new_markdown_cell(text_cnn),
    nbf.v4.new_code_cell(code_cnn),
    nbf.v4.new_markdown_cell(text_hier_cnn),
    nbf.v4.new_code_cell(code_hier_cnn),
    nbf.v4.new_markdown_cell(text_svm),
    nbf.v4.new_code_cell(code_svm),
    nbf.v4.new_markdown_cell(text_hier_svm),
    nbf.v4.new_code_cell(code_hier_svm),
    nbf.v4.new_markdown_cell(text_compare),
    nbf.v4.new_code_cell(code_compare)
]

os.makedirs('../notebooks', exist_ok=True)
with open('../notebooks/pipeline_verification.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook generated at notebooks/pipeline_verification.ipynb")
