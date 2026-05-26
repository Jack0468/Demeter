# Demeter: Consolidated Model Evaluations

This document aggregates the evaluation results of the various models developed, experimented with, and deployed within the Demeter project.

## 1. Deep Learning Vision Models (CNNs)
A suite of convolutional neural networks (CNNs) was trained and evaluated on various datasets to handle different diagnostic and regression tasks from leaf and plant imagery.

### 1.1 Plant Pathogen Classification (PlantVillage CNN)
*   **Production Model Accuracy**: **86.20%**
*   **Baseline Run (`eval_run_1`) Accuracy**: **84.31%**
*   **Use Case**: Core vision stream for identifying plant diseases and pathogens.

### 1.2 Plant Growth Classification (Bellwether CNN)
*   **Accuracy**: **100.0%** (Tested on 6,392 images)
*   **Use Case**: Classification of growth states based on Bellwether snapshot and tile imagery.

### 1.3 Plant Biomass Prediction (Biomass CNN Regressor)
*   **RMSE**: **2.112**
*   **MAE**: **1.348**
*   **R²**: **0.645**
*   **Use Case**: Predictive modeling of plant fresh weight directly from images.

### 1.4 Tiller Count Prediction (Tiller CNN Regressor)
*   **RMSE**: **2.310**
*   **MAE**: **1.833**
*   **R²**: **-0.304**
*   **Use Case**: Predicting the number of tillers from plant imagery. The negative R² indicates the model performs worse than a simple mean predictor on this test set, likely due to extreme data scarcity (only 58 samples available).

---

## 2. Experimental Shallow Models (Biological Signal Processing)
To explore low-latency, computationally efficient alternatives to deep learning, we evaluated shallow models using biologically informed feature engineering.

### 2.1 Pure FFT Support Vector Machine (SVM)
*   **Approach**: Extracted 2D Fast Fourier Transform (FFT) features from 64x64 grayscale images, reduced the 4096 features down to 100 components using PCA, and classified with an SVM.
*   **Accuracy**: **36.39%**
*   **Conclusion**: Pure frequency-domain classification is substantially better than random guessing (~6.6%), but it is insufficient on its own for complex pathogen diagnosis because it lacks spatial color awareness. 

### 2.2 Hybrid SVM (FFT + HSV Color Distributions)
*   **Approach**: Addressed the shortcomings of the pure FFT model by combining 100-component PCA on FFT features with 64-bin HSV color histograms extracted from Otsu-segmented leaf regions.
*   **Accuracy (Full Dataset)**: **84.53%**
*   **F1-Score (Macro)**: **84.28%**
*   **Training Scale**: 20,638 PlantVillage images, fitting an SVM in just ~50.86 seconds on native hardware.

**Pipeline Preprocessing Benchmarks:**
We benchmarked several preprocessing steps to quantify how signal processing affects shallow model performance:

| Preprocessing Stage | Test Accuracy | Macro F1-Score | Key Biological Benefit |
| :--- | :---: | :---: | :--- |
| **Raw Grayscale FFT** | 5.20% | 0.96% | Fails due to high-frequency edge artifacts. |
| **Tapered FFT (Gaussian-fade)** | 30.00% | 28.41% | Eliminates artificial boundary noise using edge fading. |
| **Inpainted FFT (Seamless Pad)** | 29.60% | 27.01% | Replaces background with texture inpaint to minimize edge spikes. |
| **Multichannel LAB FFT** | 48.00% | 47.05% | Captures spatial color transitions across channels. |
| **Production Hybrid FFT + HSV** | **84.53%** | **84.28%** | **Combines texture frequency and color distributions.** |

*   **Conclusion**: Combining texture frequency with color distributions creates a robust shallow classifier that rivals the CNN in accuracy while maintaining a fraction of the computational footprint, making it ideal for edge deployment.

---

## 3. Growth Trajectory Model (Random Forest)
A Random Forest regressor was trained to predict growth milestones and environmental parameters using the Danforth dataset.

*   **Production Model RMSE**: **0.291** (R²: 0.662)
*   **Baseline Run (`eval_run_1`) RMSE**: **0.047** (R²: 0.998)
*   **5-Fold Cross-Validation RMSE**: `0.0846 (+/- 0.0023)`
*   **Use Case**: Precision agriculture predictions for crop yield and growth state modeling.

---

## 4. Unsupervised Health Clustering (K-Means)
To discover phenotypic health patterns without labeled data, an unsupervised K-Means clustering model was evaluated.

*   **Silhouette Score**: **0.1966**
*   **Davies-Bouldin Index**: **1.6112**
*   **Outcome**: Successfully identifies 3 distinct phenotypic health states (Thriving, Struggling, Critical).

---

## 5. Dual-Stream Vision Dashboard Integration
Both the primary **PlantVillage CNN** and the **Production Hybrid SVM** are fully integrated into the real-time Dual-Stream Vision Diagnostics dashboard (Flask API). 

Upon image upload, the unified server utilizes a `ThreadPoolExecutor` to parallelize inference across both models. The dashboard displays predictions from both streams side-by-side in real time, allowing users to cross-reference robust deep learning inferences with the fast biological signal processing model.
