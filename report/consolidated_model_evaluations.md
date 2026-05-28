# Demeter: Consolidated Model Evaluations

This document aggregates the evaluation results of the various models developed, experimented with, and deployed within the Demeter project.

## 1. Deep Learning Vision Models (CNNs)
A suite of convolutional neural networks (CNNs) was trained and evaluated on various datasets to handle different diagnostic and regression tasks from leaf and plant imagery.

### 1.1 Plant Pathogen Classification (Legacy Generic CNN)
*   **Model File**: `demeter_cnn_plantvillage.keras`
*   **Production Model Accuracy**: **86.20%**
*   **Baseline Run (`eval_run_1`) Accuracy**: **84.31%**
*   **Use Case**: Generic core vision stream for identifying plant diseases and pathogens across all species simultaneously.

### 1.2 Hierarchical Species-Specific Disease Models (New Architecture)
*   **Model Files**: `demeter_cnn_plantvillage_species_identifier.keras`, `demeter_cnn_plantvillage_potato.keras`, etc.
*   **Primary Identifier Accuracy**: **[PLACEHOLDER: Pending background training completion]**
*   **Species-Specific Average Accuracy**: **[PLACEHOLDER: Pending background training completion]**
*   **Use Case**: A two-stage pipeline where a primary routing model first identifies the plant species from the image. Once identified, a dedicated, highly specialized CNN (tailored only to that species) diagnoses the specific pathogen, preventing cross-species misclassification.

### 1.3 Plant Biomass Prediction (Biomass CNN Regressor)
*   **Model File**: `demeter_cnn_biomass.keras`
*   **RMSE**: **2.112**
*   **MAE**: **1.348**
*   **R²**: **0.645**
*   **Use Case**: Predictive modeling of plant fresh weight directly from images.

### 1.4 Tiller Count Prediction (Tiller CNN Regressor)
*   **Model File**: `demeter_cnn_tiller.keras`
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

| Preprocessing Stage | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Key Biological Benefit |
| :--- | :---: | :---: | :---: | :---: | :--- |
| ❌ **Raw Grayscale FFT (Baseline)** | 5.20% | 0.55% | 5.20% | 0.96% | Fails due to high-frequency edge artifacts. |
| 🖤 **Binary Segmented FFT (Flat Mask)** | 8.00% | 16.84% | 8.00% | 4.79% | Isolates the leaf but introduces sharp artificial edge noise. |
| 🌫️ **Tapered FFT (Gaussian-fade)** | 30.00% | 35.40% | 30.00% | 28.41% | Eliminates artificial boundary noise using edge fading. |
| 🩹 **Inpainted FFT (Seamless Pad)** | 29.60% | 40.13% | 29.60% | 27.01% | Replaces background with texture inpaint to minimize edge spikes. |
| 🎨 **Multichannel LAB FFT** | 48.00% | 51.48% | 48.00% | 47.05% | Captures spatial color transitions across channels. |
| 👑 **Production Hybrid FFT + HSV (Monolithic)** | 76.09% | 71.85% | 75.69% | 72.94% | Combines texture frequency and color distributions on full scale. |
| 🚀 **Hierarchical Hybrid SVM (Species-Specific)** | **86.69%** 🏆 | - | - | - | Eliminates cross-species interference via 2-stage routing. |

### 2.3 Hierarchical Hybrid SVM
*   **Approach**: Upgraded the Production Hybrid SVM to a two-stage hierarchical architecture. A primary SVM classifies the species (e.g., Potato vs Tomato), and its output routes to a secondary SVM trained exclusively on that species' classes.
*   **Primary Identifier Accuracy**: **85.71%**
*   **Species-Specific Average Accuracy**: **86.69%**
*   **Conclusion**: Breaking down the problem space dramatically improves diagnostic precision for shallow models. By ensuring early blight in tomatoes is not confused with early blight in potatoes, the hierarchical system outperforms the monolithic SVM by over 10 percentage points.
*   **Conclusion**: Combining texture frequency with color distributions creates a robust shallow classifier that rivals the CNN in accuracy while maintaining a fraction of the computational footprint, making it ideal for edge deployment.

---

## 3. Growth Trajectory and Water Stress Models (Random Forest)
A pair of Random Forest regressors was trained on tabular data to predict environmental milestones and water stress dynamics.

### 3.1 Danforth Growth Predictor
*   **Model File**: `demeter_rf_danforth.joblib`
*   **Production Model RMSE**: **0.291** (R²: 0.662)
*   **Baseline Run (`eval_run_1`) RMSE**: **0.047** (R²: 0.998)
*   **5-Fold Cross-Validation RMSE**: `0.0846 (+/- 0.0023)`
*   **Use Case**: Precision agriculture predictions for crop yield and growth state modeling.


## 4. Unsupervised Domain-Specific Health Clustering (K-Means)
To discover phenotypic health patterns without labeled data and rigid heuristic rules, unsupervised K-Means clustering was adopted. The architecture utilizes three domain-specific models to independently analyze distinct feature streams.

### 4.1 Visual Health Clustering
*   **Model File**: `visual_health_clusters.joblib`
*   **Features Used**: `plantvillage_confidence`, `biomass_weight`, `hybrid_svm_confidence`
*   **Use Case**: Clusters plant health based purely on visual traits and vision model confidences.

### 4.2 Tabular Health Clustering
*   **Model File**: `tabular_health_clusters.joblib`
*   **Features Used**: `predicted_growth_milestone`
*   **Use Case**: Clusters plant health based purely on environmental tabular inputs and Random Forest regression outputs.

### 4.3 Master Health Clustering
*   **Model File**: `master_health_clusters.joblib`
*   **Features Used**: Combined visual and tabular features.
*   **Use Case**: Provides a holistic, unified health status integrating both modalities.

*Note: These K-Means models have completely replaced the static hardcoded heuristics for fertilizer and moisture stress in the Demeter dashboard, enabling entirely data-driven health status inference.*

---

## 5. Dual-Stream Vision Dashboard Integration
Both the primary **PlantVillage CNN** and the **Production Hybrid SVM** are fully integrated into the real-time Dual-Stream Vision Diagnostics dashboard (Flask API). 

Upon image upload, the unified server utilizes a `ThreadPoolExecutor` to parallelize inference across both models. The dashboard displays predictions from both streams side-by-side in real time, allowing users to cross-reference robust deep learning inferences with the fast biological signal processing model.
