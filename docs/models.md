# Demeter Model Inventory

This document serves as the master record for all machine learning models actively used in the Demeter project. It tracks their architecture, purpose, source datasets, and versions.

| Model File | Architecture | Domain | Purpose | Dataset Source |
| :--- | :--- | :--- | :--- | :--- |
| `demeter_cnn_plantvillage.keras` | Convolutional Neural Network (CNN) | Vision | Predicts plant disease from leaf imagery. | PlantVillage (15 classes) |
| `demeter_rf_danforth.joblib` | Random Forest Regressor | Tabular | Predicts plant growth milestones based on environmental heuristics (soil type, sunlight, water, temp, humidity). | Danforth Environment Dataset |
| `demeter_cnn_biomass.keras` | Convolutional Neural Network (CNN) | Vision | Continuous prediction of plant fresh biomass weight (in grams). | Manual Biomass Measurements |
| `demeter_cnn_tiller.keras` | Convolutional Neural Network (CNN) | Vision | Continuous prediction of tiller counts. | Manual Tiller Data |
| `demeter_cnn.keras` | Convolutional Neural Network (CNN) | Vision | Bellwether water stress classification (Water_Stressed vs Well_Watered). | Bellwether Dataset |
| `demeter_rf.joblib` | Random Forest Regressor | Tabular | Bellwether biomass/tiller fallback predictions based on snapshot physical area. | Bellwether Dataset |
| `hybrid_full_svm.joblib` | SVM Classifier | Vision (Signal) | Secondary disease verification using Fast Fourier Transform (FFT) preprocessing on biological signals. | PlantVillage |
| `visual_health_clusters.joblib` | K-Means | Unsupervised | Clusters plants into unsupervised health groups (0, 1, 2) based on visual outputs (Disease Confidence, Biomass, SVM Confidence). | Bootstrapped Test Sets |
| `tabular_health_clusters.joblib` | K-Means | Unsupervised | Clusters plants based purely on tabular RF growth predictions. | Bootstrapped Test Sets |
| `master_health_clusters.joblib` | K-Means | Unsupervised | Holistic health clustering merging both Visual and Tabular feature outputs. | Bootstrapped Test Sets |

## Data Leakage & Version Control

To prevent data leakage, Demeter utilizes a `split_tracker.py` module. During training, every model records the explicit file paths (for vision) or row indices (for tabular) that were used in its `train` and `test` sets to `data/processed/data_split_manifest.json`.

Unsupervised evaluation models (like the K-Means clusters) are strictly fitted and evaluated using ONLY the data in the underlying models' `test` split.

*Last Retrained: Pending Pipeline Execution*
