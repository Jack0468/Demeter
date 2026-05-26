# Domain-Specific K-Means Models Architecture

## Overview
Initially, Demeter utilized a single Unsupervised K-Means clustering model. However, merging completely distinct domains (Visual probabilities and Tabular physical properties) without ground-truth alignment introduced significant noise. 

To resolve this, the clustering architecture has been split into Domain-Specific pipelines.

## 1. Visual Health Clusters
- **Model:** `models/visual_health_clusters.joblib`
- **Features Used:** `plantvillage_confidence`, `biomass_weight`, `hybrid_svm_confidence`
- **Data Source:** Trained on raw vision datasets (e.g., PlantVillage, Biomass).
- **Purpose:** Segments plants entirely on visual characteristics. Useful when hardware only supports camera inputs and no physical sensors are available.

## 2. Tabular Health Clusters
- **Model:** `models/tabular_health_clusters.joblib`
- **Features Used:** `predicted_growth_milestone`
- **Data Source:** Trained on purely tabular records like `danforth_growth.csv`.
- **Purpose:** Classifies plants based on environmental sensor outputs.

## 3. Master Integrated Clusters
- **Model:** `models/master_health_clusters.joblib`
- **Features Used:** `plantvillage_confidence`, `biomass_weight`, `hybrid_svm_confidence`, `predicted_growth_milestone`
- **Data Source:** Trained on the **Bellwether paired dataset**.
- **Purpose:** Because Bellwether includes both an image *and* a physical sensor reading for the exact same physical plant at a specific point in time, we can safely merge visual predictions and tabular predictions to create a master cluster.

## Train/Test Data Architecture Note
*As of writing, the underlying CNNs and RF models require a massive GPU retrain over 20,000+ files to guarantee perfectly clean cluster evaluation. A miniature validation pipeline has been constructed to log training splits to `data_split_manifest.json`, ensuring future K-Means evaluations are completely separated from base-model training sets.*
