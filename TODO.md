# Demeter Project - Delegated Action Plan

## 1. Jack (Data Pipeline & Architecture)
*Core modularization is complete. Focus is now on strict data ontology and full-scale execution.*

- [x] **Repository Cleanup:** Legacy flat scripts have been removed; project is cleanly modularized into `src/api`, `src/core`, `src/training`, and `src/evaluation`.
- [x] **Dataset 3 Handoff:** `setup_tiller_data.py` is complete and dynamically links images to phenotypic tabular data.
- [ ] **Data Ontology Enforcement:** Ensure the new strict `data/` structure (`data/raw/`, `data/processed/`, `data/logs/`) is created across WSL/production and config files are updated to match.
- [ ] **Full Data Run:** Run the `main.py` pipeline against the full 118GB dataset on the external drive in WSL. Monitor for memory leaks.

## 2. Aman (System Integration & Random Forest)
*The core web integration and rule-based systems are live.*

- [x] **Threshold Logic:** Rule-based health scoring and environmental thresholds successfully moved to `src/core/status_engine.py`.
- [x] **Inference Engine Update:** Continuous growth target from the Danforth RF is integrated into the `generate_complete_diagnosis` JSON payload.
- [ ] **Resolve Dataset Domain Mismatch:** Work with Jack to find a statistically valid way to correlate visual PlantVillage predictions with Danforth environmental predictions, as they currently assume independence.

## 3. Edward (CNN Architecture Development)
*CNN pipelines are robust, now shifting to continuous trajectory prediction.*

- [x] **Tiller Count Regression:** `train_tiller_cnn_regressor` is built in `model_builder.py` to predict continuous physical traits directly from pixels.
- [x] **Disease Classification:** PlantVillage CNN is fully integrated into the unified inference pipeline.
- [ ] **Bellwether Trajectory Mapping:** Transition the original Setaria model from predicting binary water stress to longitudinal tracking (mapping `snapshot` images over time).

## 4. SVM Integration (Health Status)
- [x] **Phase 1 (Bootstrap Data):** Create script to generate the SVM training dataset from `status_engine.py` logic.
- [ ] **Phase 2 (Training):** Build an SVM classifier (`train_health_svm_classifier`) in `model_builder.py` using `StandardScaler` and RBF kernel.
- [ ] **Phase 3 (Inference Update):** Update `inference_engine.py` to route predictions through the SVM instead of the hardcoded `status_engine.py` logic.

## 5. Aneesh (Evaluation Metrics & Testing)
*Evaluation tools are functional but need stricter statistical validation.*

- [x] **Regressor Metrics:** `evaluate_rf.py` now successfully calculates RMSE, MAE, and R2 for the continuous models.
- [ ] **Cross-Validation Strategy:** Implement k-fold cross-validation in the evaluation scripts to ensure models aren't overfitting to the WSL environment setup.
- [ ] **Standardize Test Set Evaluation:** Update evaluation scripts to pull strictly from the `data/processed/test_sets/` partitions to prevent data leakage.


## FROM JACK on 12/05/2026
## DO NOT TOUCH from here down

verify correctness / accruacy.

check baseline model versions so we can compare outputs based on different input techniques.
e.g. no augmenter, different activation functions on CNN

CNN SPECIFIC FEATURES
padding of image pixels to perform convolutions.

evaluation. 

fix connection to dashboard.html

REDO TODO
FIX data dile directory structure 

SVM:

Must re interpret data availible and its useage.
GOAL: use data / regressor model / CNN health to train SVM

or if the outputs of the various models can be fed into a NN

PROMPTS TODO:

CRITICAL INFO TO CONSIDER:
1. The "Frankenstein" Dataset Problem (Domain Mismatch)
The most glaring critical flaw in your data strategy is that you are attempting to build a unified inference pipeline (inference_engine.py) using datasets that are entirely detached from one another in the real world:

PlantVillage Data: Highly controlled, zoomed-in images of individual leaves used for disease classification (e.g., Tomato Early Blight).
Danforth Data: Tabular multi-modal environmental data focusing on Temperature, Soil_Moisture, Sunlight_Hours, and a continuous Growth_Milestone.

Bellwether Dataset: Longitudinal data tracking water amount, weight before, weight after, and snapshot images of full plants for stress detection.

The Critique: You are merging predictions from a CNN trained on single leaves (PlantVillage) with an RF model trained on separate environmental data (Danforth). generate_complete_diagnosis stitches these together into a single JSON payload. This is biologically and statistically flawed. A plant's visual disease symptoms are directly correlated with its environment, but your models treat them as completely independent variables because the training datasets share no intersection.

3. Structural Fragmentation and Clutter
The data/ directory lacks a cohesive structure and is suffering from "dumping ground" syndrome. Based on the file paths hardcoded across your scripts:

Raw vs. Processed Data: You have layer2_health_rgb/PlantVillage and layer3_environment/plant_growth_data.csv acting as raw sources, but then prepare_bellwether_test_set.py dumps bellwether_rf_test.csv and bellwether_test_images/ straight into the root data/ folder.
Artifact Clutter: metadata_cache.pkl is sitting at the root of data/.
Conflicting Log Files: inference_engine.py defines a fallback log path of data/demeter_logs.csv, while SETUP_DASHBOARD.md and the SVM script refer to data/plant_diagnostics.csv. Having multiple overlapping CSV files for logging predictions will lead to missing historical data when evaluating the system's actual performance.

The Critique: The evaluation scripts (evaluate_rf.py) and inference logic are being twisted to handle datasets that don't share the same schema. If the goal is a continuous trajectory prediction, standardizing the target variable (e.g., standardizing on "biomass delta" or "weight after") across all tabular datasets before training is critical. Right now, the data pipeline is duct-taping over schema mismatches.
