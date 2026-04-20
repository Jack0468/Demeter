# Demeter Project - Delegated Action Plan

## 1. Jack (Data Pipeline & Preprocessing)
*Primary pipeline is built. The next steps involve execution and validation.*

- [ ] **Full Data Run:** Run `main.py` against the full dataset on the external drive in WSL. Monitor for memory leaks during the Pandas merge or TensorFlow dataset generation.
- [ ] **Dataset 3 Handoff:** Provide the standalone `load_manual_tiller_data` function to Edward so he can immediately begin his next task without waiting on the 118 GB run.
- [ ] **Repository Cleanup:** Delete the entire `src/` directory (`src/layer1_cnn.py`, `src/layer2_health.py`, `src/layer3_rf.py`, `src/setup_data.py`) and `build_transfer_learning_cnn.py`.

## 2. Aman (Random Forest & System Integration)
*The RF model in `model_builder.py` was recently converted from a Classifier to a Regressor to align with the proposal goals.*

- [ ] **Inference Engine Update:** Update `inference_engine.py`. Rewrite `analyze_plant_status` to handle the new continuous output (predicting the plant's future weight/biomass) instead of binary 1/0.
- [ ] **Threshold Logic:** Establish new logic based on the regression output (e.g., if predicted future weight is < 90% of current weight, trigger a "Needs Fertilizer" warning).

## 3. Edward (CNN Architecture Development)
*The primary CNN classifying `Water_Stressed` vs `Well_Watered` is built. The system needs deeper visual analytics for trajectory goals.*

- [ ] **Tiller Count Regression:** Using Dataset 3 (the 58 manual images), build a secondary CNN architecture that predicts a continuous number (`tiller_count` or `leaf_angle`) directly from a visual input.
- [ ] **Species Classification Integration:** If still required, integrate the PlantVillage/PlantNet model as a parallel pipeline to the Setaria stress model.

## 4. Aneesh (Evaluation Metrics & Testing)
*The `model_evaluation.py` file is currently lagging behind the architectural changes.*

- [ ] **Regressor Metrics:** Write an `evaluate_rf_regressor(rf_model, X_test, y_test)` function that calculates and prints Root Mean Square Error (RMSE).
- [ ] **Cross-Validation Strategy:** Implement k-fold cross-validation in the evaluation scripts to ensure models aren't overfitting to the WSL environment setup.