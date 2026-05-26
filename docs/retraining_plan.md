# Model Retraining & Data Leakage Prevention Plan

The goal of this implementation is to refactor our training pipeline such that every base model explicitly records the exact data samples it used for training vs. testing into a centralized manifest. This will ensure that our downstream Unsupervised K-Means Models are strictly fitted and evaluated ONLY on the test data from the underlying models, preventing nested data leakage.

## Proposed Changes

### `src/training/vision_models.py`
We will update the CNN training functions to track explicit file paths:
- Modify `train_and_save_cnn_plantvillage` to parse the files assigned to the `train_generator` and `validation_generator` and invoke `split_tracker.update_manifest('plantvillage_cnn', train_files, test_files)`.
- Update `train_tiller_cnn_regressor` and `train_biomass_cnn_regressor` to log the image IDs/paths split by `train_test_split`.

### `src/training/tabular_models.py`
We will update the Random Forest tabular training scripts to track DataFrame indices:
- In `train_and_save_rf_danforth`, after splitting the dataset via `train_test_split`, capture the DataFrame index of the train and test subsets and log them via `update_manifest('danforth_rf', train_indices, test_indices)`.
- Replicate this tracking for the Bellwether Random Forest `train_and_save_rf`.

### `src/training/train_hybrid_fft_svm_full.py` (and related Hybrid SVM scripts)
- Ensure that the Train/Test split for the Hybrid SVM is recorded. Ideally, this model should use the same test set as the `plantvillage_cnn` if possible, but at a minimum, it must record its split into the manifest.

### `src/training/train_kmeans_cluster.py` & Bootstrapping
- We will modify the K-Means cluster training to only use data points that exist in the **TEST** split of all underlying base models required for its features.
- If we bootstrap predictions using intermediate models, we must fetch the `data_split_manifest.json` and isolate the feature generation to exclusively run on the test splits.
- *Note on Test Size*: If the resulting test set intersection is too small, we will adjust the base model `train_test_split` ratio (e.g., from 80/20 to 70/30) to generate sufficient samples for robust clustering.

### `docs/models.md`
- [NEW] A central repository document describing all 10 trained models, their domain, source datasets, and versions. *(Completed)*

## Verification Plan

### Automated Verification
1. We will execute `python src/training/train_pipeline.py` with all `force_retrain` flags toggled on.
2. We will inspect `data/processed/data_split_manifest.json` to verify that `plantvillage_cnn`, `danforth_rf`, `biomass_cnn`, etc. have populated lists of string IDs for their `train` and `test` keys.
3. We will assert that the number of data points used to train the K-Means models is less than or equal to the size of the combined test splits, confirming that K-Means has not been exposed to base model training data.
