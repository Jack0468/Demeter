# Evaluation tools for Demeter models

This folder contains scripts to evaluate the CNN (image classifier) and the Random Forest (tabular model), produce slide-ready CSV/PNG outputs, and summarise/compare runs.

Top-level scripts
- `evaluate_cnn.py` — detailed CNN evaluator (also supports `--mode simple` for a quick-check). Exports CSVs and PNGs:
  - `cnn_overall_metrics.csv`, `cnn_per_class_metrics.csv`, `cnn_confusion_matrix.csv` and PNGs
- `evaluate_rf.py` — RF evaluator (regression-first, classification fallback) with `--mode simple`. Exports:
  - Regression: `rf_regression_metrics.csv` (RMSE/MAE/R2) + `rf_predicted_vs_actual.png`
  - Classification: per-target CSVs and confusion matrix PNGs
- `summarise_results.py` — produces `evaluation_outputs/summary.csv` by reading CNN + RF outputs
- `compare_model_runs.py` — combine multiple summaries and make comparison plots
- `run_evaluation_suite.py` — convenience runner that executes CNN → RF → summariser in sequence

Quick examples (zsh)

- Full CNN evaluator:
```bash
python3 evaluation/evaluate_cnn.py --cnn models/demeter_cnn.keras --test_dir data/raw_images/test_set --out_dir evaluation_outputs/cnn --mode full
```

- Simple CNN quick-check (faster / minimal outputs):
```bash
python3 evaluation/evaluate_cnn.py --cnn models/demeter_cnn.keras --test_dir data/raw_images/test_set --out_dir evaluation_outputs/cnn_simple --mode simple
```

- Full RF evaluator (regression-first):
```bash
python3 evaluation/evaluate_rf.py --model models/demeter_rf.joblib --csv data/tabular/eval.csv --out_dir evaluation_outputs/rf --mode full
```

- Run the whole suite (CNN then RF then summarise):
```bash
python3 evaluation/run_evaluation_suite.py --cnn_test_dir data/raw_images/test_set --rf_csv data/tabular/eval.csv --run_name my_run
```

Notes
- Defaults and assumptions:
  - CNN test data: a directory with one subfolder per class (standard Keras layout).
  - RF CSV: if the CSV contains `Needs_Water` / `Needs_Fertilizer` / `Needs_Light` the script will run per-target classification; otherwise it will try to infer a regression target (common names list or last numeric column).
- Use `--mode simple` for quick, minimal checks that write a small `*_metrics.csv` file suitable for slides.
- If you want per-run logs or process isolation (separate Python processes for each evaluator), ask and I can update `run_evaluation_suite.py` to spawn subprocesses and capture logs.

Dependencies
- numpy, pandas, scikit-learn, joblib, matplotlib, seaborn, tensorflow

If anything here should follow a different filename or layout for your workflows, I can adapt the scripts and README to match.
Evaluation tools for Demeter models

This folder contains scripts to evaluate the CNN (image classifier) and the Random Forest (tabular model).

Files
- evaluate_cnn.py - Runs predictions on a directory of labeled test images (one subfolder per class) and outputs accuracy, precision, recall, F1, confusion matrix, and per-class accuracy as CSV and PNG files.
- evaluate_rf.py - Attempts to evaluate a Random Forest model. If the CSV contains the columns 'Needs_Water', 'Needs_Fertilizer', 'Needs_Light' it will compute classification metrics per target. Otherwise it will try to infer a regression target and compute RMSE, MAE, and R² plus a predicted vs actual plot.

Quick usage

1) CNN
python evaluation/evaluate_cnn.py --cnn models/demeter_cnn.keras --test_dir data/raw_images/split_ttv_dataset_type_of_plants/Test_Set_Folder --out_dir evaluation_outputs/cnn

2) Random Forest
python evaluation/evaluate_rf.py --model models/demeter_rf.joblib --csv data/tabular/plant_growth_data.csv --out_dir evaluation_outputs/rf

Requirements
- numpy, pandas, scikit-learn, joblib, matplotlib, seaborn, tensorflow

Notes
- The scripts make conservative assumptions about dataset layout. If your files are located differently or your tabular CSV has different column names, update the arguments or the script accordingly.
