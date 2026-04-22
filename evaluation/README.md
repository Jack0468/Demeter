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
