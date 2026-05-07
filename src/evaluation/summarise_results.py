import os
import argparse
import pandas as pd
from pathlib import Path

# --- DYNAMIC PROJECT ROOT RESOLUTION ---
_current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _current_dir.parent.parent if _current_dir.parent.name == "src" else _current_dir.parent

def find_file_if_exists(folder, patterns):
    for p in patterns:
        path = os.path.join(folder, p)
        if os.path.exists(path):
            return path
    return None


def summarise(cnn_out=None, rf_out=None, out_file=None, run_name=None):
    if cnn_out is None:
        cnn_out = str(PROJECT_ROOT / 'evaluation_outputs/cnn')
    if rf_out is None:
        rf_out = str(PROJECT_ROOT / 'evaluation_outputs/rf')
    if out_file is None:
        out_file = str(PROJECT_ROOT / 'evaluation_outputs/summary.csv')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    rows = []

    # CNN overall metrics
    cnn_metrics_file = find_file_if_exists(cnn_out, ['cnn_overall_metrics.csv', 'overall_metrics.csv'])
    if cnn_metrics_file:
        cnn_df = pd.read_csv(cnn_metrics_file)
        # flatten to single row
        cnn_row = {'model': 'cnn'}
        for c in cnn_df.columns:
            cnn_row[c] = cnn_df.iloc[0][c]
        if run_name:
            cnn_row['run'] = run_name
        rows.append(cnn_row)

    # RF regression metrics
    rf_reg_file = find_file_if_exists(rf_out, ['rf_regression_metrics.csv', 'rf_regression.csv', 'rf_metrics_regression.csv'])
    if rf_reg_file:
        rf_df = pd.read_csv(rf_reg_file)
        rf_row = {'model': 'rf'}
        for c in rf_df.columns:
            rf_row[c] = rf_df.iloc[0][c]
        if run_name:
            rf_row['run'] = run_name
        rows.append(rf_row)

    # RF classification metrics (fallback)
    # Note: we treat regression as primary. If both regression and classification
    # metric files exist, the regression metrics will be used. If you prefer
    # classification to take precedence, change the order here.
    rf_clf_file = find_file_if_exists(rf_out, ['rf_classification_metrics.csv', 'rf_classification.csv'])
    if rf_clf_file and not rf_reg_file:
        rf_df = pd.read_csv(rf_clf_file)
        # expand per-class columns into single aggregated row (mean)
        rf_row = {'model': 'rf'}
        rf_row['accuracy_mean'] = rf_df['accuracy'].mean()
        rf_row['precision_mean'] = rf_df['precision'].mean()
        rf_row['recall_mean'] = rf_df['recall'].mean()
        rf_row['f1_mean'] = rf_df['f1'].mean()
        if run_name:
            rf_row['run'] = run_name
        rows.append(rf_row)

    if not rows:
        raise FileNotFoundError('No known metric files found in cnn or rf output folders')

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_file, index=False)
    print('Summary written to', out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarise evaluation outputs from CNN and RF into one CSV')
    parser.add_argument('--cnn_out', default=str(PROJECT_ROOT / 'evaluation_outputs/cnn'))
    parser.add_argument('--rf_out', default=str(PROJECT_ROOT / 'evaluation_outputs/rf'))
    parser.add_argument('--out_file', default=str(PROJECT_ROOT / 'evaluation_outputs/summary.csv'))
    args = parser.parse_args()

    summarise(args.cnn_out, args.rf_out, args.out_file)
