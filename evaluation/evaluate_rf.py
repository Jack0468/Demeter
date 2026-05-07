import os
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_recall_fscore_support, confusion_matrix


def evaluate_rf(model_path, csv_dataset_path, out_dir="evaluation_outputs/rf"):
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"RF model not found at {model_path}")
    if not os.path.exists(csv_dataset_path):
        raise FileNotFoundError(f"CSV dataset not found at {csv_dataset_path}")

    print("Loading Random Forest model...")
    model = joblib.load(model_path)

    print("Loading dataset...")
    df = pd.read_csv(csv_dataset_path)

    # If the tabular CSV has Needs_* columns we treat as multi-output classification
    if {'Needs_Water', 'Needs_Fertilizer', 'Needs_Light'}.issubset(set(df.columns)):
        X = df[['Species_Code', 'Temp', 'Moisture', 'Light']]
        y = df[['Needs_Water', 'Needs_Fertilizer', 'Needs_Light']]

        results = {}
        for col in y.columns:
            y_true = y[col].values
            y_pred = model.predict(X)
            if y_pred.ndim == 2 and y_pred.shape[1] == y.shape[1]:
                y_pred_col = y_pred[:, list(y.columns).index(col)]
            else:
                y_pred_col = y_pred

            acc = accuracy_score(y_true, y_pred_col)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_col, average='binary', zero_division=0)
            cm = confusion_matrix(y_true, y_pred_col)

            results[col] = {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}
            pd.DataFrame([results[col]]).to_csv(os.path.join(out_dir, f'rf_metrics_{col}.csv'), index=False)

            # Save confusion matrix plot
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'RF Confusion Matrix - {col}')
            plt.ylabel('True')
            plt.xlabel('Pred')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'rf_confusion_{col}.png'))
            plt.close()

        # summary
        pd.DataFrame(results).T.to_csv(os.path.join(out_dir, 'rf_classification_metrics.csv'))
        print("Random Forest classification evaluation complete. Results saved to:", out_dir)
        return

    # Otherwise treat as regression
    possible_targets = ['Growth_Score', 'Yield', 'Final_Height', 'Days_to_Maturity']
    target_col = None
    for t in possible_targets:
        if t in df.columns:
            target_col = t
            break

    if target_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            target_col = numeric_cols[-1]
        else:
            raise ValueError('No suitable regression target found in CSV')

    print(f"Using '{target_col}' as regression target.")
    features = df.drop(columns=[target_col])
    X = features.select_dtypes(include=[np.number])
    y_true = df[target_col].values

    print("Generating predictions...")
    y_pred = model.predict(X)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics = {'RMSE': rmse, 'MSE': mse, 'MAE': mae, 'R2': r2}
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, 'rf_regression_metrics.csv'), index=False)

    # Predicted vs Actual plot
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred)
    minv = min(min(y_true), min(y_pred))
    maxv = max(max(y_true), max(y_pred))
    plt.plot([minv, maxv], [minv, maxv], color='red', linestyle='--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'rf_predicted_vs_actual.png'))
    plt.close()

    print("Random Forest regression evaluation complete. Results saved to:", out_dir)


def evaluate_rf_simple(model_path, csv_dataset_path, out_dir='evaluation_outputs/rf_simple'):
    """A minimal RF quick-check: prints regression or classification quick metrics."""
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"RF model not found at {model_path}")
    if not os.path.exists(csv_dataset_path):
        raise FileNotFoundError(f"CSV dataset not found at {csv_dataset_path}")

    print('Loading model ->', model_path)
    model = joblib.load(model_path)
    df = pd.read_csv(csv_dataset_path)

    if 'Needs_Water' in df.columns:
        X = df[['Species_Code', 'Temp', 'Moisture', 'Light']]
        y = df['Needs_Water']
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        pd.DataFrame([{'model': os.path.basename(model_path), 'accuracy': acc, 'num_samples': int(len(y))}]).to_csv(os.path.join(out_dir, 'rf_classification_metrics.csv'), index=False)
        print('Simple RF classification done — saved rf_classification_metrics.csv to', out_dir)
        return

    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        raise ValueError('not enough numeric columns for regression quick-check')

    target = numeric.columns[-1]
    X = numeric.drop(columns=[target])
    y_true = numeric[target].values
    y_pred = model.predict(X)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    pd.DataFrame([{'model': os.path.basename(model_path), 'target': target, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'num_samples': int(len(y_true))}]).to_csv(os.path.join(out_dir, 'rf_regression_metrics.csv'), index=False)
    print('Simple RF regression done — saved rf_regression_metrics.csv to', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a saved Random Forest model (classification or regression)')
    parser.add_argument('--model', type=str, required=True, help='Path to saved RF model (joblib)')
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV dataset used for evaluation')
    parser.add_argument('--out_dir', type=str, default='evaluation_outputs/rf', help='Directory to write outputs')
    parser.add_argument('--mode', choices=['full', 'simple'], default='full', help='Evaluation mode: full (detailed) or simple (quick-check)')
    args = parser.parse_args()

    if args.mode == 'full':
        evaluate_rf(args.model, args.csv, out_dir=args.out_dir)
    else:
        evaluate_rf_simple(args.model, args.csv, out_dir=args.out_dir)
