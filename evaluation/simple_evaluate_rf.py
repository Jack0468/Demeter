import os
import argparse
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score


def evaluate_simple_rf(model_path, csv_path):
    print('loading rf ->', model_path)
    model = joblib.load(model_path)

    print('loading csv ->', csv_path)
    df = pd.read_csv(csv_path)

    # if dataset has Needs_Water then do quick classification check
    if 'Needs_Water' in df.columns:
        X = df[['Species_Code', 'Temp', 'Moisture', 'Light']]
        y = df['Needs_Water']
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        print('\nquick RF check (Needs_Water):')
        print('samples ->', len(y))
        print('accuracy ->', f'{acc:.4f}')
        return

    # otherwise try regression - use last numeric column as target
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        print('not enough numeric cols to eval regression')
        return

    target = numeric.columns[-1]
    X = numeric.drop(columns=[target])
    y_true = numeric[target].values
    y_pred = model.predict(X)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print('\nquick RF regression:')
    print('target ->', target)
    print('samples ->', len(y_true))
    print('rmse ->', f'{rmse:.4f}')
    print('mae ->', f'{mae:.4f}')
    print('r2 ->', f'{r2:.4f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Simple RF eval')
    p.add_argument('--model', required=True, help='path to joblib model')
    p.add_argument('--csv', required=True, help='path to csv data')
    args = p.parse_args()
    evaluate_simple_rf(args.model, args.csv)
