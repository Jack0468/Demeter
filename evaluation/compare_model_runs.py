import os
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def collect_summaries(paths):
    rows = []
    for p in paths:
        if os.path.isdir(p):
            # look for summary.csv in folder
            candidate = os.path.join(p, 'summary.csv')
            if os.path.exists(candidate):
                df = pd.read_csv(candidate)
                rows.append(df)
            else:
                # try common output folder names
                c1 = os.path.join(p, 'evaluation_outputs', 'summary.csv')
                if os.path.exists(c1):
                    rows.append(pd.read_csv(c1))
        elif os.path.isfile(p):
            rows.append(pd.read_csv(p))
        else:
            # glob
            for fp in glob.glob(p):
                if os.path.isfile(fp):
                    rows.append(pd.read_csv(fp))

    if not rows:
        raise FileNotFoundError('No summary files found for the provided paths')

    combined = pd.concat(rows, ignore_index=True, sort=False)
    return combined


def compare(paths, out_csv='evaluation_outputs/comparison.csv', out_dir='evaluation_outputs'):
    os.makedirs(out_dir, exist_ok=True)
    combined = collect_summaries(paths)
    combined.to_csv(out_csv, index=False)
    print('Combined summaries written to', out_csv)

    # Simple plots
    # CNN metrics: accuracy, f1
    if 'accuracy' in combined.columns or 'f1' in combined.columns:
        plt.figure(figsize=(8, 4))
        sns.barplot(x='run' if 'run' in combined.columns else combined.index, y='accuracy', data=combined)
        plt.title('Accuracy by run')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'comparison_accuracy.png'))
        plt.close()

    # RF regression: RMSE/MAE/R2
    if 'RMSE' in combined.columns or 'MAE' in combined.columns or 'R2' in combined.columns:
        metrics = []
        if 'RMSE' in combined.columns:
            metrics.append('RMSE')
        if 'MAE' in combined.columns:
            metrics.append('MAE')
        if 'R2' in combined.columns:
            metrics.append('R2')

        if metrics:
            plt.figure(figsize=(10, 5))
            melted = combined.melt(id_vars=['run'] if 'run' in combined.columns else None, value_vars=metrics, var_name='metric', value_name='value')
            if 'run' in combined.columns:
                sns.barplot(x='run', y='value', hue='metric', data=melted)
                plt.xticks(rotation=45, ha='right')
            else:
                sns.barplot(x=melted.index, y='value', hue='metric', data=melted)
            plt.title('RF regression metrics by run')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'comparison_rf_regression.png'))
            plt.close()

    print('Comparison plots saved to', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare multiple model run summaries')
    parser.add_argument('paths', nargs='+', help='Paths to summary CSVs or folders containing summary.csv (supports glob)')
    parser.add_argument('--out_csv', default='evaluation_outputs/comparison.csv')
    parser.add_argument('--out_dir', default='evaluation_outputs')
    args = parser.parse_args()

    compare(args.paths, out_csv=args.out_csv, out_dir=args.out_dir)
