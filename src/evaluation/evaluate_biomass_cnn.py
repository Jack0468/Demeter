"""
CNN Regressor Evaluation Script for Biomass

Designed for evaluating the Biomass CNN model that outputs continuous numerical values (fresh_weight).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path

# --- DYNAMIC PROJECT ROOT RESOLUTION ---
_current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _current_dir.parent.parent if _current_dir.parent.name == "src" else _current_dir.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_prep.setup_biomass_data import load_biomass_data

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a CNN Regressor for Biomass.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained .keras model.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the manual_biomass_samples.csv.")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to the directory containing images.")
    parser.add_argument("--out_dir", type=str, default="evaluation_outputs/biomass", help="Output directory for metrics.")
    parser.add_argument("--mode", type=str, choices=["simple", "full"], default="full", help="Evaluation mode.")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Loading Biomass data using mapping from {args.csv_path}...")
    df = load_biomass_data(args.csv_path, args.img_dir, multi_angle=True)
    
    if df is None or df.empty:
        print("[!] Failed to load dataset mapping.")
        sys.exit(1)
        
    if 'fresh_weight' not in df.columns:
        print("[!] ERROR: 'fresh_weight' column missing from tabular data.")
        sys.exit(1)

    df = df.dropna(subset=['filepath', 'fresh_weight']).copy()
    print(f"Evaluating on {len(df)} samples...")

    def process_path(filepath, label):
        img = tf.io.read_file(filepath)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [150, 150]) # Use 150x150 as standard
        return img, label

    filepaths = df['filepath'].values
    y_true = df['fresh_weight'].values.astype(np.float32)
    
    ds = tf.data.Dataset.from_tensor_slices((filepaths, y_true))
    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(16).prefetch(tf.data.AUTOTUNE)

    print(f"Loading CNN Regressor from {args.model}...")
    model = tf.keras.models.load_model(args.model)
    
    print("Running predictions...")
    predictions = model.predict(ds).flatten()
    
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    
    print("\n" + "="*30)
    print(" BIOMASS CNN REGRESSOR METRICS")
    print("="*30)
    print(f" RMSE: {rmse:.4f}")
    print(f" MAE:  {mae:.4f}")
    print(f" R²:   {r2:.4f}")
    print("="*30)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    metrics_df = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "R-Squared", "MSE", "Total Samples"],
        "Value": [rmse, mae, r2, mse, len(y_true)]
    })
    metrics_csv = os.path.join(args.out_dir, "biomass_cnn_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    
    if args.mode == "full":
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, predictions, alpha=0.6, color='#2d6a4f', edgecolors='white', s=60)
        
        min_val = min(np.min(y_true), np.min(predictions))
        max_val = max(np.max(y_true), np.max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")
        
        plt.xlabel("Actual Fresh Weight")
        plt.ylabel("Predicted Fresh Weight")
        plt.title(f"Biomass CNN Performance\nRMSE: {rmse:.2f} | R²: {r2:.2f}")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        
        plot_path = os.path.join(args.out_dir, "biomass_cnn_predicted_vs_actual.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()
