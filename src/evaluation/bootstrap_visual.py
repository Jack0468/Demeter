import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json
from pathlib import Path
import cv2
import sys

_current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _current_dir.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.plantvillage_classes import PLANTVILLAGE_CLASSES

def process_visual_data():
    print("Loading vision models and manifest...")
    cnn_pv_path = PROJECT_ROOT / "models" / "demeter_cnn_plantvillage.keras"
    cnn_pv = tf.keras.models.load_model(str(cnn_pv_path)) if cnn_pv_path.exists() else None
    
    manifest_path = PROJECT_ROOT / "data" / "processed" / "data_split_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    else:
        print("[!] ERROR: No data split manifest found. Run base model training to generate it.")
        return
        
    # Get only the test split for PlantVillage
    pv_test_paths = manifest.get("demeter_cnn_plantvillage", {}).get("test", [])
    if not pv_test_paths:
        print("[!] ERROR: No test paths found in manifest for PlantVillage.")
        return
        
    print(f"Found {len(pv_test_paths)} test images to bootstrap.")
    
    # We use tf.data for fast batched inference
    def process_path(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [150, 150])
        img = img / 255.0
        return img, file_path

    ds = tf.data.Dataset.from_tensor_slices(pv_test_paths)
    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(64).prefetch(tf.data.AUTOTUNE)
    
    results = []
    
    # Predict in batches
    if cnn_pv:
        for images, paths in ds:
            preds = cnn_pv.predict(images, verbose=0)
            confidences = np.max(preds, axis=1)
            
            for path, conf in zip(paths.numpy(), confidences):
                results.append({
                    "image_path": path.decode('utf-8'),
                    "plantvillage_confidence": float(conf),
                    "biomass_weight": np.nan, # Biomass not fully evaluated on PV data
                    "hybrid_svm_confidence": np.nan
                })
                
    if results:
        df = pd.DataFrame(results)
        # Handle Nans so KMeans doesn't fail
        df.fillna(df.mean(numeric_only=True), inplace=True)
        out_path = PROJECT_ROOT / "data" / "processed" / "visual_features_test.csv"
        os.makedirs(out_path.parent, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} vision records to {out_path}")

if __name__ == "__main__":
    process_visual_data()
