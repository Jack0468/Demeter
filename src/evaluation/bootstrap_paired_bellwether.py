import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
import cv2
import sys

# Add src to path
_current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _current_dir.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.core.inference_engine import (
    diagnose_plant_disease,
    predict_biomass,
    predict_hybrid_disease,
    predict_growth_milestone
)
from src.core.plantvillage_classes import PLANTVILLAGE_CLASSES

def process_paired_data(max_samples=50):
    print("Loading models for paired dataset...")
    cnn_pv_path = PROJECT_ROOT / "models" / "demeter_cnn_plantvillage.keras"
    cnn_bio_path = PROJECT_ROOT / "models" / "demeter_cnn_biomass.keras"
    rf_path = PROJECT_ROOT / "models" / "demeter_rf_danforth.joblib"
    svm_dir = PROJECT_ROOT / "models" / "experimentation"
    
    cnn_pv = tf.keras.models.load_model(str(cnn_pv_path)) if cnn_pv_path.exists() else None
    cnn_bio = tf.keras.models.load_model(str(cnn_bio_path)) if cnn_bio_path.exists() else None
    rf_model = joblib.load(str(rf_path)) if rf_path.exists() else None
    
    try:
        scaler_fft = joblib.load(str(svm_dir / "hybrid_full_fft_scaler.joblib"))
        scaler_hist = joblib.load(str(svm_dir / "hybrid_full_hist_scaler.joblib"))
        pca_fft = joblib.load(str(svm_dir / "hybrid_full_fft_pca.joblib"))
        svm = joblib.load(str(svm_dir / "hybrid_full_svm.joblib"))
    except:
        svm = None

    class_names_pv = PLANTVILLAGE_CLASSES
    
    bellwether_dir = PROJECT_ROOT / "data" / "raw" / "vision" / "Bellwether"
    csv_path = bellwether_dir / "SnapshotInfo.csv"
    images_dir = bellwether_dir / "images"
    
    if not csv_path.exists():
        print(f"Paired data CSV not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    print(f"Loaded Bellwether SnapshotInfo with {len(df)} rows.")
    
    results = []
    processed = 0
    
    for idx, row in df.iterrows():
        if processed >= max_samples:
            break
            
        snapshot_id = row.get("id")
        if pd.isna(snapshot_id): continue
        
        # Find image directory
        snap_dir = images_dir / f"snapshot{int(snapshot_id)}"
        if not snap_dir.exists():
            continue
            
        # Find first image in that snapshot directory
        imgs = glob.glob(str(snap_dir / "*.png")) + glob.glob(str(snap_dir / "*.jpg"))
        if not imgs:
            continue
            
        img_path = imgs[0]
        
        # --- VISION PREDICTIONS ---
        pv_conf = np.nan
        bio_weight = np.nan
        hybrid_conf = np.nan
        
        img_bgr = cv2.imread(img_path)
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (150, 150))
            img_array = np.expand_dims(img_resized, axis=0) / 255.0
            
            if cnn_pv:
                pv_res = diagnose_plant_disease(img_array, img_path, cnn_pv, class_names_pv)
                pv_conf = pv_res["Disease_Confidence"]
            if cnn_bio:
                bio_weight = predict_biomass(img_array, cnn_bio)
            if svm:
                try:
                    hybrid_res = predict_hybrid_disease(img_path, scaler_fft, scaler_hist, pca_fft, svm, class_names_pv)
                    hybrid_conf = hybrid_res["confidence"]
                except: pass
                
        # --- TABULAR PREDICTIONS ---
        pred_growth = np.nan
        if rf_model:
            # Map Bellwether data to RF expected format roughly for the proof of concept
            # RF expects: ['Soil_Type', 'Sunlight_Hours', 'Water_Frequency', 'Fertilizer_Type', 'Temperature', 'Humidity']
            # We map water_amount to Water_Frequency, and impute the rest.
            water_amt = row.get("water amount", 0)
            env_data = {
                "Soil_Type": 1,
                "Sunlight_Hours": 12.0,
                "Water_Frequency": float(water_amt) / 100.0 if not pd.isna(water_amt) else 1.0,
                "Fertilizer_Type": 1,
                "Temperature": 24.0,
                "Humidity": 60.0
            }
            try:
                res = predict_growth_milestone(env_data, rf_model)
                pred_growth = res["Predicted_Growth_Milestone"]
            except:
                pass
                
        results.append({
            "snapshot_id": snapshot_id,
            "plantvillage_confidence": pv_conf,
            "biomass_weight": bio_weight,
            "hybrid_svm_confidence": hybrid_conf,
            "predicted_growth_milestone": pred_growth
        })
        processed += 1
        
    if results:
        out_df = pd.DataFrame(results)
        # Drop any rows where ALL predictions failed
        out_df.dropna(subset=['plantvillage_confidence', 'biomass_weight', 'hybrid_svm_confidence', 'predicted_growth_milestone'], how='all', inplace=True)
        # Fill remaining nans with means so KMeans doesn't crash
        out_df.fillna(out_df.mean(numeric_only=True), inplace=True)
        
        out_path = PROJECT_ROOT / "data" / "processed" / "paired_integrated_features.csv"
        os.makedirs(out_path.parent, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"Saved {len(out_df)} paired integrated records to {out_path}")

if __name__ == "__main__":
    process_paired_data()
