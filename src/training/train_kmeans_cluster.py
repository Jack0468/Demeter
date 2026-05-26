import os
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from pathlib import Path

_current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _current_dir.parent.parent if _current_dir.parent.name == "src" else _current_dir.parent

def train_health_clusters(csv_path: str, save_path: str, features_present: list, n_clusters: int = 3):
    print(f"Initializing Unsupervised Health Clustering for {save_path}...")
    
    if not os.path.exists(csv_path):
        print(f"[!] ERROR: Data file not found at {csv_path}")
        return False
        
    df = pd.read_csv(csv_path)
    available_features = [col for col in features_present if col in df.columns]
    
    if len(available_features) < 1:
        print("[!] ERROR: No suitable numeric features found for clustering.")
        return False
        
    print(f"Clustering on features: {available_features}")
    
    df_clean = df.dropna(subset=available_features).copy()
    if df_clean.empty:
        print("[!] ERROR: Dataset is empty after dropping NaNs.")
        return False
        
    X = df_clean[available_features]
    
    # We use the entire bootstrapped CSV (which is already restricted to the base models' Test splits)
    # We will do a further 80/20 train/test split purely for K-Means evaluation
    from sklearn.model_selection import train_test_split
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Running K-Means with {n_clusters} clusters on {len(X_train)} train samples...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    
    # Train only on X_train
    kmeans.fit(X_train_scaled)
    
    # Evaluate on X_test
    try:
        test_labels = kmeans.predict(X_test_scaled)
        if len(set(test_labels)) > 1:
            score = silhouette_score(X_test_scaled, test_labels)
            print(f"Silhouette Score on Test Set: {score:.4f}")
        else:
            print("Silhouette Score: N/A (Only 1 cluster found in test set)")
    except Exception as e:
        print(f"Could not compute silhouette score: {e}")
        
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    print("\n--- Cluster Center Profiles ---")
    for i in range(len(cluster_centers)):
        print(f"\nCluster {i} Profile:")
        for j, feature in enumerate(available_features):
            print(f"  - {feature}: {cluster_centers[i][j]:.4f}")
            
    cluster_pipeline = Pipeline([
        ('scaler', scaler),
        ('kmeans', kmeans)
    ])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(cluster_pipeline, save_path)
    print(f"Pipeline successfully saved to {save_path}\n")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train K-Means clustering for plant health states.")
    parser.add_argument("--clusters", type=int, default=3, help="Number of clusters.")
    args = parser.parse_args()
    
    # 1. Visual Clusters
    visual_csv = PROJECT_ROOT / "data" / "processed" / "visual_features_test.csv"
    visual_out = PROJECT_ROOT / "models" / "visual_health_clusters.joblib"
    visual_feats = ['plantvillage_confidence', 'biomass_weight', 'hybrid_svm_confidence']
    train_health_clusters(str(visual_csv), str(visual_out), visual_feats, args.clusters)
    
    # 2. Tabular Clusters
    tabular_csv = PROJECT_ROOT / "data" / "processed" / "tabular_features_test.csv"
    tabular_out = PROJECT_ROOT / "models" / "tabular_health_clusters.joblib"
    tabular_feats = ['predicted_growth_milestone']
    train_health_clusters(str(tabular_csv), str(tabular_out), tabular_feats, args.clusters)
