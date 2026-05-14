import os
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# --- DYNAMIC PROJECT ROOT RESOLUTION ---
_current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _current_dir.parent.parent if _current_dir.parent.name == "src" else _current_dir.parent

def train_health_clusters(csv_path: str, save_path: str, n_clusters: int = 3):
    """
    Trains an unsupervised K-Means clustering model to group plants into
    health states based on statistical similarities in environmental and visual data.
    
    This replaces the flawed supervised NN/SVM ensemble approach.
    """
    print("Initializing Unsupervised Health Clustering Pipeline (K-Means)...")
    
    if not os.path.exists(csv_path):
        print(f"[!] ERROR: Data file not found at {csv_path}")
        return False
        
    df = pd.read_csv(csv_path)
    
    # Identify available numeric features to cluster on.
    # In a production environment, this should be the joined table of
    # CNN outputs (e.g., disease_confidence) and RF inputs (e.g., Temp, Moisture).
    expected_features = ['Temp', 'Moisture', 'Light', 'disease_confidence', 'water amount']
    features_present = [col for col in expected_features if col in df.columns]
    
    if len(features_present) < 2:
        print("[!] Warning: Few expected features found. Falling back to all numeric columns.")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove known target/label columns from clustering features
        blacklist = ['Growth_Milestone', 'label_int', 'tiller_count', 'weight after']
        features_present = [c for c in numeric_cols if c not in blacklist]
        
    if not features_present:
        print("[!] ERROR: No suitable numeric features found for clustering.")
        return False
        
    print(f"Clustering on features: {features_present}")
    
    df_clean = df.dropna(subset=features_present).copy()
    X = df_clean[features_present]
    
    # 1. Normalize the features. 
    # K-Means is highly sensitive to variance and scale (e.g., Temp is 25, Confidence is 0.8).
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Train K-Means
    print(f"Running K-Means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # 3. Analyze the generated clusters
    df_clean['Cluster'] = cluster_labels
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    print("\n--- Cluster Center Profiles ---")
    for i in range(n_clusters):
        print(f"\nCluster {i} Profile:")
        for j, feature in enumerate(features_present):
            print(f"  - {feature}: {cluster_centers[i][j]:.4f}")
            
    # Save the model and the scaler as a pipeline
    from sklearn.pipeline import Pipeline
    cluster_pipeline = Pipeline([
        ('scaler', scaler),
        ('kmeans', kmeans)
    ])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(cluster_pipeline, save_path)
    print(f"\nK-Means Pipeline successfully saved to {save_path}")
    
    # Optionally save the labeled data for analysis
    output_csv = os.path.join(os.path.dirname(save_path), 'clustered_data.csv')
    df_clean.to_csv(output_csv, index=False)
    print(f"Labeled dataset saved to {output_csv}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train K-Means clustering for plant health states.")
    parser.add_argument("--csv", type=str, required=True, help="Path to the unified CSV dataset.")
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "models" / "health_clusters.joblib"), help="Path to save the pipeline.")
    parser.add_argument("--clusters", type=int, default=3, help="Number of clusters (default 3 for Thriving, Fair, Critical).")
    
    args = parser.parse_args()
    train_health_clusters(args.csv, args.out, args.clusters)
