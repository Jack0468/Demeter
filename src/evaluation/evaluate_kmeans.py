import os
import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from pathlib import Path

# --- DYNAMIC PROJECT ROOT RESOLUTION ---
_current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _current_dir.parent.parent if _current_dir.parent.name == "src" else _current_dir.parent

def evaluate_kmeans(model_path: str, csv_path: str, out_dir: str):
    """
    Evaluates the quality of K-Means clustering using Silhouette Score and 
    generates a 2D PCA scatter plot to visually confirm cluster separation.
    """
    print("\n--- Evaluating K-Means Clusters ---")
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"[!] ERROR: K-Means model not found at {model_path}")
        return False
    if not os.path.exists(csv_path):
        print(f"[!] ERROR: CSV dataset not found at {csv_path}")
        return False
        
    # Load Model Pipeline and Data
    print("Loading pipeline and dataset...")
    pipeline = joblib.load(model_path)
    df = pd.read_csv(csv_path)
    
    # We must extract the exact features the model was trained on
    # In a production script, the pipeline or an attached config should store feature names.
    # We fallback to numeric matching minus blacklisted columns as used in training.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    blacklist = ['Growth_Milestone', 'label_int', 'tiller_count', 'weight after', 'Cluster']
    features = [c for c in numeric_cols if c not in blacklist]
    
    if not features:
        print("[!] ERROR: No suitable numeric features found for evaluation.")
        return False
        
    df_clean = df.dropna(subset=features).copy()
    X = df_clean[features]
    
    if len(X) < 3:
        print("[!] ERROR: Not enough data points to evaluate clusters.")
        return False
        
    # 1. Generate predictions and extract the scaler
    # Since our pipeline is [scaler, kmeans], we can pass raw X
    cluster_labels = pipeline.predict(X)
    df_clean['Cluster'] = cluster_labels
    
    scaler = pipeline.named_steps['scaler']
    kmeans = pipeline.named_steps['kmeans']
    
    # 2. Mathematical Metrics (Require Scaled Data)
    X_scaled = scaler.transform(X)
    
    # Silhouette Score: -1 (bad) to 1 (perfect separation)
    sil_score = silhouette_score(X_scaled, cluster_labels)
    # Davies-Bouldin Index: Lower is better (tighter, more separated clusters)
    db_score = davies_bouldin_score(X_scaled, cluster_labels)
    
    print(f"Silhouette Score: {sil_score:.4f} (Closer to 1 is better)")
    print(f"Davies-Bouldin Index: {db_score:.4f} (Lower is better)")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Model': ['K-Means Pipeline'],
        'Silhouette_Score': [sil_score],
        'Davies_Bouldin_Index': [db_score],
        'Num_Clusters': [kmeans.n_clusters],
        'Num_Samples': [len(X)]
    })
    metrics_path = os.path.join(out_dir, 'kmeans_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")
    
    # 3. Visual Verification using PCA (2D Projection)
    print("Generating 2D PCA visual projection...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    df_clean['PCA1'] = X_pca[:, 0]
    df_clean['PCA2'] = X_pca[:, 1]
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='PCA1', y='PCA2', 
        hue='Cluster', 
        palette='viridis', 
        data=df_clean, 
        alpha=0.7, 
        edgecolor=None
    )
    plt.title(f'K-Means Cluster Projection (PCA)\nSilhouette Score: {sil_score:.2f}')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.legend(title='Health Cluster')
    plt.tight_layout()
    
    plot_path = os.path.join(out_dir, 'kmeans_pca_clusters.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"PCA Plot saved to {plot_path}")
    print("--- K-Means Evaluation Complete ---\n")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate K-Means clustering.")
    parser.add_argument("--model", type=str, required=True, help="Path to the K-Means pipeline (.joblib).")
    parser.add_argument("--csv", type=str, required=True, help="Path to the dataset used for clustering.")
    parser.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "evaluation_outputs" / "kmeans"), help="Output directory.")
    
    args = parser.parse_args()
    evaluate_kmeans(args.model, args.csv, args.out_dir)
