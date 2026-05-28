import os
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

PROJECT_ROOT = Path(r"c:\Users\Admin\Documents\Windows_codespace\DEMETER\Demeter")
sys.path.insert(0, str(PROJECT_ROOT))
from src.training.train_hybrid_fft_svm_full import load_data_parallel
MODELS_DIR = PROJECT_ROOT / "models/experimentation"
SPECIES_MODELS_DIR = MODELS_DIR / "species_svms"

def main():
    X_fft, X_hist, y, classes = load_data_parallel()
    
    y_species = np.array([label.split('_')[0] for label in y])
    unique_species = sorted(list(set(y_species)))
    
    # 1. Evaluate Primary Species Identifier
    X_train_fft, X_test_fft, X_train_hist, X_test_hist, y_train, y_test = train_test_split(
        X_fft, X_hist, y_species, test_size=0.20, stratify=y_species, random_state=42
    )
    
    scaler_fft = joblib.load(MODELS_DIR / "hybrid_svm_species_identifier_fft_scaler.joblib")
    scaler_hist = joblib.load(MODELS_DIR / "hybrid_svm_species_identifier_hist_scaler.joblib")
    pca_fft = joblib.load(MODELS_DIR / "hybrid_svm_species_identifier_fft_pca.joblib")
    svm_ident = joblib.load(MODELS_DIR / "hybrid_svm_species_identifier.joblib")
    
    X_test_fft_scaled = scaler_fft.transform(X_test_fft)
    X_test_hist_scaled = scaler_hist.transform(X_test_hist)
    X_test_fft_pca = pca_fft.transform(X_test_fft_scaled)
    X_test_hybrid = np.hstack([X_test_fft_pca, X_test_hist_scaled])
    
    y_pred = svm_ident.predict(X_test_hybrid)
    acc_species_id = accuracy_score(y_test, y_pred)
    print(f"\nPrimary Species Identifier Accuracy: {acc_species_id:.2%}")
    
    # 2. Evaluate Species-Specific Models
    species_accs = {}
    for target_species in unique_species:
        indices = [i for i, label in enumerate(y) if label.startswith(target_species)]
        if len(indices) == 0:
            continue
            
        X_fft_sub = X_fft[indices]
        X_hist_sub = X_hist[indices]
        y_sub = y[indices]
        classes_sub = sorted(list(set(y_sub)))
        
        if len(classes_sub) <= 1:
            species_accs[target_species] = 1.0
            print(f"{target_species} SVM Accuracy: 100.00% (1 class)")
            continue
            
        X_train_fft_s, X_test_fft_s, X_train_hist_s, X_test_hist_s, y_train_s, y_test_s = train_test_split(
            X_fft_sub, X_hist_sub, y_sub, test_size=0.20, stratify=y_sub, random_state=42
        )
        
        prefix = target_species.lower()
        scaler_fft_s = joblib.load(SPECIES_MODELS_DIR / f"hybrid_svm_{prefix}_fft_scaler.joblib")
        scaler_hist_s = joblib.load(SPECIES_MODELS_DIR / f"hybrid_svm_{prefix}_hist_scaler.joblib")
        pca_fft_s = joblib.load(SPECIES_MODELS_DIR / f"hybrid_svm_{prefix}_fft_pca.joblib")
        svm_s = joblib.load(SPECIES_MODELS_DIR / f"hybrid_svm_{prefix}.joblib")
        
        X_test_fft_scaled_s = scaler_fft_s.transform(X_test_fft_s)
        X_test_hist_scaled_s = scaler_hist_s.transform(X_test_hist_s)
        X_test_fft_pca_s = pca_fft_s.transform(X_test_fft_scaled_s)
        X_test_hybrid_s = np.hstack([X_test_fft_pca_s, X_test_hist_scaled_s])
        
        y_pred_s = svm_s.predict(X_test_hybrid_s)
        acc_s = accuracy_score(y_test_s, y_pred_s)
        species_accs[target_species] = acc_s
        print(f"{target_species} SVM Accuracy: {acc_s:.2%}")
        
    avg_species_acc = np.mean(list(species_accs.values()))
    print(f"\nSpecies-Specific Average Accuracy: {avg_species_acc:.2%}")

if __name__ == '__main__':
    main()
