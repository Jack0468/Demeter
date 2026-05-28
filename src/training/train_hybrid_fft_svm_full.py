import os
import sys
import cv2
import numpy as np
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
from concurrent.futures import ThreadPoolExecutor

# Dynamic paths
PROJECT_ROOT = Path(os.getcwd())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.core.svm_preprocessor import SVMPreprocessor
except ImportError:
    from svm_preprocessor import SVMPreprocessor

DATA_DIR = PROJECT_ROOT / "data/raw/vision/PlantVillage"
MODELS_DIR = PROJECT_ROOT / "models/experimentation"
SPECIES_MODELS_DIR = MODELS_DIR / "species_svms"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SPECIES_MODELS_DIR, exist_ok=True)

IMG_SIZE = 64

def extract_features_single(args):
    img_path, cls_name = args
    try:
        preprocessor = SVMPreprocessor(img_size=IMG_SIZE)
        mag_gray, color_hist = preprocessor.extract_features(img_path)
        return mag_gray, color_hist, cls_name
    except Exception:
        return None

def load_data_parallel():
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR / d)])
    tasks = []
    
    for cls_name in classes:
        cls_path = DATA_DIR / cls_name
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img_name in images:
            tasks.append((cls_path / img_name, cls_name))
            
    print(f"Loading {len(tasks)} images using ThreadPoolExecutor...")
    
    X_gray_fft = []
    X_color_hist = []
    y = []
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        results = executor.map(extract_features_single, tasks)
        
        for res in results:
            if res is not None:
                mag_gray, color_hist, cls_name = res
                X_gray_fft.append(mag_gray)
                X_color_hist.append(color_hist)
                y.append(cls_name)
                
    duration = time.time() - start_time
    print(f"Loaded {len(y)} images successfully in {duration:.2f} seconds.")
    
    return np.array(X_gray_fft), np.array(X_color_hist), np.array(y), classes

def main():
    print("--- Demeter Hybrid FFT + HSV SVM Full Dataset Training ---")
    
    X_fft, X_hist, y, classes = load_data_parallel()
    
    # Train/Test Split
    X_train_fft, X_test_fft, X_train_hist, X_test_hist, y_train, y_test = train_test_split(
        X_fft, X_hist, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Train size: {len(X_train_fft)} | Test size: {len(X_test_fft)}")
    
    # Scaling
    print("Scaling features...")
    scaler_fft = StandardScaler()
    X_train_fft_scaled = scaler_fft.fit_transform(X_train_fft)
    X_test_fft_scaled = scaler_fft.transform(X_test_fft)
    
    scaler_hist = StandardScaler()
    X_train_hist_scaled = scaler_hist.fit_transform(X_train_hist)
    X_test_hist_scaled = scaler_hist.transform(X_test_hist)
    
    # PCA on FFT
    print("Applying PCA on FFT magnitude spectrum...")
    pca_fft = PCA(n_components=100, random_state=42)
    X_train_fft_pca = pca_fft.fit_transform(X_train_fft_scaled)
    X_test_fft_pca = pca_fft.transform(X_test_fft_scaled)
    
    # Concatenate
    X_train_hybrid = np.hstack([X_train_fft_pca, X_train_hist_scaled])
    X_test_hybrid = np.hstack([X_test_fft_pca, X_test_hist_scaled])
    
    # Train SVM
    print("Training SVM Regressor/Classifier on Full Dataset (with probability estimation)...")
    start_time = time.time()
    svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    svm.fit(X_train_hybrid, y_train)
    duration = time.time() - start_time
    print(f"SVM trained successfully in {duration:.2f} seconds.")
    
    # Evaluate
    print("Evaluating model...")
    y_pred = svm.predict(X_test_hybrid)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nFinal Accuracy on Full Dataset: {acc:.2%}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classes, digits=4))
    
    # Save production models
    print("Saving trained production models...")
    joblib.dump(scaler_fft, MODELS_DIR / "hybrid_full_fft_scaler.joblib")
    joblib.dump(scaler_hist, MODELS_DIR / "hybrid_full_hist_scaler.joblib")
    joblib.dump(pca_fft, MODELS_DIR / "hybrid_full_fft_pca.joblib")
    joblib.dump(svm, MODELS_DIR / "hybrid_full_svm.joblib")
    print(f"Production models successfully saved in {MODELS_DIR}/")

def evaluate_hybrid_species_identifier(X_fft, X_hist, y, classes):
    print("\n--- Evaluating Hybrid Species Identifier (Full Dataset) ---")
    
    # Map labels to species
    y_species = np.array([label.split('_')[0] for label in y])
    unique_species = sorted(list(set(y_species)))
    
    X_train_fft, X_test_fft, X_train_hist, X_test_hist, y_train, y_test = train_test_split(
        X_fft, X_hist, y_species, test_size=0.20, stratify=y_species, random_state=42
    )
    
    scaler_fft = StandardScaler()
    X_train_fft_scaled = scaler_fft.fit_transform(X_train_fft)
    X_test_fft_scaled = scaler_fft.transform(X_test_fft)
    
    scaler_hist = StandardScaler()
    X_train_hist_scaled = scaler_hist.fit_transform(X_train_hist)
    X_test_hist_scaled = scaler_hist.transform(X_test_hist)
    
    pca_fft = PCA(n_components=100, random_state=42)
    X_train_fft_pca = pca_fft.fit_transform(X_train_fft_scaled)
    X_test_fft_pca = pca_fft.transform(X_test_fft_scaled)
    
    X_train_hybrid = np.hstack([X_train_fft_pca, X_train_hist_scaled])
    X_test_hybrid = np.hstack([X_test_fft_pca, X_test_hist_scaled])
    
    start_time = time.time()
    svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    svm.fit(X_train_hybrid, y_train)
    duration = time.time() - start_time
    print(f"Primary Species SVM trained successfully in {duration:.2f} seconds.")
    
    y_pred = svm.predict(X_test_hybrid)
    acc = accuracy_score(y_test, y_pred)
    print(f"Species Identifier Accuracy: {acc:.2%}")
    
    joblib.dump(scaler_fft, MODELS_DIR / "hybrid_svm_species_identifier_fft_scaler.joblib")
    joblib.dump(scaler_hist, MODELS_DIR / "hybrid_svm_species_identifier_hist_scaler.joblib")
    joblib.dump(pca_fft, MODELS_DIR / "hybrid_svm_species_identifier_fft_pca.joblib")
    joblib.dump(svm, MODELS_DIR / "hybrid_svm_species_identifier.joblib")
    joblib.dump(unique_species, MODELS_DIR / "hybrid_svm_species_identifier_classes.joblib")
    
    return acc

def evaluate_hybrid_species_specific(X_fft, X_hist, y, target_species):
    print(f"\n--- Evaluating Hybrid Species-Specific Model ({target_species}) (Full Dataset) ---")
    
    # Filter for target species
    indices = [i for i, label in enumerate(y) if label.startswith(target_species)]
    
    if len(indices) == 0:
        print(f"No samples found for {target_species}.")
        return 0.0
        
    X_fft_sub = X_fft[indices]
    X_hist_sub = X_hist[indices]
    y_sub = y[indices]
    classes_sub = sorted(list(set(y_sub)))
    
    if len(classes_sub) <= 1:
        print(f"Only 1 class found for {target_species}. Skipping SVM training.")
        return 1.0
    
    X_train_fft, X_test_fft, X_train_hist, X_test_hist, y_train, y_test = train_test_split(
        X_fft_sub, X_hist_sub, y_sub, test_size=0.20, stratify=y_sub, random_state=42
    )
    
    scaler_fft = StandardScaler()
    X_train_fft_scaled = scaler_fft.fit_transform(X_train_fft)
    X_test_fft_scaled = scaler_fft.transform(X_test_fft)
    
    scaler_hist = StandardScaler()
    X_train_hist_scaled = scaler_hist.fit_transform(X_train_hist)
    X_test_hist_scaled = scaler_hist.transform(X_test_hist)
    
    pca_fft = PCA(n_components=100, random_state=42)
    X_train_fft_pca = pca_fft.fit_transform(X_train_fft_scaled)
    X_test_fft_pca = pca_fft.transform(X_test_fft_scaled)
    
    X_train_hybrid = np.hstack([X_train_fft_pca, X_train_hist_scaled])
    X_test_hybrid = np.hstack([X_test_fft_pca, X_test_hist_scaled])
    
    start_time = time.time()
    svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    svm.fit(X_train_hybrid, y_train)
    duration = time.time() - start_time
    print(f"{target_species} SVM trained successfully in {duration:.2f} seconds.")
    
    y_pred = svm.predict(X_test_hybrid)
    acc = accuracy_score(y_test, y_pred)
    print(f"{target_species} SVM Accuracy: {acc:.2%}")
    
    prefix = target_species.lower()
    joblib.dump(scaler_fft, SPECIES_MODELS_DIR / f"hybrid_svm_{prefix}_fft_scaler.joblib")
    joblib.dump(scaler_hist, SPECIES_MODELS_DIR / f"hybrid_svm_{prefix}_hist_scaler.joblib")
    joblib.dump(pca_fft, SPECIES_MODELS_DIR / f"hybrid_svm_{prefix}_fft_pca.joblib")
IMG_SIZE = 64

def extract_features_single(args):
    img_path, cls_name = args
    try:
        preprocessor = SVMPreprocessor(img_size=IMG_SIZE)
        mag_gray, color_hist = preprocessor.extract_features(img_path)
        return mag_gray, color_hist, cls_name
    except Exception:
        return None

def load_data_parallel():
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR / d)])
    tasks = []
    
    for cls_name in classes:
        cls_path = DATA_DIR / cls_name
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img_name in images:
            tasks.append((cls_path / img_name, cls_name))
            
    print(f"Loading {len(tasks)} images using ThreadPoolExecutor...")
    
    X_gray_fft = []
    X_color_hist = []
    y = []
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        results = executor.map(extract_features_single, tasks)
        
        for res in results:
            if res is not None:
                mag_gray, color_hist, cls_name = res
                X_gray_fft.append(mag_gray)
                X_color_hist.append(color_hist)
                y.append(cls_name)
                
    duration = time.time() - start_time
    print(f"Loaded {len(y)} images successfully in {duration:.2f} seconds.")
    
    return np.array(X_gray_fft), np.array(X_color_hist), np.array(y), classes

def main():
    print("--- Demeter Hybrid FFT + HSV SVM Full Dataset Training ---")
    
    X_fft, X_hist, y, classes = load_data_parallel()
    
    # Train/Test Split
    X_train_fft, X_test_fft, X_train_hist, X_test_hist, y_train, y_test = train_test_split(
        X_fft, X_hist, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Train size: {len(X_train_fft)} | Test size: {len(X_test_fft)}")
    
    # Scaling
    print("Scaling features...")
    scaler_fft = StandardScaler()
    X_train_fft_scaled = scaler_fft.fit_transform(X_train_fft)
    X_test_fft_scaled = scaler_fft.transform(X_test_fft)
    
    scaler_hist = StandardScaler()
    X_train_hist_scaled = scaler_hist.fit_transform(X_train_hist)
    X_test_hist_scaled = scaler_hist.transform(X_test_hist)
    
    # PCA on FFT
    print("Applying PCA on FFT magnitude spectrum...")
    pca_fft = PCA(n_components=100, random_state=42)
    X_train_fft_pca = pca_fft.fit_transform(X_train_fft_scaled)
    X_test_fft_pca = pca_fft.transform(X_test_fft_scaled)
    
    # Concatenate
    X_train_hybrid = np.hstack([X_train_fft_pca, X_train_hist_scaled])
    X_test_hybrid = np.hstack([X_test_fft_pca, X_test_hist_scaled])
    
    # Train SVM
    print("Training SVM Regressor/Classifier on Full Dataset (with probability estimation)...")
    start_time = time.time()
    svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    svm.fit(X_train_hybrid, y_train)
    duration = time.time() - start_time
    print(f"SVM trained successfully in {duration:.2f} seconds.")
    
    # Evaluate
    print("Evaluating model...")
    y_pred = svm.predict(X_test_hybrid)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nFinal Accuracy on Full Dataset: {acc:.2%}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classes, digits=4))
    
    # Save production models
    print("Saving trained production models...")
    joblib.dump(scaler_fft, MODELS_DIR / "hybrid_full_fft_scaler.joblib")
    joblib.dump(scaler_hist, MODELS_DIR / "hybrid_full_hist_scaler.joblib")
    joblib.dump(pca_fft, MODELS_DIR / "hybrid_full_fft_pca.joblib")
    joblib.dump(svm, MODELS_DIR / "hybrid_full_svm.joblib")
    print(f"Production models successfully saved in {MODELS_DIR}/")

def evaluate_hybrid_species_identifier(X_fft, X_hist, y, classes):
    print("\n--- Evaluating Hybrid Species Identifier (Full Dataset) ---")
    
    # Map labels to species
    y_species = np.array([label.split('_')[0] for label in y])
    unique_species = sorted(list(set(y_species)))
    
    X_train_fft, X_test_fft, X_train_hist, X_test_hist, y_train, y_test = train_test_split(
        X_fft, X_hist, y_species, test_size=0.20, stratify=y_species, random_state=42
    )
    
    scaler_fft = StandardScaler()
    X_train_fft_scaled = scaler_fft.fit_transform(X_train_fft)
    X_test_fft_scaled = scaler_fft.transform(X_test_fft)
    
    scaler_hist = StandardScaler()
    X_train_hist_scaled = scaler_hist.fit_transform(X_train_hist)
    X_test_hist_scaled = scaler_hist.transform(X_test_hist)
    
    pca_fft = PCA(n_components=100, random_state=42)
    X_train_fft_pca = pca_fft.fit_transform(X_train_fft_scaled)
    X_test_fft_pca = pca_fft.transform(X_test_fft_scaled)
    
    X_train_hybrid = np.hstack([X_train_fft_pca, X_train_hist_scaled])
    X_test_hybrid = np.hstack([X_test_fft_pca, X_test_hist_scaled])
    
    start_time = time.time()
    svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    svm.fit(X_train_hybrid, y_train)
    duration = time.time() - start_time
    print(f"Primary Species SVM trained successfully in {duration:.2f} seconds.")
    
    y_pred = svm.predict(X_test_hybrid)
    acc = accuracy_score(y_test, y_pred)
    print(f"Species Identifier Accuracy: {acc:.2%}")
    
    joblib.dump(scaler_fft, MODELS_DIR / "hybrid_svm_species_identifier_fft_scaler.joblib")
    joblib.dump(scaler_hist, MODELS_DIR / "hybrid_svm_species_identifier_hist_scaler.joblib")
    joblib.dump(pca_fft, MODELS_DIR / "hybrid_svm_species_identifier_fft_pca.joblib")
    joblib.dump(svm, MODELS_DIR / "hybrid_svm_species_identifier.joblib")
    joblib.dump(unique_species, MODELS_DIR / "hybrid_svm_species_identifier_classes.joblib")
    
    return acc

def evaluate_hybrid_species_specific(X_fft, X_hist, y, target_species):
    print(f"\n--- Evaluating Hybrid Species-Specific Model ({target_species}) (Full Dataset) ---")
    
    # Filter for target species
    indices = [i for i, label in enumerate(y) if label.startswith(target_species)]
    
    if len(indices) == 0:
        print(f"No samples found for {target_species}.")
        return 0.0
        
    X_fft_sub = X_fft[indices]
    X_hist_sub = X_hist[indices]
    y_sub = y[indices]
    classes_sub = sorted(list(set(y_sub)))
    
    if len(classes_sub) <= 1:
        print(f"Only 1 class found for {target_species}. Skipping SVM training.")
        return 1.0
    
    X_train_fft, X_test_fft, X_train_hist, X_test_hist, y_train, y_test = train_test_split(
        X_fft_sub, X_hist_sub, y_sub, test_size=0.20, stratify=y_sub, random_state=42
    )
    
    scaler_fft = StandardScaler()
    X_train_fft_scaled = scaler_fft.fit_transform(X_train_fft)
    X_test_fft_scaled = scaler_fft.transform(X_test_fft)
    
    scaler_hist = StandardScaler()
    X_train_hist_scaled = scaler_hist.fit_transform(X_train_hist)
    X_test_hist_scaled = scaler_hist.transform(X_test_hist)
    
    pca_fft = PCA(n_components=100, random_state=42)
    X_train_fft_pca = pca_fft.fit_transform(X_train_fft_scaled)
    X_test_fft_pca = pca_fft.transform(X_test_fft_scaled)
    
    X_train_hybrid = np.hstack([X_train_fft_pca, X_train_hist_scaled])
    X_test_hybrid = np.hstack([X_test_fft_pca, X_test_hist_scaled])
    
    start_time = time.time()
    svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    svm.fit(X_train_hybrid, y_train)
    duration = time.time() - start_time
    print(f"{target_species} SVM trained successfully in {duration:.2f} seconds.")
    
    y_pred = svm.predict(X_test_hybrid)
    acc = accuracy_score(y_test, y_pred)
    print(f"{target_species} SVM Accuracy: {acc:.2%}")
    
    prefix = target_species.lower()
    joblib.dump(scaler_fft, SPECIES_MODELS_DIR / f"hybrid_svm_{prefix}_fft_scaler.joblib")
    joblib.dump(scaler_hist, SPECIES_MODELS_DIR / f"hybrid_svm_{prefix}_hist_scaler.joblib")
    joblib.dump(pca_fft, SPECIES_MODELS_DIR / f"hybrid_svm_{prefix}_fft_pca.joblib")
    joblib.dump(svm, SPECIES_MODELS_DIR / f"hybrid_svm_{prefix}.joblib")
    joblib.dump(classes_sub, SPECIES_MODELS_DIR / f"hybrid_svm_{prefix}_classes.joblib")
    
    return acc

if __name__ == "__main__":
    main()
    
    print("\n--- Starting Hierarchical SVM Training on Full Dataset ---")
    X_fft, X_hist, y, classes = load_data_parallel()
    
    acc_species_id = evaluate_hybrid_species_identifier(X_fft, X_hist, y, classes)
    
    species_names = sorted(list(set([c.split('_')[0] for c in classes])))
    species_accs = {}
    for s in species_names:
        species_accs[s] = evaluate_hybrid_species_specific(X_fft, X_hist, y, s)
    
    avg_species_acc = np.mean(list(species_accs.values()))
    
    print("\n=================================")
    print("      FINAL COMPARISONS (FULL)")
    print("=================================")
    print("\nHierarchical Hybrid Pipeline:")
    print(f" - Primary Species Identifier:  {acc_species_id:.2%}")
    print(f" - Species-Specific Average:    {avg_species_acc:.2%}")
    print("=================================")
