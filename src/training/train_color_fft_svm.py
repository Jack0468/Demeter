import os
import sys
import cv2
import numpy as np
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Dynamic paths
PROJECT_ROOT = Path(os.getcwd())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data/raw/vision/PlantVillage"
MODELS_DIR = PROJECT_ROOT / "models/experimentation"
SPECIES_MODELS_DIR = MODELS_DIR / "species_svms"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SPECIES_MODELS_DIR, exist_ok=True)

SAMPLES_PER_CLASS = 70  # 50 train + 20 test
IMG_SIZE = 64

def get_otsu_mask(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def extract_features(img_path):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Could not load {img_path}")
    
    img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    mask = get_otsu_mask(img_bgr)
    
    # --- Feature Extraction 1: Multichannel LAB FFT ---
    # Inpaint background to avoid edge spikes
    inpaint_mask = cv2.bitwise_not(mask)
    inpainted = cv2.inpaint(img_bgr, inpaint_mask, 5, cv2.INPAINT_TELEA)
    
    lab = cv2.cvtColor(inpainted, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    fft_l = np.fft.fft2(l)
    fft_a = np.fft.fft2(a)
    fft_b = np.fft.fft2(b)
    
    mag_l = 20 * np.log(np.abs(np.fft.fftshift(fft_l)) + 1)
    mag_a = 20 * np.log(np.abs(np.fft.fftshift(fft_a)) + 1)
    mag_b = 20 * np.log(np.abs(np.fft.fftshift(fft_b)) + 1)
    
    lab_fft_features = np.concatenate([mag_l.flatten(), mag_a.flatten(), mag_b.flatten()])
    
    # --- Feature Extraction 2: Hybrid (Raw Grayscale FFT + HSV Color Histogram) ---
    # Grayscale raw FFT
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    fft_gray = np.fft.fft2(gray)
    mag_gray = 20 * np.log(np.abs(np.fft.fftshift(fft_gray)) + 1).flatten()
    
    # HSV Histogram on segmented region
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], mask, [32], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], mask, [32], [0, 256]).flatten()
    
    # Normalize histograms
    h_hist /= h_hist.sum() + 1e-6
    s_hist /= s_hist.sum() + 1e-6
    
    color_hist = np.concatenate([h_hist, s_hist])
    
    return lab_fft_features, mag_gray, color_hist

def load_data():
    X_lab_fft = []
    X_gray_fft = []
    X_color_hist = []
    y = []
    
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR / d)])
    print(f"Sampling {SAMPLES_PER_CLASS} images per class...")
    
    for cls_idx, cls_name in enumerate(classes):
        cls_path = DATA_DIR / cls_name
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(images) > SAMPLES_PER_CLASS:
            images = random.sample(images, SAMPLES_PER_CLASS)
            
        for img_name in images:
            img_path = cls_path / img_name
            try:
                lab_fft, gray_fft, color_hist = extract_features(img_path)
                X_lab_fft.append(lab_fft)
                X_gray_fft.append(gray_fft)
                X_color_hist.append(color_hist)
                y.append(cls_name)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
    return np.array(X_lab_fft), np.array(X_gray_fft), np.array(X_color_hist), np.array(y), classes

def evaluate_pipeline(X, y, classes, method_name):
    print(f"\n--- Evaluating {method_name} Pipeline ---")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, stratify=y, random_state=42)
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # PCA
    pca = PCA(n_components=100, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.2%}")
    
    # Train SVM
    svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    svm.fit(X_train_pca, y_train)
    
    # Evaluate
    y_pred = svm.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2%}")
    
    # Save models
    joblib.dump(scaler, MODELS_DIR / f"{method_name.lower().replace(' ', '_')}_scaler.joblib")
    joblib.dump(pca, MODELS_DIR / f"{method_name.lower().replace(' ', '_')}_pca.joblib")
    joblib.dump(svm, MODELS_DIR / f"{method_name.lower().replace(' ', '_')}_svm.joblib")
    
    return acc

def evaluate_hybrid(X_fft, X_hist, y, classes):
    print("\n--- Evaluating Hybrid (FFT + Color Histogram) Pipeline ---")
    
    # Train/Test Split
    X_train_fft, X_test_fft, X_train_hist, X_test_hist, y_train, y_test = train_test_split(
        X_fft, X_hist, y, test_size=0.28, stratify=y, random_state=42
    )
    print(f"Train size: {len(X_train_fft)} | Test size: {len(X_test_fft)}")
    
    # Step 1: Scale both sets independently
    scaler_fft = StandardScaler()
    X_train_fft_scaled = scaler_fft.fit_transform(X_train_fft)
    X_test_fft_scaled = scaler_fft.transform(X_test_fft)
    
    scaler_hist = StandardScaler()
    X_train_hist_scaled = scaler_hist.fit_transform(X_train_hist)
    X_test_hist_scaled = scaler_hist.transform(X_test_hist)
    
    # Step 2: PCA on FFT features to reduce from 4096 to 100
    pca_fft = PCA(n_components=100, random_state=42)
    X_train_fft_pca = pca_fft.fit_transform(X_train_fft_scaled)
    X_test_fft_pca = pca_fft.transform(X_test_fft_scaled)
    
    # Step 3: Concatenate PCA-FFT features (100) with Color Histogram features (64)
    X_train_hybrid = np.hstack([X_train_fft_pca, X_train_hist_scaled])
    X_test_hybrid = np.hstack([X_test_fft_pca, X_test_hist_scaled])
    print(f"Hybrid feature vector shape: {X_train_hybrid.shape}")
    
    # Step 4: Train SVM on Hybrid features
    svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    svm.fit(X_train_hybrid, y_train)
    
    # Evaluate
    y_pred = svm.predict(X_test_hybrid)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2%}")
    print(classification_report(y_test, y_pred, target_names=classes, digits=4))
    
    # Save
    joblib.dump(scaler_fft, MODELS_DIR / "hybrid_fft_scaler.joblib")
    joblib.dump(scaler_hist, MODELS_DIR / "hybrid_hist_scaler.joblib")
    joblib.dump(pca_fft, MODELS_DIR / "hybrid_fft_pca.joblib")
    joblib.dump(svm, MODELS_DIR / "hybrid_svm.joblib")
    
    return acc

def evaluate_hybrid_species_identifier(X_fft, X_hist, y, classes):
    print("\n--- Evaluating Hybrid Species Identifier ---")
    
    # Map labels to species
    y_species = np.array([label.split('_')[0] for label in y])
    unique_species = sorted(list(set(y_species)))
    
    X_train_fft, X_test_fft, X_train_hist, X_test_hist, y_train, y_test = train_test_split(
        X_fft, X_hist, y_species, test_size=0.28, stratify=y_species, random_state=42
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
    
    svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    svm.fit(X_train_hybrid, y_train)
    
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
    print(f"\n--- Evaluating Hybrid Species-Specific Model ({target_species}) ---")
    
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
        X_fft_sub, X_hist_sub, y_sub, test_size=0.28, stratify=y_sub, random_state=42
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
    
    svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    svm.fit(X_train_hybrid, y_train)
    
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

def main():
    print("--- Demeter Color Frequency SVM Experiment ---")
    
    # Extract data
    X_lab_fft, X_gray_fft, X_color_hist, y, classes = load_data()
    
    # Run pipelines
    acc_multichannel = evaluate_pipeline(X_lab_fft, y, classes, "Multichannel LAB FFT")
    acc_hybrid = evaluate_hybrid(X_gray_fft, X_color_hist, y, classes)
    
    # Run Hierarchical pipelines
    acc_species_id = evaluate_hybrid_species_identifier(X_gray_fft, X_color_hist, y, classes)
    
    species_names = sorted(list(set([c.split('_')[0] for c in classes])))
    species_accs = {}
    for s in species_names:
        species_accs[s] = evaluate_hybrid_species_specific(X_gray_fft, X_color_hist, y, s)
    
    avg_species_acc = np.mean(list(species_accs.values()))

    print("\n=================================")
    print("      FINAL COMPARISONS")
    print("=================================")
    print("Grayscale FFT Baselines:")
    print(" - Raw FFT Baseline:           36.39%")
    print(" - Binary Masked Baseline:     25.85%")
    print(" - Seamless Inpainted Baseline: 28.57%")
    print("\nColor Spectra Pipelines:")
    print(f" - Multichannel LAB FFT:       {acc_multichannel:.2%}")
    print(f" - Hybrid (Grayscale FFT+Hist): {acc_hybrid:.2%}")
    print("\nHierarchical Hybrid Pipeline:")
    print(f" - Primary Species Identifier:  {acc_species_id:.2%}")
    print(f" - Species-Specific Average:    {avg_species_acc:.2%}")
    print("=================================")

if __name__ == "__main__":
    main()
