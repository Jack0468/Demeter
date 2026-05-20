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
os.makedirs(MODELS_DIR, exist_ok=True)

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

def process_tapered(img_bgr, mask):
    """Fades leaf smoothly to black at the boundaries."""
    mask_float = mask.astype(float) / 255.0
    tapered_mask = cv2.GaussianBlur(mask_float, (15, 15), 0)
    tapered_img = (img_bgr.astype(float) * tapered_mask[:, :, np.newaxis]).astype(np.uint8)
    gray = cv2.cvtColor(tapered_img, cv2.COLOR_BGR2GRAY)
    
    # 2D FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = 20 * np.log(np.abs(fshift) + 1)
    return mag.flatten()

def process_inpainted(img_bgr, mask):
    """Seamlessly texture-maps the background with the leaf's boundary pixels."""
    inpaint_mask = cv2.bitwise_not(mask)
    inpainted = cv2.inpaint(img_bgr, inpaint_mask, 5, cv2.INPAINT_TELEA)
    gray = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
    
    # 2D FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = 20 * np.log(np.abs(fshift) + 1)
    return mag.flatten()

def load_data():
    X_tap = []
    X_inp = []
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
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    continue
                img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
                mask = get_otsu_mask(img_bgr)
                
                # 1. Tapered Masking features
                feat_tap = process_tapered(img_bgr, mask)
                X_tap.append(feat_tap)
                
                # 2. Inpainted Padding features
                feat_inp = process_inpainted(img_bgr, mask)
                X_inp.append(feat_inp)
                
                y.append(cls_name)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
    return np.array(X_tap), np.array(X_inp), np.array(y), classes

def evaluate_pipeline(X, y, classes, method_name):
    print(f"\n--- Evaluating {method_name} Pipeline ---")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, stratify=y, random_state=42)
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    
    # Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # PCA
    pca = PCA(n_components=100, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.2%}")
    
    # Train SVM
    svm = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    svm.fit(X_train_pca, y_train)
    
    # Evaluate
    y_pred = svm.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2%}")
    
    # Print sample report
    print(classification_report(y_test, y_pred, target_names=classes, digits=4))
    
    # Save best models
    joblib.dump(scaler, MODELS_DIR / f"{method_name.lower()}_scaler.joblib")
    joblib.dump(pca, MODELS_DIR / f"{method_name.lower()}_pca.joblib")
    joblib.dump(svm, MODELS_DIR / f"{method_name.lower()}_svm.joblib")
    
    return acc

def main():
    print("--- Demeter Boundary Mitigation SVM Experiment ---")
    
    # Load and extract both feature sets
    X_tap, X_inp, y, classes = load_data()
    print(f"Dataset shape: {X_tap.shape[0]} samples with {X_tap.shape[1]} features.")
    
    # Benchmark both
    acc_tap = evaluate_pipeline(X_tap, y, classes, "Tapered")
    acc_inp = evaluate_pipeline(X_inp, y, classes, "Inpainted")
    
    print("\n=================================")
    print("      FINAL COMPARISONS")
    print("=================================")
    print(f"Raw FFT Baseline:          36.39%")
    print(f"Binary Masked Baseline:    25.85%")
    print(f"Tapered (Smoothed) Mask:   {acc_tap:.2%}")
    print(f"Inpainted (Seamless) Mask: {acc_inp:.2%}")
    print("=================================")

if __name__ == "__main__":
    main()
