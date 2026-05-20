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
from sklearn.metrics import classification_report, accuracy_score
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

def segment_and_enhance(img_path):
    """Loads image, segments background, applies CLAHE, computes FFT, and flattens."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Could not load {img_path}")
    
    # 1. Resize
    img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    
    # 2. Segment (Otsu)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    segmented = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    
    # 3. Enhance (CLAHE)
    lab = cv2.cvtColor(segmented, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # 4. Compute FFT
    gray_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray_enh)
    fshift = np.fft.fftshift(f)
    mag = 20 * np.log(np.abs(fshift) + 1)
    
    return mag.flatten()

def load_data():
    X = []
    y = []
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR / d)])
    
    print(f"Found {len(classes)} classes. Sampling {SAMPLES_PER_CLASS} images per class...")
    
    for cls_idx, cls_name in enumerate(classes):
        cls_path = DATA_DIR / cls_name
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(images) > SAMPLES_PER_CLASS:
            images = random.sample(images, SAMPLES_PER_CLASS)
            
        for img_name in images:
            img_path = cls_path / img_name
            try:
                features = segment_and_enhance(img_path)
                X.append(features)
                y.append(cls_name)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
    return np.array(X), np.array(y), classes

def main():
    print("--- Demeter Segmented FFT SVM Experiment ---")
    
    # 1. Load Data
    X, y, classes = load_data()
    print(f"Extracted features shape: {X.shape}")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, stratify=y, random_state=42)
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    
    # 2. Scale Features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. PCA Dimensionality Reduction
    print("Applying PCA (reducing to 100 components)...")
    pca = PCA(n_components=100, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.2%}")
    
    # 4. Train SVM
    print("Training SVM (RBF Kernel)...")
    svm = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    svm.fit(X_train_pca, y_train)
    
    # 5. Evaluate
    print("Evaluating...")
    y_pred = svm.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n--- RESULTS ---")
    print(f"Accuracy: {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # 6. Save Models
    print(f"Saving models to {MODELS_DIR}...")
    joblib.dump(scaler, MODELS_DIR / "segmented_fft_scaler.joblib")
    joblib.dump(pca, MODELS_DIR / "segmented_fft_pca.joblib")
    joblib.dump(svm, MODELS_DIR / "segmented_fft_svm.joblib")
    print("Done!")

if __name__ == "__main__":
    main()
