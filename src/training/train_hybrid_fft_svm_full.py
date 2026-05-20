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

DATA_DIR = PROJECT_ROOT / "data/raw/vision/PlantVillage"
MODELS_DIR = PROJECT_ROOT / "models/experimentation"
os.makedirs(MODELS_DIR, exist_ok=True)

IMG_SIZE = 64

def get_otsu_mask(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def extract_features_single(args):
    img_path, cls_name = args
    try:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return None
        
        img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
        mask = get_otsu_mask(img_bgr)
        
        # 1. Grayscale raw FFT
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        fft_gray = np.fft.fft2(gray)
        mag_gray = 20 * np.log(np.abs(np.fft.fftshift(fft_gray)) + 1).flatten()
        
        # 2. HSV Histogram on segmented region
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], mask, [32], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], mask, [32], [0, 256]).flatten()
        
        h_hist /= h_hist.sum() + 1e-6
        s_hist /= s_hist.sum() + 1e-6
        color_hist = np.concatenate([h_hist, s_hist])
        
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

if __name__ == "__main__":
    main()
