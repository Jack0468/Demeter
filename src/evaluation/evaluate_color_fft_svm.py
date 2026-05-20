import os
import sys
import cv2
import numpy as np
import random
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib

# Dynamic paths
PROJECT_ROOT = Path(os.getcwd())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data/raw/vision/PlantVillage"
MODELS_DIR = PROJECT_ROOT / "models/experimentation"
OUT_DIR = PROJECT_ROOT / "evaluation_outputs/fft_svm_experiment"
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLES_PER_CLASS = 50  # Larger test evaluation set size per class
IMG_SIZE = 64

def get_otsu_mask(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def apply_tapered_mask(img_bgr, mask):
    mask_float = mask.astype(float) / 255.0
    tapered_mask = cv2.GaussianBlur(mask_float, (15, 15), 0)
    tapered_img = (img_bgr.astype(float) * tapered_mask[:, :, np.newaxis]).astype(np.uint8)
    return tapered_img

def apply_inpainting(img_bgr, mask):
    inpaint_mask = cv2.bitwise_not(mask)
    inpainted = cv2.inpaint(img_bgr, inpaint_mask, 5, cv2.INPAINT_TELEA)
    return inpainted

def extract_all_feature_sets(img_path):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Could not load {img_path}")
    
    img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    mask = get_otsu_mask(img_bgr)
    
    # 1. Raw Gray
    gray_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    fft_raw = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray_raw))) + 1).flatten()
    
    # 2. Segmented Flat Black
    segmented_bin = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    gray_bin = cv2.cvtColor(segmented_bin, cv2.COLOR_BGR2GRAY)
    fft_bin = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray_bin))) + 1).flatten()
    
    # 3. Tapered Masking
    tapered_img = apply_tapered_mask(img_bgr, mask)
    gray_tap = cv2.cvtColor(tapered_img, cv2.COLOR_BGR2GRAY)
    fft_tap = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray_tap))) + 1).flatten()
    
    # 4. Inpainted Seamless
    inpainted_img = apply_inpainting(img_bgr, mask)
    gray_inp = cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2GRAY)
    fft_inp = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray_inp))) + 1).flatten()
    
    # 5. Multichannel LAB FFT
    lab = cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    fft_l = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(l))) + 1).flatten()
    fft_a = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(a))) + 1).flatten()
    fft_b = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(b))) + 1).flatten()
    lab_fft_features = np.concatenate([fft_l, fft_a, fft_b])
    
    # 6. Hybrid Grayscale FFT + HSV Histogram
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], mask, [32], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], mask, [32], [0, 256]).flatten()
    h_hist /= h_hist.sum() + 1e-6
    s_hist /= s_hist.sum() + 1e-6
    color_hist = np.concatenate([h_hist, s_hist])
    
    return fft_raw, fft_bin, fft_tap, fft_inp, lab_fft_features, color_hist

def load_eval_data():
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR / d)])
    
    data = {
        "raw": [],
        "bin": [],
        "tap": [],
        "inp": [],
        "lab": [],
        "hist": [],
        "y": []
    }
    
    print(f"Sampling {SAMPLES_PER_CLASS} images per class for test evaluation...")
    random.seed(42) # For reproducible evaluation set
    
    for cls_name in classes:
        cls_path = DATA_DIR / cls_name
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Take a distinct slice for evaluation (e.g. from the end of the lists)
        if len(images) > SAMPLES_PER_CLASS:
            eval_images = images[-SAMPLES_PER_CLASS:]
        else:
            eval_images = images
            
        for img_name in eval_images:
            img_path = cls_path / img_name
            try:
                raw, bin_f, tap, inp, lab, hist = extract_all_feature_sets(img_path)
                data["raw"].append(raw)
                data["bin"].append(bin_f)
                data["tap"].append(tap)
                data["inp"].append(inp)
                data["lab"].append(lab)
                data["hist"].append(hist)
                data["y"].append(cls_name)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                
    return (
        np.array(data["raw"]),
        np.array(data["bin"]),
        np.array(data["tap"]),
        np.array(data["inp"]),
        np.array(data["lab"]),
        np.array(data["hist"]),
        np.array(data["y"]),
        classes
    )

def evaluate_model(X_eval, y_eval, classes, prefix, title):
    scaler_path = MODELS_DIR / f"{prefix}_scaler.joblib"
    pca_path = MODELS_DIR / f"{prefix}_pca.joblib"
    svm_path = MODELS_DIR / f"{prefix}_svm.joblib"
    
    if not (scaler_path.exists() and pca_path.exists() and svm_path.exists()):
        print(f"[Warning] Model files for {title} not found. Skipping.")
        return None
        
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    svm = joblib.load(svm_path)
    
    # Preprocess
    X_scaled = scaler.transform(X_eval)
    X_pca = pca.transform(X_scaled)
    
    # Predict
    y_pred = svm.predict(X_pca)
    
    # Calculate metrics
    acc = accuracy_score(y_eval, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_eval, y_pred, average='macro', zero_division=0)
    
    # Save Confusion Matrix Plot
    cm = confusion_matrix(y_eval, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix: {title}')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{prefix}_confusion_matrix.png")
    plt.close()
    
    print(f"Evaluated {title} -> Accuracy: {acc:.2%}, F1 (macro): {f1:.4f}")
    
    return {
        "pipeline": title,
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1
    }

def evaluate_hybrid_model(X_fft, X_hist, y_eval, classes):
    scaler_fft_path = MODELS_DIR / "hybrid_fft_scaler.joblib"
    scaler_hist_path = MODELS_DIR / "hybrid_hist_scaler.joblib"
    pca_fft_path = MODELS_DIR / "hybrid_fft_pca.joblib"
    svm_path = MODELS_DIR / "hybrid_svm.joblib"
    
    if not (scaler_fft_path.exists() and scaler_hist_path.exists() and pca_fft_path.exists() and svm_path.exists()):
        print("[Warning] Hybrid SVM files not found. Skipping.")
        return None
        
    scaler_fft = joblib.load(scaler_fft_path)
    scaler_hist = joblib.load(scaler_hist_path)
    pca_fft = joblib.load(pca_fft_path)
    svm = joblib.load(svm_path)
    
    # Preprocess
    X_fft_scaled = scaler_fft.transform(X_fft)
    X_fft_pca = pca_fft.transform(X_fft_scaled)
    
    X_hist_scaled = scaler_hist.transform(X_hist)
    
    X_hybrid = np.hstack([X_fft_pca, X_hist_scaled])
    
    # Predict
    y_pred = svm.predict(X_hybrid)
    
    acc = accuracy_score(y_eval, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_eval, y_pred, average='macro', zero_division=0)
    
    # Save Confusion Matrix Plot
    cm = confusion_matrix(y_eval, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix: Hybrid FFT + Color Histogram')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "hybrid_confusion_matrix.png")
    plt.close()
    
    print(f"Evaluated Hybrid SVM -> Accuracy: {acc:.2%}, F1 (macro): {f1:.4f}")
    
    return {
        "pipeline": "Hybrid (Raw FFT + HSV Histogram)",
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1
    }

def evaluate_full_hybrid_model(X_fft, X_hist, y_eval, classes):
    scaler_fft_path = MODELS_DIR / "hybrid_full_fft_scaler.joblib"
    scaler_hist_path = MODELS_DIR / "hybrid_full_hist_scaler.joblib"
    pca_fft_path = MODELS_DIR / "hybrid_full_fft_pca.joblib"
    svm_path = MODELS_DIR / "hybrid_full_svm.joblib"
    
    if not (scaler_fft_path.exists() and scaler_hist_path.exists() and pca_fft_path.exists() and svm_path.exists()):
        print("[Warning] Full Hybrid SVM files not found. Skipping.")
        return None
        
    scaler_fft = joblib.load(scaler_fft_path)
    scaler_hist = joblib.load(scaler_hist_path)
    pca_fft = joblib.load(pca_fft_path)
    svm = joblib.load(svm_path)
    
    # Preprocess
    X_fft_scaled = scaler_fft.transform(X_fft)
    X_fft_pca = pca_fft.transform(X_fft_scaled)
    X_hist_scaled = scaler_hist.transform(X_hist)
    X_hybrid = np.hstack([X_fft_pca, X_hist_scaled])
    
    # Predict
    y_pred = svm.predict(X_hybrid)
    
    acc = accuracy_score(y_eval, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_eval, y_pred, average='macro', zero_division=0)
    
    # Save Confusion Matrix Plot
    cm = confusion_matrix(y_eval, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix: Full-Dataset Hybrid FFT + HSV')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "hybrid_full_confusion_matrix.png")
    plt.close()
    
    print(f"Evaluated Full-Dataset Hybrid SVM -> Accuracy: {acc:.2%}, F1 (macro): {f1:.4f}")
    
    return {
        "pipeline": "Hybrid (Full-Dataset FFT + HSV)",
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1
    }

def main():
    print("--- Demeter Signal & Preprocessing SVM Comparison Suite ---")
    
    # Load data
    X_raw, X_bin, X_tap, X_inp, X_lab, X_hist, y_eval, classes = load_eval_data()
    
    results = []
    
    # 1. Grayscale FFT Raw
    res = evaluate_model(X_raw, y_eval, classes, "fft", "Raw Grayscale FFT (Baseline)")
    if res: results.append(res)
        
    # 2. Segmented Flat Black
    res = evaluate_model(X_bin, y_eval, classes, "segmented_fft", "Binary Segmented FFT (Flat Black)")
    if res: results.append(res)
        
    # 3. Tapered Masking
    res = evaluate_model(X_tap, y_eval, classes, "tapered", "Tapered (Gaussian-faded) FFT")
    if res: results.append(res)
        
    # 4. Inpainted Seamless
    res = evaluate_model(X_inp, y_eval, classes, "inpainted", "Inpainted (Seamless Background) FFT")
    if res: results.append(res)
        
    # 5. Hybrid Raw FFT + HSV Histogram
    res = evaluate_hybrid_model(X_raw, X_hist, y_eval, classes)
    if res: results.append(res)
    
    # 6. Full-Dataset Hybrid Raw FFT + HSV
    res = evaluate_full_hybrid_model(X_raw, X_hist, y_eval, classes)
    if res: results.append(res)
        
    # 7. Multichannel LAB FFT
    res = evaluate_model(X_lab, y_eval, classes, "multichannel_lab_fft", "Multichannel LAB FFT (Color Spectra)")
    if res: results.append(res)
    
    if not results:
        print("No models evaluated. Please train the SVMs first.")
        return
        
    # Save comparison report
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUT_DIR / "fft_svm_comparison.csv", index=False)
    
    # Save a gorgeous comparison bar plot
    plt.figure(figsize=(12, 6))
    
    df_melt = df_results.melt(id_vars="pipeline", value_vars=["accuracy", "f1_macro"], var_name="metric", value_name="score")
    
    sns.barplot(x="pipeline", y="score", hue="metric", data=df_melt, palette="viridis")
    plt.title("Signal Processing & Feature Pipeline Performance Comparison", fontsize=14, fontweight="bold")
    plt.ylabel("Score")
    plt.xlabel("Pipeline Configuration")
    plt.xticks(rotation=15, ha='right')
    plt.ylim(0, 0.65)
    
    # Annotate bars
    for p in plt.gca().patches:
        score = p.get_height()
        if score > 0:
            plt.gca().annotate(f"{score:.1%}", (p.get_x() + p.get_width() / 2., score),
                                ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=9)
            
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pipeline_comparison_chart.png")
    plt.close()
    
    print(f"\nEvaluation comparison report written to {OUT_DIR / 'fft_svm_comparison.csv'}")
    print(f"Gorgeous comparative performance chart saved to {OUT_DIR / 'pipeline_comparison_chart.png'}")
    
    # Display plain Markdown table
    print("\n" + "="*60)
    print("             PIPELINE PERFORMANCE SUMMARY")
    print("="*60)
    print(df_results.to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    main()
