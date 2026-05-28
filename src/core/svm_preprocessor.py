import cv2
import numpy as np

class SVMPreprocessor:
    def __init__(self, img_size=64):
        self.img_size = img_size

    def get_otsu_mask(self, img_bgr):
        """Generates a binary mask using Otsu thresholding."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def extract_features(self, img_path):
        """
        Isolate the leaf (remove background) to extract the HSV Color Histogram,
        and perform mathematical operations for the Grayscale FFT magnitude.
        """
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(f"Could not load {img_path}")
        
        img_bgr = cv2.resize(img_bgr, (self.img_size, self.img_size))
        
        # 1. Otsu thresholding (background removal mask)
        mask = self.get_otsu_mask(img_bgr)
        
        # 2. 2D FFT Magnitude Spectrum
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        fft_gray = np.fft.fft2(gray)
        mag_gray = 20 * np.log(np.abs(np.fft.fftshift(fft_gray)) + 1).flatten()
        
        # 3. HSV Histogram on the isolated leaf region
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], mask, [32], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], mask, [32], [0, 256]).flatten()
        
        # Normalize histograms
        h_hist /= h_hist.sum() + 1e-6
        s_hist /= s_hist.sum() + 1e-6
        color_hist = np.concatenate([h_hist, s_hist])
        
        return mag_gray, color_hist

    def preprocess_for_inference(self, img_path, scaler_fft, scaler_hist, pca_fft):
        """
        Extracts features, scales them, applies PCA to FFT, and concatenates 
        them into the final hybrid feature vector used by the SVM.
        """
        mag_gray, color_hist = self.extract_features(img_path)
        
        X_fft_scaled = scaler_fft.transform([mag_gray])
        X_fft_pca = pca_fft.transform(X_fft_scaled)
        X_hist_scaled = scaler_hist.transform([color_hist])
        
        X_hybrid = np.hstack([X_fft_pca, X_hist_scaled])
        return X_hybrid
