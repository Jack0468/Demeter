import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from pathlib import Path
import random

# Dynamic paths
PROJECT_ROOT = Path(os.getcwd())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.inference_engine import diagnose_plant_disease

# Output path for the artifact
ARTIFACT_DIR = Path(r"C:\Users\Admin\.gemini\antigravity\brain\e9d0c65b-b4b5-466b-bbd5-fe71e179ee9b")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Load CNN Model
cnn_path = str(PROJECT_ROOT / 'models/demeter_cnn_plantvillage.keras')
cnn_model = tf.keras.models.load_model(cnn_path)

# Load class directories
plantvillage_dir = str(PROJECT_ROOT / 'data/raw/vision/PlantVillage')
class_dirs = sorted([d for d in os.listdir(plantvillage_dir) if os.path.isdir(os.path.join(plantvillage_dir, d))])

def get_fft(img_gray):
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return fshift, magnitude_spectrum

def apply_filter(fshift, filter_type, r1=30, r2=60):
    rows, cols = fshift.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    y, x = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((x - ccol)**2 + (y - crow)**2)
    
    if filter_type == 'lowpass':
        mask[dist_from_center <= r1] = 1
    elif filter_type == 'highpass':
        mask[dist_from_center > r1] = 1
    elif filter_type == 'bandpass':
        mask[(dist_from_center >= r1) & (dist_from_center <= r2)] = 1
    elif filter_type == 'bandstop':
        mask[dist_from_center < r1] = 1
        mask[dist_from_center > r2] = 1
        
    return fshift * mask, mask

def get_ifft(fshift_filtered):
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    # Normalize to 0-255
    img_min, img_max = np.min(img_back), np.max(img_back)
    img_back = ((img_back - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    return img_back

def predict_on_img(img_array, title):
    img = Image.fromarray(img_array)
    img_resized = img.resize((150, 150))
    if img_resized.mode != 'RGB':
        img_resized = img_resized.convert('RGB')
    
    img_tensor = tf.expand_dims(np.array(img_resized), 0)
    res = diagnose_plant_disease(img_tensor, "memory", cnn_model, class_dirs)
    print(f"{title:<20} -> {res['Detected_Disease']:<25} (Conf: {res['Disease_Confidence']:.2%})")
    return res['Detected_Disease'], res['Disease_Confidence']

disease_class = next(c for c in class_dirs if "healthy" not in c.lower())
healthy_class = next(c for c in class_dirs if "healthy" in c.lower())

def get_random_image(class_name):
    path = os.path.join(plantvillage_dir, class_name)
    imgs = [f for f in os.listdir(path) if f.endswith((".jpg", ".JPG"))]
    return os.path.join(path, random.choice(imgs))

img_disease_path = get_random_image(disease_class)
img_disease_rgb = np.array(Image.open(img_disease_path).convert('RGB'))
img_disease_gray = np.array(Image.open(img_disease_path).convert('L'))

fshift_d, mag_d = get_fft(img_disease_gray)
filters = ["lowpass", "highpass", "bandpass", "bandstop"]

# Plotting Magnitude and Reconstructions
plt.figure(figsize=(20, 10))

plt.subplot(2, 5, 1)
plt.imshow(img_disease_rgb)
plt.title(f"Raw RGB\n({disease_class})")
plt.axis('off')

plt.subplot(2, 5, 6)
plt.imshow(mag_d, cmap='gray')
plt.title("FFT Magnitude Spectrum")
plt.axis('off')

print("--- EVALUATION ---")
predict_on_img(img_disease_rgb, "Raw RGB")
predict_on_img(img_disease_gray, "Raw Grayscale")

for i, f_type in enumerate(filters):
    fshift_filtered, mask = apply_filter(fshift_d, f_type, 20, 60)
    img_reconstructed = get_ifft(fshift_filtered)
    
    plt.subplot(2, 5, i+2)
    plt.imshow(mask, cmap='gray')
    plt.title(f"{f_type.capitalize()} Mask")
    plt.axis('off')
    
    plt.subplot(2, 5, i+7)
    plt.imshow(img_reconstructed, cmap='gray')
    plt.title(f"{f_type.capitalize()} Recon")
    plt.axis('off')
    
    predict_on_img(img_reconstructed, f_type.capitalize())

plt.tight_layout()
plt.savefig(str(ARTIFACT_DIR / "fft_results.png"))
print(f"Saved plot to {ARTIFACT_DIR / 'fft_results.png'}")
