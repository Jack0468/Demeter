import json
import os

nb = {
 'cells': [
  {
   'cell_type': 'markdown',
   'metadata': {},
   'source': [
    '# Discrete-Time Fourier Transform (DTFT) Image Preprocessing\n',
    'This notebook explores applying 2D FFT to agricultural imagery (PlantVillage). ',
    'By shifting into the frequency domain, we apply High-Pass, Low-Pass, Band-Pass, and Band-Stop filters to see if we can isolate disease lesions or structural patterns for better CNN classification.'
   ]
  },
  {
   'cell_type': 'code',
   'execution_count': None,
   'metadata': {},
   'outputs': [],
   'source': [
    'import os\n',
    'import sys\n',
    'import numpy as np\n',
    'import matplotlib.pyplot as plt\n',
    'import cv2\n',
    'import tensorflow as tf\n',
    'from pathlib import Path\n',
    '\n',
    '# Dynamic paths\n',
    'PROJECT_ROOT = Path(os.getcwd()).parent if Path(os.getcwd()).name == "notebooks" else Path(os.getcwd())\n',
    'if str(PROJECT_ROOT) not in sys.path:\n',
    '    sys.path.insert(0, str(PROJECT_ROOT))\n',
    '\n',
    'try:\n',
    '    from src.core.inference_engine import diagnose_plant_disease\n',
    'except ModuleNotFoundError:\n',
    '    from inference_engine import diagnose_plant_disease\n',
    '\n',
    '# Load CNN Model\n',
    'cnn_path = str(PROJECT_ROOT / "models/demeter_cnn_plantvillage.keras")\n',
    'print("Loading CNN...")\n',
    'cnn_model = tf.keras.models.load_model(cnn_path)\n',
    '\n',
    '# Load class directories\n',
    'plantvillage_dir = str(PROJECT_ROOT / "data/raw/vision/PlantVillage")\n',
    'class_dirs = sorted([d for d in os.listdir(plantvillage_dir) if os.path.isdir(os.path.join(plantvillage_dir, d))])\n',
    'print("Loaded! Classes:", len(class_dirs))'
   ]
  },
  {
   'cell_type': 'markdown',
   'metadata': {},
   'source': [
    '## 1. Define FFT Filter Functions'
   ]
  },
  {
   'cell_type': 'code',
   'execution_count': None,
   'metadata': {},
   'outputs': [],
   'source': [
    'def get_fft(img_gray):\n',
    '    """Compute 2D FFT and shift zero-frequency to center"""\n',
    '    f = np.fft.fft2(img_gray)\n',
    '    fshift = np.fft.fftshift(f)\n',
    '    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)\n',
    '    return fshift, magnitude_spectrum\n',
    '\n',
    'def apply_filter(fshift, filter_type, r1=30, r2=60):\n',
    '    rows, cols = fshift.shape\n',
    '    crow, ccol = rows // 2, cols // 2\n',
    '    \n',
    '    # Create mask\n',
    '    mask = np.zeros((rows, cols), np.uint8)\n',
    '    \n',
    '    y, x = np.ogrid[:rows, :cols]\n',
    '    dist_from_center = np.sqrt((x - ccol)**2 + (y - crow)**2)\n',
    '    \n',
    '    if filter_type == "lowpass":\n',
    '        mask[dist_from_center <= r1] = 1\n',
    '    elif filter_type == "highpass":\n',
    '        mask[dist_from_center > r1] = 1\n',
    '    elif filter_type == "bandpass":\n',
    '        mask[(dist_from_center >= r1) & (dist_from_center <= r2)] = 1\n',
    '    elif filter_type == "bandstop":\n',
    '        mask[dist_from_center < r1] = 1\n',
    '        mask[dist_from_center > r2] = 1\n',
    '        \n',
    '    return fshift * mask, mask\n',
    '\n',
    'def get_ifft(fshift_filtered):\n',
    '    """Inverse FFT to get spatial image back"""\n',
    '    f_ishift = np.fft.ifftshift(fshift_filtered)\n',
    '    img_back = np.fft.ifft2(f_ishift)\n',
    '    img_back = np.abs(img_back)\n',
    '    # Normalize to 0-255\n',
    '    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)\n',
    '    return img_back'
   ]
  },
  {
   'cell_type': 'markdown',
   'metadata': {},
   'source': [
    '## 2. Load Sample Images (Diseased & Healthy)'
   ]
  },
  {
   'cell_type': 'code',
   'execution_count': None,
   'metadata': {},
   'outputs': [],
   'source': [
    'import random\n',
    '\n',
    '# Find a diseased class and a healthy class\n',
    'disease_class = next(c for c in class_dirs if "healthy" not in c.lower())\n',
    'healthy_class = next(c for c in class_dirs if "healthy" in c.lower())\n',
    '\n',
    'def get_random_image(class_name):\n',
    '    path = os.path.join(plantvillage_dir, class_name)\n',
    '    imgs = [f for f in os.listdir(path) if f.endswith((".jpg", ".JPG"))]\n',
    '    return os.path.join(path, random.choice(imgs))\n',
    '\n',
    'img_disease_path = get_random_image(disease_class)\n',
    'img_healthy_path = get_random_image(healthy_class)\n',
    '\n',
    '# Load grayscale for FFT and RGB for visualization\n',
    'img_disease_rgb = cv2.cvtColor(cv2.imread(img_disease_path), cv2.COLOR_BGR2RGB)\n',
    'img_disease_gray = cv2.cvtColor(img_disease_rgb, cv2.COLOR_RGB2GRAY)\n',
    '\n',
    'img_healthy_rgb = cv2.cvtColor(cv2.imread(img_healthy_path), cv2.COLOR_BGR2RGB)\n',
    'img_healthy_gray = cv2.cvtColor(img_healthy_rgb, cv2.COLOR_RGB2GRAY)\n',
    '\n',
    'plt.figure(figsize=(10, 5))\n',
    'plt.subplot(1, 2, 1), plt.imshow(img_disease_rgb), plt.title(f"Diseased: {disease_class}")\n',
    'plt.subplot(1, 2, 2), plt.imshow(img_healthy_rgb), plt.title(f"Healthy: {healthy_class}")\n',
    'plt.show()'
   ]
  },
  {
   'cell_type': 'markdown',
   'metadata': {},
   'source': [
    '## 3. Visualize Magnitude Spectrums'
   ]
  },
  {
   'cell_type': 'code',
   'execution_count': None,
   'metadata': {},
   'outputs': [],
   'source': [
    'fshift_d, mag_d = get_fft(img_disease_gray)\n',
    'fshift_h, mag_h = get_fft(img_healthy_gray)\n',
    '\n',
    'plt.figure(figsize=(10, 5))\n',
    'plt.subplot(1, 2, 1), plt.imshow(mag_d, cmap="gray"), plt.title("Diseased FFT Magnitude")\n',
    'plt.subplot(1, 2, 2), plt.imshow(mag_h, cmap="gray"), plt.title("Healthy FFT Magnitude")\n',
    'plt.show()'
   ]
  },
  {
   'cell_type': 'markdown',
   'metadata': {},
   'source': [
    '## 4. Apply Filters & Inverse FFT (Reconstruction)\n',
    'Let us see how different filters affect the diseased image.'
   ]
  },
  {
   'cell_type': 'code',
   'execution_count': None,
   'metadata': {},
   'outputs': [],
   'source': [
    'filters = ["lowpass", "highpass", "bandpass", "bandstop"]\n',
    'r1, r2 = 20, 60\n',
    '\n',
    'plt.figure(figsize=(15, 10))\n',
    'for i, f_type in enumerate(filters):\n',
    '    fshift_filtered, mask = apply_filter(fshift_d, f_type, r1, r2)\n',
    '    img_reconstructed = get_ifft(fshift_filtered)\n',
    '    \n',
    '    plt.subplot(2, 4, i+1)\n',
    '    plt.imshow(mask, cmap="gray")\n',
    '    plt.title(f"{f_type.capitalize()} Mask")\n',
    '    plt.axis("off")\n',
    '    \n',
    '    plt.subplot(2, 4, i+5)\n',
    '    plt.imshow(img_reconstructed, cmap="gray")\n',
    '    plt.title(f"{f_type.capitalize()} Reconstruction")\n',
    '    plt.axis("off")\n',
    '\n',
    'plt.tight_layout()\n',
    'plt.show()'
   ]
  },
  {
   'cell_type': 'markdown',
   'metadata': {},
   'source': [
    '## 5. Evaluate CNN Performance on Preprocessed Images\n',
    'We will take the reconstructed 1-channel images, convert them back to 3-channel (RGB), and pass them to the CNN to see if confidence increases.'
   ]
  },
  {
   'cell_type': 'code',
   'execution_count': None,
   'metadata': {},
   'outputs': [],
   'source': [
    'def predict_on_img(img_array, title):\n',
    '    # Resize to 150x150 as expected by CNN\n',
    '    img_resized = cv2.resize(img_array, (150, 150))\n',
    '    \n',
    '    # If grayscale, convert to 3 channel\n',
    '    if len(img_resized.shape) == 2:\n',
    '        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)\n',
    '        \n',
    '    img_tensor = tf.expand_dims(img_resized, 0)\n',
    '    \n',
    '    # Use inference engine\n',
    '    res = diagnose_plant_disease(img_tensor, "memory", cnn_model, class_dirs)\n',
    '    print(f"{title:<20} -> {res[\\"Detected_Disease\\"]:<25} (Conf: {res[\\"Disease_Confidence\\"]:.2%})")\n',
    '\n',
    'print("--- DISEASED LEAF EVALUATION ---" )\n',
    'predict_on_img(img_disease_rgb, "Raw RGB")\n',
    'predict_on_img(img_disease_gray, "Raw Grayscale")\n',
    'for f_type in filters:\n',
    '    fshift_filtered, _ = apply_filter(fshift_d, f_type, 20, 60)\n',
    '    img_reconstructed = get_ifft(fshift_filtered)\n',
    '    predict_on_img(img_reconstructed, f_type.capitalize())\n',
    '\n',
    'print("\\n--- HEALTHY LEAF EVALUATION ---" )\n',
    'predict_on_img(img_healthy_rgb, "Raw RGB")\n',
    'predict_on_img(img_healthy_gray, "Raw Grayscale")\n',
    'for f_type in filters:\n',
    '    fshift_filtered, _ = apply_filter(fshift_h, f_type, 20, 60)\n',
    '    img_reconstructed = get_ifft(fshift_filtered)\n',
    '    predict_on_img(img_reconstructed, f_type.capitalize())'
   ]
  }
 ],
 'metadata': {
  'kernelspec': {
   'display_name': 'demeter_env',
   'language': 'python',
   'name': 'python3'
  },
  'language_info': {
   'codemirror_mode': {'name': 'ipython', 'version': 3},
   'file_extension': '.py',
   'mimetype': 'text/x-python',
   'name': 'python',
   'nbconvert_exporter': 'python',
   'pygments_lexer': 'ipython3',
   'version': '3.10.20'
  }
 },
 'nbformat': 4,
 'nbformat_minor': 4
}

with open('notebooks/fft_exploration.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('fft_exploration.ipynb generated successfully!')
