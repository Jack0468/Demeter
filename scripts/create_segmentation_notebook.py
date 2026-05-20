import json
import os
from pathlib import Path

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# OpenCV Segmentation, Filter & Boundary Mitigation Exploration\n",
    "Testing advanced signal processing techniques (Mask Tapering and OpenCV Inpainting) to eliminate spectral leakage (edge artifacts) at the leaf boundary."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import sys\n",
    "import cv2\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from pathlib import Path\n",
    "\n",
    "# Setup paths\n",
    "PROJECT_ROOT = Path(os.getcwd()).parent\n",
    "if str(PROJECT_ROOT) not in sys.path:\n",
    "    sys.path.insert(0, str(PROJECT_ROOT))\n",
    "    \n",
    "DATA_DIR = PROJECT_ROOT / 'data/raw/vision/PlantVillage'\n",
    "class_dirs = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR / d)])\n",
    "print(f\"Found {len(class_dirs)} classes.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def segment_leaf(img_bgr):\n",
    "    \"\"\"Segments the leaf using Otsu thresholding.\"\"\"\n",
    "    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)\n",
    "    blur = cv2.GaussianBlur(gray, (5, 5), 0)\n",
    "    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)\n",
    "    \n",
    "    kernel = np.ones((5,5), np.uint8)\n",
    "    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)\n",
    "    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)\n",
    "    \n",
    "    segmented = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)\n",
    "    return segmented, mask\n",
    "\n",
    "def apply_tapered_mask(img_bgr, mask):\n",
    "    \"\"\"Applies a smoothly tapered Gaussian mask to eliminate sharp edges.\"\"\"\n",
    "    # Convert mask to float and blur it extensively\n",
    "    mask_float = mask.astype(float) / 255.0\n",
    "    tapered_mask = cv2.GaussianBlur(mask_float, (15, 15), 0)\n",
    "    \n",
    "    # Apply the tapered mask to the image\n",
    "    tapered_img = (img_bgr.astype(float) * tapered_mask[:, :, np.newaxis]).astype(np.uint8)\n",
    "    return tapered_img, tapered_mask\n",
    "\n",
    "def apply_inpainting(img_bgr, mask):\n",
    "    \"\"\"Inpaints the background with the boundary pixels of the leaf.\"\"\"\n",
    "    # The mask contains 255 for leaf, 0 for background.\n",
    "    # cv2.inpaint needs a mask where the region to be inpainted (background) is 255.\n",
    "    inpaint_mask = cv2.bitwise_not(mask)\n",
    "    \n",
    "    # Perform inpainting to propagate leaf pixels into the background\n",
    "    inpainted = cv2.inpaint(img_bgr, inpaint_mask, 5, cv2.INPAINT_TELEA)\n",
    "    return inpainted\n",
    "\n",
    "def get_fft_magnitude(img_gray):\n",
    "    \"\"\"Computes the 2D FFT magnitude spectrum.\"\"\"\n",
    "    f = np.fft.fft2(img_gray)\n",
    "    fshift = np.fft.fftshift(f)\n",
    "    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)\n",
    "    return magnitude_spectrum"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Pick a test image\n",
    "sample_dir = DATA_DIR / 'Tomato_Early_blight'\n",
    "sample_img = os.listdir(sample_dir)[0]\n",
    "img_path = str(sample_dir / sample_img)\n",
    "\n",
    "img_bgr = cv2.imread(img_path)\n",
    "img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)\n",
    "\n",
    "# 1. Binary Segmentation\n",
    "segmented_bin, mask = segment_leaf(img_bgr)\n",
    "segmented_bin_rgb = cv2.cvtColor(segmented_bin, cv2.COLOR_BGR2RGB)\n",
    "\n",
    "# 2. Tapered Masking\n",
    "tapered_img, tapered_mask = apply_tapered_mask(img_bgr, mask)\n",
    "tapered_rgb = cv2.cvtColor(tapered_img, cv2.COLOR_BGR2RGB)\n",
    "\n",
    "# 3. Inpainted Padding\n",
    "inpainted_img = apply_inpainting(img_bgr, mask)\n",
    "inpainted_rgb = cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB)\n",
    "\n",
    "# FFT Computations\n",
    "gray_raw = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)\n",
    "gray_bin = cv2.cvtColor(segmented_bin, cv2.COLOR_BGR2GRAY)\n",
    "gray_tap = cv2.cvtColor(tapered_img, cv2.COLOR_BGR2GRAY)\n",
    "gray_inp = cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2GRAY)\n",
    "\n",
    "fft_raw = get_fft_magnitude(gray_raw)\n",
    "fft_bin = get_fft_magnitude(gray_bin)\n",
    "fft_tap = get_fft_magnitude(gray_tap)\n",
    "fft_inp = get_fft_magnitude(gray_inp)\n",
    "\n",
    "# Plot comparisons\n",
    "fig, axes = plt.subplots(4, 2, figsize=(12, 18))\n",
    "\n",
    "# Row 1: Raw\n",
    "axes[0, 0].imshow(img_rgb)\n",
    "axes[0, 0].set_title(\"Original Raw Image\")\n",
    "axes[0, 1].imshow(fft_raw, cmap='inferno')\n",
    "axes[0, 1].set_title(\"Raw FFT Magnitude\")\n",
    "\n",
    "# Row 2: Binary\n",
    "axes[1, 0].imshow(segmented_bin_rgb)\n",
    "axes[1, 0].set_title(\"Binary Masked Leaf\")\n",
    "axes[1, 1].imshow(fft_bin, cmap='inferno')\n",
    "axes[1, 1].set_title(\"Binary FFT Magnitude (Notice starburst boundary spikes!)\")\n",
    "\n",
    "# Row 3: Tapered\n",
    "axes[2, 0].imshow(tapered_rgb)\n",
    "axes[2, 0].set_title(\"Tapered (Smooth Faded) Leaf\")\n",
    "axes[2, 1].imshow(fft_tap, cmap='inferno')\n",
    "axes[2, 1].set_title(\"Tapered FFT Magnitude (Boundary spikes are softened)\")\n",
    "\n",
    "# Row 4: Inpainted\n",
    "axes[3, 0].imshow(inpainted_rgb)\n",
    "axes[3, 0].set_title(\"Inpainted (Seamless Background) Leaf\")\n",
    "axes[3, 1].imshow(fft_inp, cmap='inferno')\n",
    "axes[3, 1].set_title(\"Inpainted FFT Magnitude (Zero edge artifacts!)\")\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

PROJECT_ROOT = Path(os.getcwd())
out_path = PROJECT_ROOT / 'notebooks' / 'segmentation_filters_exploration.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
print("Notebook generated successfully with advanced boundary mitigations.")
