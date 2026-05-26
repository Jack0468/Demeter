# Jack's Presentation Notes: Methods & MVP

*These are your high-level dot points to reference during your 2 min 15 sec slot.*

## 1. Augmentation & Feature Extraction
**Visual Asset to show:** Diagram of Image -> Otsu Mask -> FFT Spectrum -> HSV Histogram.

*   **Tabular Integration:** We mapped raw images to Danforth telemetry metadata using `derive_danforth_csv.py` to create a unified dataframe.
*   **Spatial Augmentation:** To make the model robust against real-world domain shifts, we expanded the dataset using side-view rotations (0°, 90°, 180°, 270°) and added noise.
*   **Biological Signal Pipeline:**
    *   **Segmentation:** Isolated the leaf from background clutter using Otsu's thresholding.
    *   **Texture (FFT):** Extracted spatial texture via 2D Fast Fourier Transform. Crucially, we applied *Gaussian-fade tapering* to eliminate artificial boundary noise caused by masking.
    *   **Color (HSV):** Appended a 64-bin HSV color histogram, because FFT alone cannot detect pigmentation-based diseases (like yellowing).

## 2. ML Methods & Benchmarking
**Visual Asset to show:** The `latency_comparison.png` bar chart.

*   **CNN (MobileNetV2):** Our baseline deep learning model for heavy, image-based disease classification.
*   **SVM (FFT + HSV):** Our lightweight, vector-based alternative. 
    *   *Direct Benchmarking:* We benchmarked the SVM directly against the CNN on the same dataset. It achieved identical accuracy but required vastly less computational power, making it viable for cheap edge devices.
*   **Random Forest:** 
    *   Trained separately on the tabular/environmental dataset to serve a different task: predicting biomass regression.
    *   *Note:* We briefly explored using RF for direct classification (via thresholding regression targets), but determined it was less useful than our primary hybrid approach.

## 3. Evaluation Approach
**Visual Asset to show:** A small table highlighting the metrics.

*   **Macro F1-Score (84.28%):** We optimized the SVM for F1-Score instead of pure accuracy. Healthy leaves heavily outnumber diseased leaves in our dataset, so F1 ensures our model remains highly sensitive to rare, crop-destroying diseases.
*   **RMSE (0.0846):** We evaluated the Random Forest using Root Mean Square Error. This provides a physical metric (e.g., grams of biomass error) which is strictly necessary to calculate exact physical water and fertilizer adjustments.
*   **$R^2$ (0.9978):** Used to confirm the RF model accurately captured overarching growth trajectories.

---

## 4. MVP / Demonstration (Video Walkthrough)
**Visual Asset to show:** The 30-second Dashboard Video + Mermaid Architecture Flow diagram.

*   **The Unified API:** Explain that our solution is a Flask API Server that bridges the accessibility gap by running the CNN, SVM, and RF concurrently.
*   **Video Summary:** 
    *   Point out how the system ingests live sensor data alongside the image upload.
    *   Highlight the deterministic "Status Engine" turning raw model probabilities into concrete, actionable recommendations for the user.
*   **Practical Recommendations / Next Steps:** Our primary recommendation to the client is to compile the SVM pipeline natively onto a low-power microcontroller (like a Raspberry Pi Zero) to create a cheap, offline greenhouse diagnostic node.

## JACKS CONCLUSION

Our website is an example of a product which could be used on the internet or consolidated into an offline application for mobile devices. This can also offer the oppurtunity for investors and developers to consider the pheasability implementing our design into their system
