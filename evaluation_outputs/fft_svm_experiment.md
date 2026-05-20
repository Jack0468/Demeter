# FFT Support Vector Machine (SVM) Results

We successfully created the training pipeline and ran an SVM exclusively on the frequency domains of the PlantVillage dataset.

## What We Built
1. **Training Script (`src/training/train_fft_svm.py`)**: A pipeline that iterates through the dataset, extracting a subset of images (50 train, 20 test per class) to evaluate frequency features.
2. **Feature Extraction**: Loaded each image, converted it to Grayscale, resized it to 64x64, calculated the 2D FFT, and flattened the magnitude spectrum into an array of 4096 features.
3. **Dimensionality Reduction**: Used `StandardScaler` to normalize frequencies, followed by `PCA` (Principal Component Analysis) to reduce the 4096 features down to the 100 most important frequency components.
4. **Classification**: Trained a Support Vector Classifier (`SVC` with RBF kernel and balanced class weights) on the 100 components.
5. **Model Storage**: Saved the scaler, PCA, and SVM models to `models/experimentation/` for future reference.

## Experimental Results

The model achieved an overall accuracy of **36.39%** on the test set.

```text
--- RESULTS ---
Accuracy: 36.39%

Classification Report:
                                             precision    recall  f1-score   support

              Pepper__bell___Bacterial_spot       0.40      0.53      0.45        19
                     Pepper__bell___healthy       0.71      0.25      0.37        20
                      Potato___Early_blight       0.24      0.45      0.31        20
                       Potato___Late_blight       0.00      0.00      0.00        20
                           Potato___healthy       0.42      0.55      0.48        20
                      Tomato_Bacterial_spot       0.29      0.84      0.43        19
                        Tomato_Early_blight       0.33      0.20      0.25        20
                         Tomato_Late_blight       0.00      0.00      0.00        20
                           Tomato_Leaf_Mold       0.60      0.16      0.25        19
                  Tomato_Septoria_leaf_spot       0.00      0.00      0.00        19
Tomato_Spider_mites_Two_spotted_spider_mite       0.27      0.30      0.29        20
                        Tomato__Target_Spot       0.26      0.42      0.32        19
      Tomato__Tomato_YellowLeaf__Curl_Virus       1.00      0.40      0.57        20
                Tomato__Tomato_mosaic_virus       0.31      0.74      0.44        19
                             Tomato_healthy       0.93      0.65      0.76        20

                                  macro avg       0.38      0.37      0.33       294
```

> [!NOTE]
> **Performance Analysis:** 
> While 36.4% might seem low compared to our CNN's 95%+ accuracy, we have to consider that random guessing across 15 classes would yield ~6.6%. The SVM performed substantially better than random chance using **only pure textural frequency** without knowing what shape or color the leaf was. 
> 
> However, some classes (like `Potato___Late_blight` and `Tomato_Septoria_leaf_spot`) failed entirely, indicating that their visual diagnosis relies heavily on spatial structures (where the spot is) rather than pure frequency (what texture the spot is).

### Engineering Conclusion
Frequency-domain classification provides an interesting signal, but it is not robust enough to act as a primary diagnostic tool. The spatial patterns (learned by the CNN convolutions) are vastly superior for plant pathology than frequency amplitudes. We will keep these models in `models/experimentation/`, but the CNN will remain the core vision stream.
