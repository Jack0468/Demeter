import joblib
import os
import numpy as np

path = "c:/Users/Admin/Documents/Windows_codespace/DEMETER/Demeter/models/health_clusters.joblib"
model = joblib.load(path)
print(f"Model loaded: {model}")
print(f"Features expected: {model.named_steps['scaler'].n_features_in_}")
print(f"Feature names (if available): {getattr(model.named_steps['scaler'], 'feature_names_in_', 'Not available')}")
