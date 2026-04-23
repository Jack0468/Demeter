import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd # Import pandas for DataFrame handling in RF evaluation

def evaluate_cnn_model(model, val_ds, class_names):
    """
    Evaluates the trained CNN model and prints classification metrics.
    """
    print("\nEvaluating CNN Model...")
    val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    y_true = []
    y_pred = []
    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1-Score: {f1:.4f}")
    
    return val_accuracy, precision, recall, f1

def evaluate_rf_model(rf_model, X_test, y_test):
    """
    Evaluates the trained Random Forest model and prints classification metrics.
    """
    print("\nEvaluating Random Forest Model...")
    y_pred_rf = rf_model.predict(X_test)

    target_names = y_test.columns # Assuming y_test is a pandas DataFrame
    for i, target in enumerate(target_names):
        print(f"\nMetrics for {target}:")
        print(f"  Accuracy: {accuracy_score(y_test.iloc[:, i], y_pred_rf[:, i]):.4f}")
        print(f"  Precision: {precision_score(y_test.iloc[:, i], y_pred_rf[:, i], zero_division=0):.4f}")
        print(f"  Recall: {recall_score(y_test.iloc[:, i], y_pred_rf[:, i], zero_division=0):.4f}")
        print(f"  F1-Score: {f1_score(y_test.iloc[:, i], y_pred_rf[:, i], zero_division=0):.4f}")
