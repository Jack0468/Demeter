import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_cnn(cnn_path, test_dir, img_height=150, img_width=150, batch_size=32, out_dir="evaluation_outputs"):
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(cnn_path):
        raise FileNotFoundError(f"CNN model not found at {cnn_path}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test image directory not found at {test_dir}")

    print("Loading CNN model...")
    model = tf.keras.models.load_model(cnn_path)

    print("Loading test dataset from directory (no shuffling)...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='int',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False
    )

    class_names = test_ds.class_names
    print(f"Found classes: {class_names}")

    # Gather true labels and predictions
    y_true = []
    y_pred = []

    print("Running predictions on test dataset...")
    for batch_images, batch_labels in test_ds:
        preds = model.predict(batch_images, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        y_true.extend(batch_labels.numpy().tolist())
        y_pred.extend(pred_labels.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    class_report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    print("Saving classification report and metrics...")
    metrics_summary = {
        'accuracy': acc,
        'num_samples': int(len(y_true))
    }
    metrics_df = pd.DataFrame([metrics_summary])
    metrics_df.to_csv(os.path.join(out_dir, 'cnn_overall_metrics.csv'), index=False)

    # Per-class metrics table
    per_class = pd.DataFrame({
        'class': class_names,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support
    })
    per_class.to_csv(os.path.join(out_dir, 'cnn_per_class_metrics.csv'), index=False)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(out_dir, 'cnn_confusion_matrix.csv'))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('CNN Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cnn_confusion_matrix.png'))
    plt.close()

    # Per-class accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1).astype(float)
    acc_df = pd.DataFrame({'class': class_names, 'per_class_accuracy': class_acc})
    acc_df.to_csv(os.path.join(out_dir, 'cnn_per_class_accuracy.csv'), index=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(x='class', y='per_class_accuracy', data=acc_df)
    plt.ylim(0, 1)
    plt.title('Per-Class Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cnn_per_class_accuracy.png'))
    plt.close()

    # Save textual classification report
    with open(os.path.join(out_dir, 'cnn_classification_report.txt'), 'w') as fh:
        fh.write(class_report)

    print("CNN evaluation complete. Outputs written to:", out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Demeter CNN model on a test image directory')
    parser.add_argument('--cnn', type=str, default='models/demeter_cnn.keras', help='Path to saved Keras CNN')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory with test images organized by class')
    parser.add_argument('--out_dir', type=str, default='evaluation_outputs/cnn', help='Directory to write outputs')
    args = parser.parse_args()

    evaluate_cnn(args.cnn, args.test_dir, out_dir=args.out_dir)
