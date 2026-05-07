import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_cnn(cnn_path, test_dir, img_height=150, img_width=150, batch_size=32, out_dir="evaluation_outputs/cnn"):
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(cnn_path):
        raise FileNotFoundError(f"CNN model not found at {cnn_path}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found at {test_dir}")

    print("Loading CNN model...")
    model = tf.keras.models.load_model(cnn_path)

    print("Loading test dataset (no shuffle)...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='int',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False
    )

    class_names = test_ds.class_names

    y_true = []
    y_pred = []

    print("Running predictions...")
    for batch_images, batch_labels in test_ds:
        preds = model.predict(batch_images, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        y_true.extend(batch_labels.numpy().tolist())
        y_pred.extend(pred_labels.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Metrics
    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    class_report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    # Save overall metrics
    overall = {'model': os.path.basename(cnn_path), 'dataset': os.path.basename(test_dir), 'accuracy': acc, 'num_samples': int(len(y_true))}
    pd.DataFrame([overall]).to_csv(os.path.join(out_dir, 'cnn_overall_metrics.csv'), index=False)

    # Per-class metrics
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

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('CNN Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cnn_confusion_matrix.png'))
    plt.close()

    # Per-class accuracy
    with np.errstate(divide='ignore', invalid='ignore'):
        class_acc = np.diag(cm) / cm.sum(axis=1).astype(float)
        class_acc = np.nan_to_num(class_acc)
    acc_df = pd.DataFrame({'class': class_names, 'per_class_accuracy': class_acc})
    acc_df.to_csv(os.path.join(out_dir, 'cnn_per_class_accuracy.csv'), index=False)

    # Plot per-class accuracy
    plt.figure(figsize=(10, 5))
    sns.barplot(x='class', y='per_class_accuracy', data=acc_df)
    plt.ylim(0, 1)
    plt.title('Per-Class Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cnn_per_class_accuracy.png'))
    plt.close()

    # Save classification report text
    with open(os.path.join(out_dir, 'cnn_classification_report.txt'), 'w') as fh:
        fh.write(class_report)

    print("CNN evaluation complete. Outputs written to:", out_dir)


def evaluate_cnn_simple(cnn_path, test_dir, img_size=(150, 150), out_dir='evaluation_outputs/cnn_simple'):
    """Very small, easy-to-read evaluator (single-image predict loop)."""
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(cnn_path):
        raise FileNotFoundError(f"CNN model not found at {cnn_path}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found at {test_dir}")

    print('Loading model ->', cnn_path)
    model = tf.keras.models.load_model(cnn_path)

    classes = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    y_true = []
    y_pred = []

    for i, cls in enumerate(classes):
        cls_path = os.path.join(test_dir, cls)
        for f in os.listdir(cls_path):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            p = os.path.join(cls_path, f)
            img = tf.keras.utils.load_img(p, target_size=img_size)
            arr = tf.keras.utils.img_to_array(img)
            arr = np.expand_dims(arr, 0)
            preds = model.predict(arr, verbose=0)
            pred_label = int(np.argmax(preds, axis=1)[0])
            y_true.append(i)
            y_pred.append(pred_label)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
    pd.DataFrame([{'model': os.path.basename(cnn_path), 'dataset': os.path.basename(test_dir), 'accuracy': acc, 'num_samples': int(len(y_true))}]).to_csv(os.path.join(out_dir, 'cnn_overall_metrics.csv'), index=False)
    print('Simple CNN done — saved cnn_overall_metrics.csv to', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CNN and save metrics/plots')
    parser.add_argument('--cnn', type=str, default='models/demeter_cnn.keras', help='Path to saved Keras CNN')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory with test images, subfolders per class')
    parser.add_argument('--out_dir', type=str, default='evaluation_outputs/cnn', help='Output folder')
    parser.add_argument('--mode', choices=['full', 'simple'], default='full', help='Evaluation mode: full (detailed) or simple (single-image loop)')
    args = parser.parse_args()
    if args.mode == 'full':
        evaluate_cnn(args.cnn, args.test_dir, out_dir=args.out_dir)
    else:
        # simple mode writes to a separate folder by default
        evaluate_cnn_simple(args.cnn, args.test_dir, out_dir=args.out_dir)
