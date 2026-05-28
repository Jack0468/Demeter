import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- DYNAMIC PROJECT ROOT RESOLUTION ---
_current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _current_dir.parent.parent if _current_dir.parent.name == "src" else _current_dir.parent


def evaluate_model(model_path, test_ds, class_names, out_dir, model_name):
    """Generic evaluator that saves metrics to the out_dir under model_name prefix."""
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"[!] Model not found at {model_path}. Skipping evaluation.")
        return
        
    print(f"Loading model -> {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    y_true = []
    y_pred = []
    
    print(f"Evaluating {model_name}...")
    for batch_images, batch_labels in test_ds:
        preds = model.predict(batch_images, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        y_true.extend(batch_labels.numpy().tolist())
        y_pred.extend(pred_labels.tolist())
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    labels_list = list(range(len(class_names)))
    
    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels_list, average=None, zero_division=0)
    class_report = classification_report(y_true, y_pred, labels=labels_list, target_names=class_names, zero_division=0)
    
    overall = {'model': model_name, 'accuracy': acc, 'num_samples': int(len(y_true))}
    pd.DataFrame([overall]).to_csv(os.path.join(out_dir, f'{model_name}_overall_metrics.csv'), index=False)
    
    per_class = pd.DataFrame({
        'class': class_names,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support
    })
    per_class.to_csv(os.path.join(out_dir, f'{model_name}_per_class_metrics.csv'), index=False)
    
    cm = confusion_matrix(y_true, y_pred, labels=labels_list)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()
    
    with open(os.path.join(out_dir, f'{model_name}_classification_report.txt'), 'w') as fh:
        fh.write(class_report)
        
    print(f"[+] {model_name} evaluation complete. Acc: {acc:.4f}. Outputs in {out_dir}")


def evaluate_species_pipeline(plantvillage_dir, out_dir=None, img_height=150, img_width=150, batch_size=32):
    if out_dir is None:
        out_dir = str(PROJECT_ROOT / "evaluation_outputs/species_cnns")
    os.makedirs(out_dir, exist_ok=True)
    
    class_dirs = [d for d in os.listdir(plantvillage_dir) if os.path.isdir(os.path.join(plantvillage_dir, d))]
    species_names = sorted(list(set([d.split('_')[0] for d in class_dirs])))
    
    print(f"Found {len(species_names)} species to evaluate: {species_names}")
    
    def process_path(file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_height, img_width])
        return img, label
    
    def create_dataset(paths, labels):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    # 1. EVALUATE PRIMARY SPECIES IDENTIFIER
    species_to_idx = {s: i for i, s in enumerate(species_names)}
    image_paths_primary = []
    labels_primary = []
    
    for class_dir_name in class_dirs:
        class_dir = os.path.join(plantvillage_dir, class_dir_name)
        species = class_dir_name.split('_')[0]
        species_idx = species_to_idx[species]
        
        for image_file in os.listdir(class_dir):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths_primary.append(os.path.join(class_dir, image_file))
                labels_primary.append(species_idx)
                
    primary_ds = create_dataset(image_paths_primary, labels_primary)
    primary_model_path = str(PROJECT_ROOT / "models/demeter_cnn_plantvillage_species_identifier.keras")
    evaluate_model(primary_model_path, primary_ds, species_names, out_dir, "primary_species_identifier")
    
    # 2. EVALUATE EACH SPECIES-SPECIFIC MODEL
    for species_name in species_names:
        species_class_dirs = sorted([d for d in class_dirs if d.startswith(species_name)])
        
        image_paths_species = []
        labels_species = []
        
        for class_idx, class_dir_name in enumerate(species_class_dirs):
            class_dir = os.path.join(plantvillage_dir, class_dir_name)
            for image_file in os.listdir(class_dir):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths_species.append(os.path.join(class_dir, image_file))
                    labels_species.append(class_idx)
                    
        species_ds = create_dataset(image_paths_species, labels_species)
        species_model_path = str(PROJECT_ROOT / f"models/demeter_cnn_plantvillage_{species_name.lower()}.keras")
        evaluate_model(species_model_path, species_ds, species_class_dirs, out_dir, f"species_specific_{species_name.lower()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Species-Specific CNNs')
    parser.add_argument('--plantvillage_dir', type=str, default=str(PROJECT_ROOT / 'data/raw/vision/PlantVillage'), help='Path to PlantVillage dir')
    parser.add_argument('--out_dir', type=str, default=str(PROJECT_ROOT / 'evaluation_outputs/species_cnns'), help='Output folder')
    args = parser.parse_args()
    
    evaluate_species_pipeline(args.plantvillage_dir, args.out_dir)
