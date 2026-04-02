import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib  # Required for saving Scikit-Learn models
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==========================================
# 1. THE DEDICATED AUGMENTER
# ==========================================
def get_demeter_augmenter(img_height, img_width):
    """
    Handles rotation, flipping, and scaling in a single pipeline.
    MobileNetV2 specifically requires scaling between [-1, 1].
    """
    return models.Sequential([
        # Rescaling: Maps [0, 255] to [-1, 1]
        layers.Rescaling(1./127.5, offset=-1, input_shape=(img_height, img_width, 3)),
        
        # Augmentation: Rotation and Horizontal Flipping
        layers.RandomRotation(0.2),
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(0.1),
    ], name="demeter_augmenter_pipeline")

# ==========================================
# 2. CNN TRAINING PIPELINE
# ==========================================
def train_and_save_cnn(dataset_dir, save_path, img_height=150, img_width=150, epochs=5):
    """Trains the Transfer Learning CNN and saves it to disk."""
    print("Initializing CNN Training Pipeline...")
    
    # Load Data from directory structure
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir, validation_split=0.2, subset="training", seed=123,
        image_size=(img_height, img_width), batch_size=32)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir, validation_split=0.2, subset="validation", seed=123,
        image_size=(img_height, img_width), batch_size=32)
    
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # Build Architecture
    base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
    base_model.trainable = False 

    # Injecting the dedicated augmenter
    augmenter = get_demeter_augmenter(img_height, img_width)

    model = models.Sequential([
        augmenter,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    print(f"Training CNN for {epochs} epochs...")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    
    # Save Model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"CNN successfully saved to {save_path}")
    
    return class_names

# ==========================================
# 3. RANDOM FOREST TRAINING PIPELINE
# ==========================================
def train_and_save_rf(csv_dataset_path, save_path):
    """Trains a Random Forest on tabular sensor data and saves it."""
    print("Initializing Random Forest Training Pipeline...")
    
    # Load Tabular Data
    df = pd.read_csv(csv_dataset_path)
    
    # Feature selection based on Kaggle dataset schema
    X = df[['Species_Code', 'Temp', 'Moisture', 'Light']] 
    y = df[['Needs_Water', 'Needs_Fertilizer', 'Needs_Light']] 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and Save
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(rf_model, save_path)
    print(f"Random Forest successfully saved to {save_path}")
