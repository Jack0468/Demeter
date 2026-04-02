import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib # Used for saving Scikit-Learn models

def train_and_save_cnn(dataset_dir, save_path, img_height=150, img_width=150, epochs=5):
    """Trains the Transfer Learning CNN and saves it to disk."""
    print("Initializing CNN Training Pipeline...")
    
    # 1. Load Data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir, validation_split=0.2, subset="training", seed=123,
        image_size=(img_height, img_width), batch_size=32)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir, validation_split=0.2, subset="validation", seed=123,
        image_size=(img_height, img_width), batch_size=32)
    
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # 2. Build Model Architecture
    base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
    base_model.trainable = False 

    model = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.Rescaling(1./127.5, offset=-1),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    # 3. Train Model
    # Note: If running this within your WSL Conda environment without GPU passthrough, 
    # start with a low number of epochs (e.g., 5) to test functionality before doing a long training run.
    print(f"Training CNN for {epochs} epochs...")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    
    # 4. Save Model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"CNN successfully saved to {save_path}")
    
    return class_names

def train_and_save_rf(csv_dataset_path, save_path):
    """Trains a Random Forest on tabular sensor data and saves it."""
    print("Initializing Random Forest Training Pipeline...")
    
    # 1. Load Tabular Data
    df = pd.read_csv(csv_dataset_path)
    
    # Assuming columns: 'Species_Code', 'Temp', 'Moisture', 'Light', 'Needs_Water', 'Needs_Fertilizer', 'Needs_Light'
    # You will need to adjust these feature names to match the Kaggle tabular dataset exactly
    X = df[['Species_Code', 'Temp', 'Moisture', 'Light']] 
    
    # For a multi-output scenario (Yes/No for 3 different things), we train on multiple targets
    y = df[['Needs_Water', 'Needs_Fertilizer', 'Needs_Light']] 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Train Model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # 3. Save Model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(rf_model, save_path)
    print(f"Random Forest successfully saved to {save_path}")
