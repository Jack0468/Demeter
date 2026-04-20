import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib  # Required for saving Scikit-Learn models
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.ensemble import RandomForestRegressor # Changed from Classifier
from sklearn.model_selection import train_test_split
from model_evaluation import evaluate_cnn_model, evaluate_rf_model

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
def train_and_save_cnn(df, base_dir, save_path, img_height=150, img_width=150, epochs=5):
    """Trains the Transfer Learning CNN using a DataFrame and saves it."""
    print("Initializing CNN Training Pipeline...")
    
    # 1. Isolate Visible Spectrum Images and construct file paths
    df_vis = df[df['spectrum'].str.contains('Visible', na=False, case=False)].copy()
    df_vis['filepath'] = df_vis.apply(
        lambda row: os.path.join(base_dir, f"snapshot{row['parent snapshot id']}", f"{row['name']}.jpg"),
        axis=1
    )
    
    # 2. Filter out paths that don't exist AND ignore 0-byte (empty) files
    df_vis = df_vis[df_vis['filepath'].apply(lambda p: os.path.exists(p) and os.path.getsize(p) > 0)]
    
    if df_vis.empty:
        print("[!] ERROR: No valid image paths found. Check WSL mount mapping.")
        return []

    # 3. Create a binary classification target (e.g., Water Stressed vs Well Watered)
    # Using the median water amount as a simple threshold for the Demeter prototype
    median_water = df_vis['water amount'].median()
    df_vis['label_int'] = (df_vis['water amount'] >= median_water).astype(int)
    class_names = ['Water_Stressed', 'Well_Watered']
    num_classes = len(class_names)

    # 4. Split data
    train_df, val_df = train_test_split(df_vis, test_size=0.2, random_state=123)
    print(f"Training on {len(train_df)} images, validating on {len(val_df)} images.")

    # 5. Build tf.data pipeline
    def process_path(file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_height, img_width])
        return img, label

    def create_dataset(dataframe):
        paths = dataframe['filepath'].values
        labels = dataframe['label_int'].values
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = create_dataset(train_df)
    val_ds = create_dataset(val_df)

    # 6. Build Architecture
    base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
    base_model.trainable = False 

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
    
    # Evaluate CNN (Assuming evaluate_cnn_model accepts tf.data.Dataset)
    evaluate_cnn_model(model, val_ds, class_names)
    
    # Save Model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"CNN successfully saved to {save_path}")
    
    return class_names


# ==========================================
# 3. RANDOM FOREST TRAINING PIPELINE
# ==========================================
def train_and_save_rf(df, save_path):
    """Trains a Random Forest Regressor on tabular sensor data to predict growth trajectory."""
    print("Initializing Random Forest Regressor Training Pipeline...")
    
    # Drop rows with NaN values in our target features
    df_clean = df.dropna(subset=['weight before', 'water amount', 'weight after']).copy()
    
    # Features: Current state (Weight) + Intervention (Water Applied)
    X = df_clean[['weight before', 'water amount']] 
    
    # Target: Continuous future growth metric (The resulting weight of the plant/soil)
    y = df_clean['weight after']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train as a Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate using RMSE (As promised in the proposal)
    predictions = rf_model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    print(f"Random Forest Regressor RMSE: {rmse:.2f}g")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(rf_model, save_path)
    print(f"Random Forest Regressor successfully saved to {save_path}")