import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib 

def get_demeter_augmenter(img_height, img_width):
    """
    DEDICATED AUGMENTER: 
    Handles rotation, flipping, and scaling in a single pipeline.
    """
    return models.Sequential([
        # 1. Rescaling: MobileNetV2 expects pixels in [-1, 1] range
        layers.Rescaling(1./127.5, offset=-1, input_shape=(img_height, img_width, 3)),
        
        # 2. Rotation: Randomly rotate images up to 20% (0.2 * 360 degrees)
        layers.RandomRotation(0.2),
        
        # 3. Flipping: Standard horizontal flipping for plant symmetry
        layers.RandomFlip("horizontal"),
        
        # 4. Zoom: Extra robustness for different camera distances
        layers.RandomZoom(0.1),
    ], name="demeter_augmenter_pipeline")

def train_and_save_cnn(dataset_dir, save_path, img_height=150, img_width=150, epochs=5):
    print("Initializing CNN Training Pipeline with Integrated Augmentation...")
    
    # 1. Load Data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir, validation_split=0.2, subset="training", seed=123,
        image_size=(img_height, img_width), batch_size=32)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir, validation_split=0.2, subset="validation", seed=123,
        image_size=(img_height, img_width), batch_size=32)
    
    class_names = train_ds.class_names
    num_classes = len(class_names)

    # 2. Build Model Architecture
    base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
    base_model.trainable = False 

    # INJECT THE NEW AUGMENTER HERE
    augmenter = get_demeter_augmenter(img_height, img_width)

    model = models.Sequential([
        augmenter,      # The dedicated preprocessing track
        base_model,     # The pre-trained brain
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    # 3. Train Model
    print(f"Training CNN for {epochs} epochs...")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    
    # 4. Save Model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"CNN successfully saved to {save_path}")
    
    return class_names

# (Keep your train_and_save_rf function as is below)
