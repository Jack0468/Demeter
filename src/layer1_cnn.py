import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

def build_demeter_cnn(img_height, img_width, num_classes):
    """Builds the Transfer Learning architecture."""
    base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
    base_model.trainable = False 

    model = models.Sequential([
        layers.Rescaling(1./127.5, offset=-1, input_shape=(img_height, img_width, 3)),
        layers.RandomRotation(0.2),
        layers.RandomFlip("horizontal"),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_local_or_kaggle(train_dir, val_dir, save_path, epochs=5):
    """A flexible training loop that takes directory paths as inputs."""
    print(f"Training Layer 1 CNN from: {train_dir}")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, image_size=(224, 224), batch_size=32)
    val_ds = tf.keras.utils.image_dataset_from_directory(val_dir, image_size=(224, 224), batch_size=32)
    
    model = build_demeter_cnn(224, 224, len(train_ds.class_names))
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    return train_ds.class_names