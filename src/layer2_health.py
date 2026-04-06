import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# ==========================================
# 1. RGB DISEASE MODEL (PlantVillage)
# ==========================================
def train_rgb_disease_model(train_dir, val_dir, save_path, epochs=10):
    """
    Trains a model to detect visual diseases (e.g., Early Blight, Leaf Mold)
    using standard RGB camera images.
    """
    print(f"Initializing RGB Disease Pipeline from: {train_dir}")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, image_size=(224, 224), batch_size=32)
    val_ds = tf.keras.utils.image_dataset_from_directory(val_dir, image_size=(224, 224), batch_size=32)
    
    num_classes = len(train_ds.class_names)
    
    # Using Transfer Learning for high accuracy on visual textures
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False 

    model = models.Sequential([
        layers.Rescaling(1./127.5, offset=-1, input_shape=(224, 224, 3)),
        layers.RandomRotation(0.2),
        layers.RandomFlip("horizontal"),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"RGB Disease Model saved to {save_path}\n")
    return train_ds.class_names

# ==========================================
# 2. THERMAL STRESS MODEL (AI4EOSC)
# ==========================================
def train_thermal_stress_model(train_dir, val_dir, save_path, epochs=15):
    """
    Trains a custom CNN to detect systemic stress via thermal imaging.
    Thermal images are often lower resolution and rely on heat-gradients 
    rather than fine textures, so a custom, lighter CNN is highly effective here.
    """
    print(f"Initializing Thermal Stress Pipeline from: {train_dir}")
    
    # We can use a smaller image size for thermal (e.g., 128x128) to save compute
    train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, image_size=(128, 128), batch_size=32, color_mode="rgb")
    val_ds = tf.keras.utils.image_dataset_from_directory(val_dir, image_size=(128, 128), batch_size=32, color_mode="rgb")
    
    num_classes = len(train_ds.class_names) # Usually e.g., 'Stressed', 'Not_Stressed'

    # Custom lightweight CNN architecture tailored for gradient blobs
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(128, 128, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Thermal Stress Model saved to {save_path}\n")
    return train_ds.class_names

# ==========================================
# 3. THE DIAGNOSTIC ASSESSOR (Inference Engine)
# ==========================================
class HealthAssessor:
    """
    This class loads both trained Layer 2 models. It is designed to be lightweight
    so that when you deploy Demeter, this logic acts as the bridge between your 
    cameras and your Layer 3 Environment Optimizer.
    """
    def __init__(self, rgb_model_path, thermal_model_path, rgb_classes, thermal_classes):
        print("Loading Layer 2 Diagnostic Models...")
        self.rgb_model = tf.keras.models.load_model(rgb_model_path)
        self.thermal_model = tf.keras.models.load_model(thermal_model_path)
        self.rgb_classes = rgb_classes
        self.thermal_classes = thermal_classes

    def assess_plant(self, rgb_image_path, thermal_image_path=None):
        """Processes live images and returns a unified health report."""
        import numpy as np

        # 1. Process RGB for Disease
        rgb_img = tf.keras.preprocessing.image.load_img(rgb_image_path, target_size=(224, 224))
        rgb_array = tf.expand_dims(tf.keras.preprocessing.image.img_to_array(rgb_img), 0)
        rgb_pred = self.rgb_model.predict(rgb_array, verbose=0)
        disease_diagnosis = self.rgb_classes[np.argmax(rgb_pred)]
        disease_confidence = np.max(rgb_pred)

        # 2. Process Thermal for Stress (if a thermal camera is attached to your hardware)
        stress_diagnosis = "Unknown (No Thermal Data)"
        stress_confidence = 0.0
        
        if thermal_image_path and os.path.exists(thermal_image_path):
            therm_img = tf.keras.preprocessing.image.load_img(thermal_image_path, target_size=(128, 128))
            therm_array = tf.expand_dims(tf.keras.preprocessing.image.img_to_array(therm_img), 0)
            therm_pred = self.thermal_model.predict(therm_array, verbose=0)
            stress_diagnosis = self.thermal_classes[np.argmax(therm_pred)]
            stress_confidence = np.max(therm_pred)

        # 3. The Unified Output (This gets passed to Layer 3)
        return {
            "disease_state": disease_diagnosis,
            "disease_confidence": float(disease_confidence),
            "thermal_stress_state": stress_diagnosis,
            "stress_confidence": float(stress_confidence),
            "requires_immediate_action": (disease_diagnosis != "Healthy" or stress_diagnosis == "Stressed")
        }