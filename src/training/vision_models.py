import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split

# ==========================================
# 1. THE DEDICATED AUGMENTER
# ==========================================
def get_augmenter(img_height, img_width):
    """
    Handles rotation, flipping, and scaling in a single pipeline.
    MobileNetV2 specifically requires scaling between [-1, 1].
    """
    return models.Sequential([
        # Input layer
        layers.Input(shape=(img_height, img_width, 3)),
        
        # Rescaling: Maps [0, 255] to [-1, 1]
        layers.Rescaling(1./127.5, offset=-1),
        
        # Augmentation: Rotation and Horizontal Flipping
        layers.RandomRotation(0.2),
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(0.1),
    ], name="demeter_augmenter_pipeline")


# ==========================================
# 2. CNN TRAINING PIPELINE - PLANTVILLAGE DISEASE CLASSIFICATION
# ==========================================
def train_and_save_cnn_plantvillage(plantvillage_dir, save_path, img_height=224, img_width=224, epochs=5):
    """Trains a Transfer Learning CNN on PlantVillage dataset for disease classification."""
    print("Initializing CNN Training Pipeline (PlantVillage)...")
    
    # 1. Scan PlantVillage directory for disease classes
    class_dirs = [d for d in os.listdir(plantvillage_dir) 
                  if os.path.isdir(os.path.join(plantvillage_dir, d))]
    class_names = sorted(class_dirs)
    num_classes = len(class_names)
    
    if num_classes == 0:
        print(f"[!] ERROR: No class directories found in {plantvillage_dir}")
        return []
    
    print(f"Found {num_classes} disease classes: {class_names}")
    
    # 2. Build file paths and labels
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(plantvillage_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for image_file in image_files:
            image_path = os.path.join(class_dir, image_file)
            # Verify file exists and is not empty
            if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                image_paths.append(image_path)
                labels.append(class_idx)
    
    if len(image_paths) == 0:
        print(f"[!] ERROR: No valid images found in {plantvillage_dir}")
        return []
    
    print(f"Loaded {len(image_paths)} images for training")
    
    # 3. Split data
    image_paths_array = np.array(image_paths)
    labels_array = np.array(labels)
    
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        image_paths_array, labels_array, test_size=0.2, random_state=123, stratify=labels_array
    )
    
    try:
        from src.training.split_tracker import update_manifest
        update_manifest("demeter_cnn_plantvillage", X_train_paths, X_val_paths)
    except Exception as e:
        print(f"Failed to save manifest: {e}")
    
    print(f"Training on {len(X_train_paths)} images, validating on {len(X_val_paths)} images.")
    
    # 4. Build tf.data pipeline
    def process_path(file_path, label):
        img = tf.io.read_file(file_path)
        # Try to decode as JPEG, fallback to PNG if fails
        try:
            img = tf.image.decode_jpeg(img, channels=3)
        except:
            img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [img_height, img_width])
        return img, label
    
    def create_dataset(paths, labels):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
        return ds
    
    train_ds = create_dataset(X_train_paths, y_train)
    val_ds = create_dataset(X_val_paths, y_val)
    
    # 5. Build Architecture
    base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    
    augmenter = get_augmenter(img_height, img_width)
    
    model = models.Sequential([
        augmenter,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    print(f"Training CNN for {epochs} epochs...")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    
    # Evaluate CNN internally
    loss, accuracy = model.evaluate(val_ds, verbose=0)
    print(f"PlantVillage CNN - Val Loss: {loss:.4f}, Val Accuracy: {accuracy:.4f}")
    
    # Save Model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"CNN successfully saved to {save_path}")
    
    return class_names


# ==========================================
# 3. CNN TRAINING PIPELINE - SPECIES-SPECIFIC DISEASE CLASSIFICATION
# ==========================================
def train_plantvillage_species_specific_disease_model(plantvillage_dir, save_path, species_name, img_height=224, img_width=224, epochs=5):
    """Trains a specialized CNN on a specific species in the PlantVillage dataset."""
    print(f"Initializing CNN Training Pipeline (PlantVillage - {species_name})...")
    
    # 1. Scan PlantVillage directory for disease classes
    class_dirs = [d for d in os.listdir(plantvillage_dir) 
                  if os.path.isdir(os.path.join(plantvillage_dir, d))]
    
    # Filter by species_name
    class_dirs = [d for d in class_dirs if d.startswith(species_name)]
    class_names = sorted(class_dirs)
    num_classes = len(class_names)
    
    if num_classes == 0:
        print(f"[!] ERROR: No class directories found in {plantvillage_dir} for species {species_name}")
        return []
    
    print(f"Found {num_classes} disease classes for {species_name}: {class_names}")
    
    # 2. Build file paths and labels
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(plantvillage_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for image_file in image_files:
            image_path = os.path.join(class_dir, image_file)
            if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                image_paths.append(image_path)
                labels.append(class_idx)
    
    if len(image_paths) == 0:
        print(f"[!] ERROR: No valid images found for {species_name}")
        return []
        
    image_paths_array = np.array(image_paths)
    labels_array = np.array(labels)
    
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        image_paths_array, labels_array, test_size=0.2, random_state=123, stratify=labels_array
    )
    
    def process_path(file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_height, img_width])
        return img, label
    
    def create_dataset(paths, labels):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
        return ds
    
    train_ds = create_dataset(X_train_paths, y_train)
    val_ds = create_dataset(X_val_paths, y_val)
    
    base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    
    augmenter = get_augmenter(img_height, img_width)
    
    model = models.Sequential([
        augmenter,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    print(f"Training CNN for {epochs} epochs...")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"CNN successfully saved to {save_path}")
    
    return class_names


def train_plantvillage_species_identifier(plantvillage_dir, save_path, img_height=224, img_width=224, epochs=5):
    """Trains a primary CNN to identify the plant species in the PlantVillage dataset."""
    print("Initializing Primary Species Identifier CNN Training Pipeline...")
    
    class_dirs = [d for d in os.listdir(plantvillage_dir) 
                  if os.path.isdir(os.path.join(plantvillage_dir, d))]
    
    # Extract unique species (first part before underscore)
    species_set = set([d.split('_')[0] for d in class_dirs])
    species_names = sorted(list(species_set))
    num_species = len(species_names)
    
    if num_species == 0:
        print(f"[!] ERROR: No species found in {plantvillage_dir}")
        return []
        
    print(f"Found {num_species} distinct species: {species_names}")
    
    species_to_idx = {s: i for i, s in enumerate(species_names)}
    
    image_paths = []
    labels = []
    
    for class_dir_name in class_dirs:
        class_dir = os.path.join(plantvillage_dir, class_dir_name)
        species = class_dir_name.split('_')[0]
        species_idx = species_to_idx[species]
        
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for image_file in image_files:
            image_path = os.path.join(class_dir, image_file)
            if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                image_paths.append(image_path)
                labels.append(species_idx)
                
    if len(image_paths) == 0:
        print("[!] ERROR: No valid images found.")
        return []
        
    image_paths_array = np.array(image_paths)
    labels_array = np.array(labels)
    
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        image_paths_array, labels_array, test_size=0.2, random_state=123, stratify=labels_array
    )
    
    def process_path(file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_height, img_width])
        return img, label
    
    def create_dataset(paths, labels):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
        return ds
    
    train_ds = create_dataset(X_train_paths, y_train)
    val_ds = create_dataset(X_val_paths, y_val)
    
    base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    
    augmenter = get_augmenter(img_height, img_width)
    
    model = models.Sequential([
        augmenter,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_species, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    print(f"Training Primary Species Identifier CNN for {epochs} epochs...")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Primary Species Identifier CNN successfully saved to {save_path}")
    
    return species_names


# ==========================================
# 4. CNN TRAINING PIPELINE - TILLER COUNT REGRESSION
# ==========================================
def train_tiller_cnn_regressor(df, save_path, img_height=224, img_width=224, epochs=10):
    """Trains a CNN Regressor to predict continuous tiller count from images."""
    print("Initializing CNN Regressor Training Pipeline (Tiller Count)...")
    
    # 1. Clean data
    if 'tiller_count' not in df.columns:
        print("[!] ERROR: 'tiller_count' column not found in dataframe.")
        return False
        
    df_clean = df.dropna(subset=['filepath', 'tiller_count']).copy()
    df_clean = df_clean[df_clean['filepath'].apply(lambda p: os.path.exists(p) and os.path.getsize(p) > 0)]
    
    if df_clean.empty:
        print("[!] ERROR: No valid image paths found for tiller dataset.")
        return False
        
    print(f"Loaded {len(df_clean)} images for Tiller Regression training")
    
    # 2. Split data
    X_train, X_val, y_train, y_val = train_test_split(
        df_clean['filepath'].values, 
        df_clean['tiller_count'].values.astype(np.float32), 
        test_size=0.2, 
        random_state=123
    )
    print(f"Training on {len(X_train)} images, validating on {len(X_val)} images.")
    
    # 3. Build tf.data pipeline
    def process_path(file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=3)  # Tiller dataset consists of .png images
        img = tf.image.resize(img, [img_height, img_width])
        return img, label

    def create_dataset(paths, labels):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(16).prefetch(tf.data.AUTOTUNE)  # Smaller batch size for the small 58-image dataset
        return ds

    train_ds = create_dataset(X_train, y_train)
    val_ds = create_dataset(X_val, y_val)
    
    # 4. Build Architecture (Regressor instead of Classifier)
    base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    
    augmenter = get_augmenter(img_height, img_width)
    
    model = models.Sequential([
        augmenter,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear')  # Output layer uses linear activation for continuous prediction
    ])
    
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae'])
    
    print(f"Training CNN Regressor for {epochs} epochs...")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    
    # Evaluate using MSE and MAE on validation set
    loss, mae = model.evaluate(val_ds, verbose=0)
    print(f"Tiller CNN Regressor - Val MSE: {loss:.4f}, Val MAE: {mae:.4f}")
    
    # Save Model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"CNN Regressor successfully saved to {save_path}")
    
    return True


# ==========================================
# 5. CNN TRAINING PIPELINE - CUSTOM BASELINE (WITH PADDING)
# ==========================================
def train_custom_baseline_cnn(train_ds, val_ds, num_classes, save_path, img_height=224, img_width=224, epochs=5, use_augmenter=True, activation='relu'):
    """Trains a custom CNN from scratch to serve as a comparative baseline against MobileNetV2."""
    print(f"Initializing Custom Baseline CNN (Augmenter: {use_augmenter}, Activation: {activation})...")
    
    # Start the sequential model
    model_layers = []
    
    # 1. Input layer
    model_layers.append(layers.Input(shape=(img_height, img_width, 3)))
    
    # 2. Rescaling
    model_layers.append(layers.Rescaling(1./255))
    
    # 3. Optional Augmenter
    if use_augmenter:
        model_layers.extend([
            layers.RandomRotation(0.2),
            layers.RandomFlip("horizontal"),
            layers.RandomZoom(0.1)
        ])
        
    # 4. Custom Convolutional Blocks (WITH EXPLICIT PADDING)
    # Using padding='same' ensures spatial dimensions are preserved.
    model_layers.extend([
        layers.Conv2D(32, (3, 3), padding='same', activation=activation),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), padding='same', activation=activation),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), padding='same', activation=activation),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation=activation),
        layers.Dropout(0.5)
    ])
    
    # 5. Output Layer
    if num_classes == 1:
        model_layers.append(layers.Dense(1, activation='linear')) # Regression
        loss_fn = 'mean_squared_error'
        metrics = ['mae']
    else:
        model_layers.append(layers.Dense(num_classes, activation='softmax')) # Classification
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        metrics = ['accuracy']
        
    model = models.Sequential(model_layers)
    model.compile(optimizer='adam', loss=loss_fn, metrics=metrics)
    
    print(f"Training Baseline CNN for {epochs} epochs...")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Baseline CNN successfully saved to {save_path}")
    
    return True


# ==========================================
# 6. CNN TRAINING PIPELINE - BIOMASS REGRESSION (Dataset 6)
# ==========================================
def train_biomass_cnn_regressor(df, save_path, target='fresh_weight', img_height=224, img_width=224, epochs=10):
    """
    Trains a CNN Regressor to predict continuous biomass (fresh or dry weight)
    from plant images.

    Uses the multi-angle expanded DataFrame produced by setup_biomass_data.py,
    where each plant contributes up to 4 side-view images, effectively
    quadrupling the 41-sample dataset to up to 164 training examples.

    Args:
        df:          DataFrame from load_biomass_data() with 'filepath' and
                     target weight columns.
        save_path:   Path to save the trained .keras model.
        target:      Column to regress on — 'fresh_weight' or 'dry_weight'.
        img_height:  Input image height.
        img_width:   Input image width.
        epochs:      Number of training epochs.
    """
    print(f"Initializing CNN Regressor Training Pipeline (Biomass — target: {target})...")

    if target not in df.columns:
        print(f"[!] ERROR: Target column '{target}' not found in dataframe.")
        return False

    df_clean = df.dropna(subset=['filepath', target]).copy()
    df_clean = df_clean[df_clean['filepath'].apply(lambda p: os.path.exists(p) and os.path.getsize(p) > 0)]

    if df_clean.empty:
        print("[!] ERROR: No valid image paths found for biomass dataset.")
        return False

    print(f"Loaded {len(df_clean)} image records ({df_clean['plant_id'].nunique()} unique plants) for Biomass Regression training")

    # Split by plant_id to avoid data leakage (same plant across train/val)
    unique_plants = df_clean['plant_id'].unique()
    n_val = max(1, int(len(unique_plants) * 0.2))
    rng = np.random.default_rng(seed=42)
    val_plants = set(rng.choice(unique_plants, size=n_val, replace=False))

    train_df = df_clean[~df_clean['plant_id'].isin(val_plants)]
    val_df   = df_clean[df_clean['plant_id'].isin(val_plants)]
    print(f"Training on {len(train_df)} images ({len(train_df['plant_id'].unique())} plants), "
          f"validating on {len(val_df)} images ({len(val_df['plant_id'].unique())} plants).")
          
    try:
        from src.training.split_tracker import update_manifest
        update_manifest("demeter_cnn_biomass", train_df['filepath'].values, val_df['filepath'].values)
    except Exception as e:
        print(f"Failed to save manifest: {e}")

    def process_path(file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [img_height, img_width])
        return img, label

    def create_dataset(dataframe, shuffle=False):
        paths  = dataframe['filepath'].values
        labels = dataframe[target].values.astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(paths), seed=42)
        ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(16).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = create_dataset(train_df, shuffle=True)
    val_ds   = create_dataset(val_df,   shuffle=False)

    # Architecture: MobileNetV2 backbone + regression head
    base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    augmenter = get_augmenter(img_height, img_width)

    model = models.Sequential([
        augmenter,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear')  # Continuous output: grams
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    print(f"Training Biomass CNN Regressor for {epochs} epochs...")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    loss, mae = model.evaluate(val_ds, verbose=0)
    print(f"Biomass CNN Regressor ({target}) — Val MSE: {loss:.4f}, Val MAE: {mae:.4f} g")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Biomass CNN Regressor successfully saved to {save_path}")

    return True
