import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

def build_transfer_learning_cnn(num_classes, img_height=150, img_width=150):
    """
    Builds a CNN using a pre-trained base model, similar to the GitHub repo's approach.
    """
    # 1. Load the pre-trained base model (without the final classification layers)
    # weights='imagenet' means it comes pre-loaded with knowledge of millions of images
    base_model = MobileNetV2(input_shape=(img_height, img_width, 3),
                             include_top=False, 
                             weights='imagenet')
    
    # 2. Freeze the base model so we don't destroy its pre-trained knowledge during early training
    base_model.trainable = False 

    # 3. Build YOUR model on top of the pre-trained base
    model = models.Sequential([
        # Data Augmentation (essential for smaller datasets)
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        
        # MobileNetV2 expects pixel values between [-1, 1], not [0, 1]
        layers.Rescaling(1./127.5, offset=-1),
        
        # The pre-trained brain
        base_model,
        
        # Our custom classification head
        layers.GlobalAveragePooling2D(), # Flattens the output
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax') # The output layer for your specific plants
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model
