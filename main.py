import time
import os
import glob
import json
from inference_engine import load_models, analyze_plant_status, log_to_csv
from model_builder import train_and_save_cnn, train_and_save_rf

# --- CONFIGURATION LOADING ---
# Load settings from config.json
try:
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    print("[!] ERROR: config.json not found. Please copy config.example.json to config.json and update the paths.")
    exit(1)

TRAIN_MODEL = config['training']['force_retrain']

# Define file paths dynamically from config
cnn_model_path = config['paths']['cnn_model']
rf_model_path = config['paths']['rf_model']
csv_log_path = config['paths']['csv_log']

image_dataset_train_dir = config['paths']['image_train_dir']
image_dataset_validation_dir = config['paths']['image_val_dir']
image_dataset_test_dir = config['paths']['image_test_dir']
tabular_dataset_csv = config['paths']['tabular_csv']

# Detect Kaggle dataset classes dynamically based on your train directory
try:
    CLASS_NAMES = sorted([f for f in os.listdir(image_dataset_train_dir) if os.path.isdir(os.path.join(image_dataset_train_dir, f))])
    print(f"Detected {len(CLASS_NAMES)} plant classes.")
except FileNotFoundError:
    print(f"[!] WARNING: Training directory not found at {image_dataset_train_dir}")
    CLASS_NAMES = []


def main():
    print("Starting Demeter Software Pipeline...")
    
    # --- TRAINING CHECK ---
    if not os.path.exists(cnn_model_path) or not os.path.exists(rf_model_path) or TRAIN_MODEL:
        print("\n[!] Models not found or retrain forced. Initiating training sequence...")
        
        if not os.path.exists(cnn_model_path) or TRAIN_MODEL:
            train_and_save_cnn(image_dataset_train_dir, cnn_model_path, epochs=config['training']['epochs'])
            
        if not os.path.exists(rf_model_path) or TRAIN_MODEL:
            train_and_save_rf(tabular_dataset_csv, rf_model_path)
            
        print("[!] Training complete.\n")

    # --- INFERENCE PIPELINE ---
    print("Loading AI Models...")
    cnn_model, rf_model = load_models(cnn_model_path, rf_model_path)
    print("System Online. Generating diagnoses...\n")
    
    # --- AUTOMATIC TEST DATA LOADING ---
    test_images = []
    for extension in ('/**/*.jpg', '/**/*.jpeg', '/**/*.png'):
        test_images.extend(glob.glob(image_dataset_test_dir + extension, recursive=True))

    if not test_images:
        print(f"No images found in {image_dataset_test_dir}. Please check your config.json path.")
        return

    print(f"Found {len(test_images)} images. Testing first 5...")
    
    for i, img_path in enumerate(test_images[:5]):
        # Simulated sensor data for testing purposes
        diagnosis = analyze_plant_status(
            image_path=img_path, 
            temp=24.0, 
            moisture=20.0, 
            light=5000, 
            cnn_model=cnn_model, 
            rf_model=rf_model, 
            class_names=CLASS_NAMES
        )
        
        print(f"Test {i+1}: {os.path.basename(img_path)}")
        print(f"[{diagnosis['Timestamp']}] Predicted: {diagnosis['Species']} "
              f"| Water: {diagnosis['Needs_Water']} "
              f"| Fert: {diagnosis['Needs_Fertilizer']} "
              f"| Light: {diagnosis['Needs_More_Sunlight']}")
        print("-" * 30)
        
        log_to_csv(diagnosis, filepath=csv_log_path)
        time.sleep(0.5)

if __name__ == "__main__":
    main()