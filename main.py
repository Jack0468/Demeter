import time
import os
import glob
from inference_engine import load_models, analyze_plant_status, log_to_csv
from model_builder import train_and_save_cnn, train_and_save_rf

TRAIN_MODEL = True # Set to True to train models from scratch (will take time, especially without GPU)

# Define file paths
cnn_model_path = 'models/demeter_cnn.keras'
rf_model_path = 'models/demeter_rf.joblib'
csv_log_path = 'data/plant_diagnostics.csv'

# NOTE: You must provide the paths to the Kaggle datasets here
image_dataset_train_dir = "data/raw_images/split_ttv_dataset_type_of_plants/Train_Set_Folder" 
image_dataset_validation_dir = "data/raw_images/split_ttv_dataset_type_of_plants/Validation_Set_Folder" 
image_dataset_test_dir = "data/raw_images/split_ttv_dataset_type_of_plants/Test_Set_Folder" 


# Note: You will need to check the exact filename of the CSV downloaded 
# inside the 'tabular' folder to replace 'plant_growth_data.csv'
tabular_dataset_csv = "data/tabular/plant_growth_data.csv"

# Your Kaggle dataset classes (ensure these match the directory names downloaded)
CLASS_NAMES = sorted([f for f in os.listdir(image_dataset_train_dir) if os.path.isdir(os.path.join(image_dataset_train_dir, f))])

print(f"Detected {len(CLASS_NAMES)} plant classes.")

def main():
    print("Starting Demeter Software Pipeline...")
    
    # --- TRAINING CHECK ---
    # If the models don't exist, build them automatically
    if not os.path.exists(cnn_model_path) or not os.path.exists(rf_model_path) or TRAIN_MODEL:
        print("\n[!] Models not found. Initiating training sequence...")
        

        
        if not os.path.exists(cnn_model_path) or TRAIN_MODEL:
            train_and_save_cnn(image_dataset_train_dir, cnn_model_path, epochs=5)
            
        if not os.path.exists(rf_model_path) or TRAIN_MODEL:
            train_and_save_rf(tabular_dataset_csv, rf_model_path)
            
        print("[!] Training complete.\n")

    # --- INFERENCE PIPELINE ---
    print("Loading AI Models...")
    cnn_model, rf_model = load_models(cnn_model_path, rf_model_path)
    print("System Online. Generating diagnoses...\n")
    # --- AUTOMATIC TEST DATA LOADING ---
    # This looks into your Test folder and finds .jpg or .png files to test
    
    # Get a list of image paths (looking into subfolders if necessary)
    test_images = []
    for extension in ('/**/*.jpg', '/**/*.jpeg', '/**/*.png'):
        test_images.extend(glob.glob(image_dataset_test_dir + extension, recursive=True))

    if not test_images:
        print(f"No images found in {image_dataset_test_dir}. Please check your path.")
        return

    # Let's test the first 5 images found in the test set
    print(f"Found {len(test_images)} images. Testing first 5...")
    
    for i, img_path in enumerate(test_images[:5]):
        # Since we are using real images but don't have a live sensor, 
        # we provide sample sensor data (temp, moisture, light)
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
