import time
import os
from inference_engine import load_models, analyze_plant_status, log_to_csv
from model_builder import train_and_save_cnn, train_and_save_rf

# Your Kaggle dataset classes (ensure these match the directory names downloaded)
CLASS_NAMES = ['Tomato', 'Succulent', 'Fern', 'Orchid'] 

def main():
    print("Starting Demeter Software Pipeline...")
    
    # Define file paths
    cnn_model_path = 'models/demeter_cnn.keras'
    rf_model_path = 'models/demeter_rf.joblib'
    csv_log_path = 'data/plant_diagnostics.csv'
    
    # --- TRAINING CHECK ---
    # If the models don't exist, build them automatically
    if not os.path.exists(cnn_model_path) or not os.path.exists(rf_model_path):
        print("\n[!] Models not found. Initiating training sequence...")
        
        # NOTE: You must provide the paths to the Kaggle datasets here
        image_dataset_dir = "data/raw_images" 
        
        # Note: You will need to check the exact filename of the CSV downloaded 
        # inside the 'tabular' folder to replace 'plant_growth_data.csv'
        tabular_dataset_csv = "data/tabular/plant_growth_data.csv"
        
        if not os.path.exists(cnn_model_path):
            train_and_save_cnn(image_dataset_dir, cnn_model_path, epochs=5)
            
        if not os.path.exists(rf_model_path):
            train_and_save_rf(tabular_dataset_csv, rf_model_path)
            
        print("[!] Training complete.\n")

    # --- INFERENCE PIPELINE ---
    print("Loading AI Models...")
    cnn_model, rf_model = load_models(cnn_model_path, rf_model_path)
    print("System Online. Generating diagnoses...\n")
    
    dummy_readings = [
        {"image": "test_images/plant_1.jpg", "temp": 25.5, "moisture": 30.0, "light": 4000},
        {"image": "test_images/plant_2.jpg", "temp": 22.0, "moisture": 15.0, "light": 8000}
    ]
    
    for reading in dummy_readings:
        if os.path.exists(reading["image"]):
            diagnosis = analyze_plant_status(
                image_path=reading["image"], 
                temp=reading["temp"], moisture=reading["moisture"], light=reading["light"], 
                cnn_model=cnn_model, rf_model=rf_model, class_names=CLASS_NAMES
            )
            
            print(f"[{diagnosis['Timestamp']}] {diagnosis['Species']} "
                  f"| Water: {diagnosis['Needs_Water']} "
                  f"| Fert: {diagnosis['Needs_Fertilizer']} "
                  f"| Light: {diagnosis['Needs_More_Sunlight']}")
            
            log_to_csv(diagnosis, filepath=csv_log_path)
        else:
            print(f"Skipping {reading['image']}: File not found.")
            
        time.sleep(1)

if __name__ == "__main__":
    main()
