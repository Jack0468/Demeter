import time
import os
from inference_engine import load_vision_model, analyze_plant_status, log_to_csv

def main():
    print("Starting Demeter Software Pipeline...")
    
    # 1. Load the model (use relative paths for smooth operation across different OS environments)
    model_path = 'models/demeter_vision_model.keras' 
    csv_log_path = 'data/plant_diagnostics.csv'
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    try:
        cnn = load_vision_model(model_path)
    except FileNotFoundError as e:
        print(e)
        print("Please train and save your model to the 'models' directory first.")
        return

    print("Model loaded. Generating diagnoses...\n")
    
    # 2. Simulated Loop (Iterating over dummy data)
    dummy_readings = [
        {"image": "test_images/plant_1.jpg", "temp": 25.5, "moisture": 30.0, "light": 4000},
        {"image": "test_images/plant_2.jpg", "temp": 22.0, "moisture": 15.0, "light": 8000},
        {"image": "test_images/plant_3.jpg", "temp": 28.0, "moisture": 60.0, "light": 2500}
    ]
    
    for reading in dummy_readings:
        # Check if the test image exists before processing
        if os.path.exists(reading["image"]):
            
            # Get the diagnosis dictionary
            diagnosis = analyze_plant_status(
                image_path=reading["image"], 
                temp=reading["temp"], 
                moisture=reading["moisture"], 
                light=reading["light"], 
                cnn_model=cnn
            )
            
            # Print to console for real-time viewing
            print(f"[{diagnosis['Timestamp']}] Species: {diagnosis['Species']} "
                  f"| Water: {diagnosis['Needs_Water']} "
                  f"| Fert: {diagnosis['Needs_Fertilizer']} "
                  f"| Light: {diagnosis['Needs_More_Sunlight']}")
            
            # Log to CSV
            log_to_csv(diagnosis, filepath=csv_log_path)
            
        else:
            print(f"Skipping {reading['image']}: File not found.")
            
        time.sleep(1) # Short delay for simulation purposes

    print(f"\nDiagnostics complete. Results saved to {csv_log_path}")

if __name__ == "__main__":
    main()
