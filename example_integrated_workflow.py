"""
example_integrated_workflow.py

Demonstrates the complete workflow of:
1. Running inference (disease detection + growth prediction)
2. Formatting outputs with output_formatter.py
3. Generating status and recommendations with status_engine.py
4. Saving for dashboard consumption

Run this to generate sample diagnoses for testing the dashboard.
"""

import os
import sys
from output_formatter import OutputFormatter
from status_engine import StatusEngine

def example_workflow():
    """
    Generate example diagnoses for testing the dashboard.
    """
    
    print("=" * 70)
    print("DEMETER - INTEGRATED WORKFLOW EXAMPLE")
    print("=" * 70)
    
    # ==========================================
    # Example 1: Plant with Disease
    # ==========================================
    print("\n[1] Generating Example 1: Plant with Disease Detection")
    
    formatter = OutputFormatter()
    status_engine = StatusEngine()
    
    # Simulate model outputs for a diseased plant
    image_path = "data/layer2_health_rgb/PlantVillage/Tomato__Early_blight/sample.jpg"
    detected_disease = "Early Blight"
    disease_confidence = 0.84
    
    all_predictions = {
        "Early Blight": 0.84,
        "Late Blight": 0.08,
        "Healthy": 0.05,
        "Septoria": 0.03
    }
    
    predicted_growth = 42.5  # RF prediction
    
    # Sensor data (poor conditions)
    temperature = 24.5
    soil_moisture = 32.0  # LOW - stress
    sunlight_hours = 4.2
    humidity = 72.0
    
    # Step 1: Format disease detection
    disease_result = OutputFormatter.format_disease_detection(
        image_path, detected_disease, disease_confidence, all_predictions
    )
    print(f"  ✓ Disease formatted: {detected_disease} ({disease_confidence:.1%})")
    
    # Step 2: Format growth prediction
    growth_result = OutputFormatter.format_growth_prediction(
        predicted_growth,
        {"temperature": temperature, "humidity": humidity}
    )
    print(f"  ✓ Growth prediction formatted: {predicted_growth}")
    
    # Step 3: Format sensor data
    sensor_data = OutputFormatter.format_sensor_data(
        temperature, soil_moisture, sunlight_hours, humidity
    )
    print(f"  ✓ Sensor data formatted")
    
    # Step 4: Merge all components
    merged = formatter.merge_diagnosis(
        disease_result, growth_result, sensor_data
    )
    print(f"  ✓ Diagnosis merged")
    
    # Step 5: Generate status, recommendations, system commands
    full_diagnosis = status_engine.generate_full_diagnosis(
        disease_confidence, detected_disease, soil_moisture,
        temperature, sunlight_hours, predicted_growth, humidity
    )
    print(f"  ✓ Status engine generated recommendations")
    
    # Combine everything
    diagnosis_1 = {**merged, **full_diagnosis}
    
    # Save outputs
    formatter.save_latest(diagnosis_1)
    formatter.append_history(diagnosis_1)
    print(f"  ✓ Diagnosis saved to data/outputs/")
    
    # Print summary
    print(f"\n  === DIAGNOSIS SUMMARY ===")
    print(f"  Health Score: {diagnosis_1['health_score']}/100")
    print(f"  Overall Status: {diagnosis_1['overall_status']}")
    print(f"  Disease: {detected_disease} ({disease_confidence:.1%})")
    print(f"  Moisture Stress: {diagnosis_1['stress_diagnosis']['moisture_stress']}")
    print(f"  System Command: {diagnosis_1['system_command']}")
    print(f"  Recommendations: {len(diagnosis_1['recommendations'])} actions")
    
    # ==========================================
    # Example 2: Healthy Plant
    # ==========================================
    print("\n[2] Generating Example 2: Healthy Plant")
    
    # Reset for healthy plant
    detected_disease = "Healthy"
    disease_confidence = 0.96
    all_predictions = {
        "Healthy": 0.96,
        "Early Blight": 0.02,
        "Late Blight": 0.01,
        "Septoria": 0.01
    }
    
    predicted_growth = 78.5  # Good growth
    temperature = 22.0
    soil_moisture = 65.0  # GOOD
    sunlight_hours = 6.5  # GOOD
    humidity = 55.0
    
    disease_result = OutputFormatter.format_disease_detection(
        image_path, detected_disease, disease_confidence, all_predictions
    )
    
    growth_result = OutputFormatter.format_growth_prediction(
        predicted_growth,
        {"temperature": temperature, "humidity": humidity}
    )
    
    sensor_data = OutputFormatter.format_sensor_data(
        temperature, soil_moisture, sunlight_hours, humidity
    )
    
    merged = formatter.merge_diagnosis(disease_result, growth_result, sensor_data)
    
    full_diagnosis = status_engine.generate_full_diagnosis(
        disease_confidence, detected_disease, soil_moisture,
        temperature, sunlight_hours, predicted_growth, humidity
    )
    
    diagnosis_2 = {**merged, **full_diagnosis}
    formatter.append_history(diagnosis_2)
    
    print(f"  ✓ Diagnosis created and saved")
    print(f"\n  === DIAGNOSIS SUMMARY ===")
    print(f"  Health Score: {diagnosis_2['health_score']}/100")
    print(f"  Overall Status: {diagnosis_2['overall_status']}")
    print(f"  Disease: {detected_disease} ({disease_confidence:.1%})")
    print(f"  Moisture Stress: {diagnosis_2['stress_diagnosis']['moisture_stress']}")
    print(f"  System Command: {diagnosis_2['system_command']}")
    print(f"  Recommendations: {len(diagnosis_2['recommendations'])} actions")
    
    # ==========================================
    # Example 3: Critical Plant
    # ==========================================
    print("\n[3] Generating Example 3: Critical Plant (Stress)")
    
    detected_disease = "Late Blight"
    disease_confidence = 0.78
    all_predictions = {
        "Late Blight": 0.78,
        "Early Blight": 0.15,
        "Healthy": 0.04,
        "Septoria": 0.03
    }
    
    predicted_growth = 18.2  # Poor growth
    temperature = 28.5  # TOO HOT
    soil_moisture = 18.0  # VERY LOW
    sunlight_hours = 2.1  # TOO LOW
    humidity = 45.0
    
    disease_result = OutputFormatter.format_disease_detection(
        image_path, detected_disease, disease_confidence, all_predictions
    )
    
    growth_result = OutputFormatter.format_growth_prediction(
        predicted_growth,
        {"temperature": temperature, "humidity": humidity}
    )
    
    sensor_data = OutputFormatter.format_sensor_data(
        temperature, soil_moisture, sunlight_hours, humidity
    )
    
    merged = formatter.merge_diagnosis(disease_result, growth_result, sensor_data)
    
    full_diagnosis = status_engine.generate_full_diagnosis(
        disease_confidence, detected_disease, soil_moisture,
        temperature, sunlight_hours, predicted_growth, humidity
    )
    
    diagnosis_3 = {**merged, **full_diagnosis}
    formatter.append_history(diagnosis_3)
    
    print(f"  ✓ Diagnosis created and saved")
    print(f"\n  === DIAGNOSIS SUMMARY ===")
    print(f"  Health Score: {diagnosis_3['health_score']}/100")
    print(f"  Overall Status: {diagnosis_3['overall_status']}")
    print(f"  Disease: {detected_disease} ({disease_confidence:.1%})")
    print(f"  Moisture Stress: {diagnosis_3['stress_diagnosis']['moisture_stress']}")
    print(f"  Temperature Stress: {diagnosis_3['stress_diagnosis']['temperature_stress']}")
    print(f"  Light Deficit: {diagnosis_3['stress_diagnosis']['light_deficit']}")
    print(f"  System Command: {diagnosis_3['system_command']}")
    print(f"  Recommendations: {len(diagnosis_3['recommendations'])} actions")
    for i, rec in enumerate(diagnosis_3['recommendations'], 1):
        print(f"    {i}. [{rec['urgency']}] {rec['action']}")
    
    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    print("\nGenerated 3 example diagnoses:")
    print("  1. Diseased plant with moisture stress")
    print("  2. Healthy thriving plant")
    print("  3. Critical plant (multiple stresses)")
    print("\nFiles created:")
    print(f"  ✓ data/outputs/latest_diagnosis.json (most recent)")
    print(f"  ✓ data/outputs/diagnosis_history.json (all diagnoses)")
    print("\nNext steps:")
    print("  1. Start API server:     python api_server.py")
    print("  2. Open dashboard:       http://localhost:5000/dashboard")
    print("  3. Dashboard will auto-load and display diagnoses")
    print("\nTo add more diagnoses:")
    print("  - Run this script again to generate more examples")
    print("  - Run main.py for real model predictions")
    print("  - Dashboard will auto-refresh every 5 seconds")
    print("\n" + "=" * 70)


def view_latest_diagnosis():
    """Helper function to view the latest diagnosis."""
    import json
    latest_path = "data/outputs/latest_diagnosis.json"
    
    if not os.path.exists(latest_path):
        print("No diagnoses generated yet. Run example_workflow() first.")
        return
    
    with open(latest_path, 'r') as f:
        diagnosis = json.load(f)
    
    print("\nLatest Diagnosis (JSON):")
    print(json.dumps(diagnosis, indent=2))


def get_statistics():
    """Show statistics from all generated diagnoses."""
    import json
    history_path = "data/outputs/diagnosis_history.json"
    
    if not os.path.exists(history_path):
        print("No diagnoses generated yet.")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    if not history:
        print("History file is empty.")
        return
    
    print(f"\nDiagnosis History Statistics:")
    print(f"  Total diagnoses: {len(history)}")
    
    health_scores = [d.get('health_score', 0) for d in history]
    statuses = [d.get('overall_status', 'Unknown') for d in history]
    
    print(f"  Average health score: {sum(health_scores) / len(health_scores):.1f}/100")
    print(f"  Status distribution:")
    for status in set(statuses):
        count = statuses.count(status)
        print(f"    - {status}: {count}")


if __name__ == "__main__":
    print("\nDemeter - Integrated Workflow Example")
    print("=" * 70)
    print("\nThis script demonstrates the complete diagnosis workflow.")
    print("It generates 3 example diagnoses and saves them for dashboard viewing.\n")
    
    # Run the workflow
    example_workflow()
    
    # Optionally view details
    print("\n\nWould you like to see more details? (optional)")
    try:
        view_latest_diagnosis()
        get_statistics()
    except Exception as e:
        print(f"(Skipped details: {e})")
