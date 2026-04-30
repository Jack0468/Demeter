"""
Output Formatter: Converts raw model predictions to standardized JSON schema
for dashboard consumption and persistent logging.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional


class OutputFormatter:
    """Serializes inference results into dashboard-compatible JSON."""
    
    def __init__(self, output_dir: str = "data/outputs"):
        """
        Initialize formatter with output directory.
        
        Args:
            output_dir: Directory to store JSON outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    @staticmethod
    def format_disease_detection(
        image_path: str,
        detected_disease: str,
        confidence: float,
        all_predictions: Dict[str, float],
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format CNN disease detection results.
        
        Args:
            image_path: Path to analyzed plant image
            detected_disease: Primary disease classification
            confidence: Confidence score (0-1)
            all_predictions: Dict of all class predictions {class: confidence}
            timestamp: ISO timestamp (auto-generated if None)
            
        Returns:
            Formatted disease result dict
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Sort predictions by confidence and get top 3
        sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
        top_3 = [{"class": cls, "confidence": float(conf)} for cls, conf in sorted_predictions[:3]]
        
        return {
            "timestamp": timestamp,
            "image_path": image_path,
            "cnn_result": {
                "primary_disease": detected_disease,
                "confidence": float(confidence),
                "top_3": top_3
            }
        }
    
    @staticmethod
    def format_growth_prediction(
        predicted_growth: float,
        environmental_input: Dict[str, Any],
        timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format RF growth prediction results.
        
        Args:
            predicted_growth: Continuous growth metric prediction
            environmental_input: Input sensor data dict
            timestamp: ISO timestamp (auto-generated if None)
            
        Returns:
            Formatted growth prediction dict
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        return {
            "timestamp": timestamp,
            "rf_result": {
                "predicted_growth": float(predicted_growth),
                "trajectory": "stable"  # Will be enriched by status_engine
            },
            "environmental_input": environmental_input
        }
    
    @staticmethod
    def format_sensor_data(
        temperature: float,
        soil_moisture: float,
        sunlight_hours: float,
        humidity: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Format environmental sensor readings.
        
        Args:
            temperature: Temperature in Celsius
            soil_moisture: Soil moisture percentage (0-100)
            sunlight_hours: Sunlight hours
            humidity: Relative humidity percentage (optional)
            
        Returns:
            Formatted sensor dict
        """
        sensors = {
            "temperature": float(temperature),
            "soil_moisture": float(soil_moisture),
            "sunlight_hours": float(sunlight_hours)
        }
        
        if humidity is not None:
            sensors["humidity"] = float(humidity)
        
        return sensors
    
    def merge_diagnosis(
        self,
        disease_result: Dict[str, Any],
        growth_result: Dict[str, Any],
        sensor_data: Dict[str, Any],
        stress_diagnosis: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Merge all diagnosis components into unified result.
        
        Args:
            disease_result: Output from format_disease_detection()
            growth_result: Output from format_growth_prediction()
            sensor_data: Output from format_sensor_data()
            stress_diagnosis: Stress levels dict {factor: level}
            
        Returns:
            Complete diagnosis record
        """
        merged = {
            **disease_result,
            **growth_result,
            "sensors": sensor_data,
            "stress_diagnosis": stress_diagnosis or {
                "moisture_stress": "Unknown",
                "temperature_stress": "Unknown",
                "light_deficit": "Unknown",
                "nutrient_status": "Unknown"
            }
        }
        
        return merged
    
    def save_latest(self, diagnosis: Dict[str, Any]) -> str:
        """
        Save diagnosis as latest_diagnosis.json (overwrites previous).
        
        Args:
            diagnosis: Complete diagnosis dict
            
        Returns:
            Path to saved file
        """
        output_path = os.path.join(self.output_dir, "latest_diagnosis.json")
        with open(output_path, 'w') as f:
            json.dump(diagnosis, f, indent=2, default=str)
        return output_path
    
    def append_history(self, diagnosis: Dict[str, Any], max_records: int = 100) -> str:
        """
        Append diagnosis to history log, keeping only latest N records.
        
        Args:
            diagnosis: Complete diagnosis dict
            max_records: Maximum number of records to keep in history
            
        Returns:
            Path to history file
        """
        history_path = os.path.join(self.output_dir, "diagnosis_history.json")
        
        # Load existing history
        history = []
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
            except (json.JSONDecodeError, IOError):
                history = []
        
        # Append new diagnosis
        history.append(diagnosis)
        
        # Keep only latest max_records
        history = history[-max_records:]
        
        # Save updated history
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        return history_path


# Convenience functions for quick usage
def quick_format_diagnosis(
    image_path: str,
    disease: str,
    disease_conf: float,
    all_predictions: Dict[str, float],
    predicted_growth: float,
    temperature: float,
    soil_moisture: float,
    sunlight_hours: float,
    humidity: Optional[float] = None,
    stress_diagnosis: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    One-shot function to format a complete diagnosis.
    
    Returns:
        Complete diagnosis dict ready for dashboard/logging
    """
    formatter = OutputFormatter()
    
    disease_result = OutputFormatter.format_disease_detection(
        image_path, disease, disease_conf, all_predictions
    )
    
    growth_result = OutputFormatter.format_growth_prediction(
        predicted_growth, {"sensors": {"temperature": temperature, "humidity": humidity}}
    )
    
    sensor_data = OutputFormatter.format_sensor_data(
        temperature, soil_moisture, sunlight_hours, humidity
    )
    
    diagnosis = formatter.merge_diagnosis(
        disease_result, growth_result, sensor_data, stress_diagnosis
    )
    
    return diagnosis
