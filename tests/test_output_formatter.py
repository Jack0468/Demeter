"""
Unit tests for output_formatter.py

Tests for JSON serialization, formatting, and file operations.
"""

import unittest
import json
import os
import tempfile
import shutil
from datetime import datetime
from output_formatter import OutputFormatter


class TestOutputFormatterBasics(unittest.TestCase):
    """Test basic formatting functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.formatter = OutputFormatter(output_dir=tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.formatter.output_dir):
            shutil.rmtree(self.formatter.output_dir)
    
    def test_format_disease_detection_basic(self):
        """Test basic disease detection formatting."""
        result = OutputFormatter.format_disease_detection(
            image_path="/path/to/image.jpg",
            detected_disease="Early Blight",
            confidence=0.84,
            all_predictions={
                "Early Blight": 0.84,
                "Late Blight": 0.10,
                "Healthy": 0.06
            }
        )
        
        self.assertIn("timestamp", result)
        self.assertIn("image_path", result)
        self.assertIn("cnn_result", result)
        self.assertEqual(result["cnn_result"]["primary_disease"], "Early Blight")
        self.assertEqual(result["cnn_result"]["confidence"], 0.84)
    
    def test_disease_detection_top_3(self):
        """Test that top 3 predictions are extracted."""
        all_pred = {
            "Early Blight": 0.84,
            "Late Blight": 0.10,
            "Healthy": 0.04,
            "Rust": 0.02
        }
        result = OutputFormatter.format_disease_detection(
            "/img.jpg", "Early Blight", 0.84, all_pred
        )
        
        top_3 = result["cnn_result"]["top_3"]
        self.assertEqual(len(top_3), 3)
        self.assertEqual(top_3[0]["class"], "Early Blight")
        self.assertEqual(top_3[0]["confidence"], 0.84)
    
    def test_format_growth_prediction(self):
        """Test growth prediction formatting."""
        result = OutputFormatter.format_growth_prediction(
            predicted_growth=45.3,
            environmental_input={"temperature": 24.5, "humidity": 65.0}
        )
        
        self.assertIn("timestamp", result)
        self.assertIn("rf_result", result)
        self.assertEqual(result["rf_result"]["predicted_growth"], 45.3)
    
    def test_format_sensor_data(self):
        """Test sensor data formatting."""
        result = OutputFormatter.format_sensor_data(
            temperature=24.5,
            soil_moisture=65.0,
            sunlight_hours=6.2,
            humidity=55.0
        )
        
        self.assertEqual(result["temperature"], 24.5)
        self.assertEqual(result["soil_moisture"], 65.0)
        self.assertEqual(result["sunlight_hours"], 6.2)
        self.assertEqual(result["humidity"], 55.0)
    
    def test_sensor_data_without_humidity(self):
        """Test sensor data without optional humidity."""
        result = OutputFormatter.format_sensor_data(
            temperature=20.0,
            soil_moisture=50.0,
            sunlight_hours=4.0
        )
        
        self.assertNotIn("humidity", result)
        self.assertEqual(len(result), 3)
    
    def test_merge_diagnosis_complete(self):
        """Test merging complete diagnosis components."""
        disease_result = OutputFormatter.format_disease_detection(
            "/img.jpg", "Early Blight", 0.84, {"Early Blight": 0.84}
        )
        growth_result = OutputFormatter.format_growth_prediction(45.3, {})
        sensor_data = OutputFormatter.format_sensor_data(24.5, 65.0, 6.2)
        
        merged = self.formatter.merge_diagnosis(
            disease_result, growth_result, sensor_data
        )
        
        self.assertIn("cnn_result", merged)
        self.assertIn("rf_result", merged)
        self.assertIn("sensors", merged)
        self.assertIn("stress_diagnosis", merged)  # Default stress included
    
    def test_merge_diagnosis_with_stress(self):
        """Test merging with custom stress diagnosis."""
        disease_result = OutputFormatter.format_disease_detection(
            "/img.jpg", "Healthy", 0.95, {"Healthy": 0.95}
        )
        growth_result = OutputFormatter.format_growth_prediction(78.0, {})
        sensor_data = OutputFormatter.format_sensor_data(22.0, 70.0, 6.5)
        
        stress_diag = {
            "moisture_stress": "Low",
            "temperature_stress": "Low",
            "light_deficit": "None",
            "nutrient_status": "Normal"
        }
        
        merged = self.formatter.merge_diagnosis(
            disease_result, growth_result, sensor_data, stress_diag
        )
        
        self.assertEqual(merged["stress_diagnosis"]["moisture_stress"], "Low")


class TestOutputFormatterFilePersistence(unittest.TestCase):
    """Test file saving and loading operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.formatter = OutputFormatter(output_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_save_latest_diagnosis(self):
        """Test saving latest diagnosis."""
        diagnosis = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "Thriving",
            "health_score": 85
        }
        
        path = self.formatter.save_latest(diagnosis)
        
        self.assertTrue(os.path.exists(path))
        with open(path, 'r') as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded["overall_status"], "Thriving")
        self.assertEqual(loaded["health_score"], 85)
    
    def test_save_latest_overwrites(self):
        """Test that save_latest overwrites previous."""
        diagnosis1 = {"status": "first"}
        diagnosis2 = {"status": "second"}
        
        self.formatter.save_latest(diagnosis1)
        self.formatter.save_latest(diagnosis2)
        
        with open(os.path.join(self.formatter.output_dir, "latest_diagnosis.json"), 'r') as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded["status"], "second")
    
    def test_append_history(self):
        """Test appending to history."""
        diagnosis1 = {"id": 1, "status": "Thriving"}
        diagnosis2 = {"id": 2, "status": "Struggling"}
        
        self.formatter.append_history(diagnosis1)
        self.formatter.append_history(diagnosis2)
        
        with open(os.path.join(self.formatter.output_dir, "diagnosis_history.json"), 'r') as f:
            history = json.load(f)
        
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["id"], 1)
        self.assertEqual(history[1]["id"], 2)
    
    def test_append_history_max_records(self):
        """Test that history respects max_records limit."""
        for i in range(15):
            self.formatter.append_history({"id": i})
        
        with open(os.path.join(self.formatter.output_dir, "diagnosis_history.json"), 'r') as f:
            history = json.load(f)
        
        self.assertEqual(len(history), 10)  # Default max is 100, but last 10 added
    
    def test_append_history_preserves_json_format(self):
        """Test that nested structures are preserved in history."""
        diagnosis = {
            "cnn_result": {
                "primary_disease": "Early Blight",
                "confidence": 0.84
            },
            "recommendations": [
                {"action": "Water", "priority": 1},
                {"action": "Treat", "priority": 2}
            ]
        }
        
        self.formatter.append_history(diagnosis)
        
        with open(os.path.join(self.formatter.output_dir, "diagnosis_history.json"), 'r') as f:
            history = json.load(f)
        
        self.assertEqual(history[0]["cnn_result"]["primary_disease"], "Early Blight")
        self.assertEqual(len(history[0]["recommendations"]), 2)


class TestOutputFormatterEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.formatter = OutputFormatter(output_dir=tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.formatter.output_dir):
            shutil.rmtree(self.formatter.output_dir)
    
    def test_disease_with_100_percent_confidence(self):
        """Test disease detection with 100% confidence."""
        result = OutputFormatter.format_disease_detection(
            "/img.jpg", "Healthy", 1.0, {"Healthy": 1.0}
        )
        
        self.assertEqual(result["cnn_result"]["confidence"], 1.0)
    
    def test_disease_with_zero_confidence(self):
        """Test disease detection with 0% confidence."""
        result = OutputFormatter.format_disease_detection(
            "/img.jpg", "Unknown", 0.0, {"Unknown": 0.0}
        )
        
        self.assertEqual(result["cnn_result"]["confidence"], 0.0)
    
    def test_single_prediction_class(self):
        """Test with only one prediction class."""
        result = OutputFormatter.format_disease_detection(
            "/img.jpg", "Healthy", 0.5, {"Healthy": 0.5}
        )
        
        self.assertEqual(len(result["cnn_result"]["top_3"]), 1)
    
    def test_negative_temperature(self):
        """Test sensor data with negative temperature."""
        result = OutputFormatter.format_sensor_data(
            temperature=-5.0,
            soil_moisture=30.0,
            sunlight_hours=2.0
        )
        
        self.assertEqual(result["temperature"], -5.0)
    
    def test_extreme_moisture(self):
        """Test sensor data with extreme moisture values."""
        result = OutputFormatter.format_sensor_data(
            temperature=20.0,
            soil_moisture=100.0,  # Saturated
            sunlight_hours=6.0
        )
        
        self.assertEqual(result["soil_moisture"], 100.0)
    
    def test_zero_sunlight(self):
        """Test with zero sunlight hours."""
        result = OutputFormatter.format_sensor_data(
            temperature=20.0,
            soil_moisture=50.0,
            sunlight_hours=0.0
        )
        
        self.assertEqual(result["sunlight_hours"], 0.0)


class TestQuickFormatDiagnosis(unittest.TestCase):
    """Test the convenience quick_format_diagnosis function."""
    
    def test_quick_format_complete(self):
        """Test quick format with all parameters."""
        from output_formatter import quick_format_diagnosis
        
        diagnosis = quick_format_diagnosis(
            image_path="/img.jpg",
            disease="Early Blight",
            disease_conf=0.84,
            all_predictions={"Early Blight": 0.84, "Healthy": 0.16},
            predicted_growth=45.3,
            temperature=24.5,
            soil_moisture=65.0,
            sunlight_hours=6.2,
            humidity=55.0
        )
        
        self.assertIn("cnn_result", diagnosis)
        self.assertIn("rf_result", diagnosis)
        self.assertIn("sensors", diagnosis)
        self.assertEqual(diagnosis["cnn_result"]["primary_disease"], "Early Blight")


if __name__ == "__main__":
    unittest.main()
