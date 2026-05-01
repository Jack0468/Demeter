"""
Unit tests for status_engine.py

Tests for health scoring, stress diagnosis, recommendations, and trajectories.
"""

import unittest
from status_engine import StatusEngine, StressThresholds


class TestStressThresholds(unittest.TestCase):
    """Test threshold configuration."""
    
    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = StressThresholds()
        
        self.assertEqual(thresholds.moisture_critical, 25.0)
        self.assertEqual(thresholds.moisture_warning, 40.0)
        self.assertEqual(thresholds.temp_too_hot, 30.0)
        self.assertEqual(thresholds.disease_critical, 0.75)
    
    def test_custom_thresholds(self):
        """Test custom threshold values."""
        thresholds = StressThresholds(
            moisture_critical=20.0,
            disease_critical=0.80
        )
        
        self.assertEqual(thresholds.moisture_critical, 20.0)
        self.assertEqual(thresholds.disease_critical, 0.80)
        # Others should retain defaults
        self.assertEqual(thresholds.moisture_warning, 40.0)


class TestMoistureStressDetection(unittest.TestCase):
    """Test moisture stress level determination."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = StatusEngine()
    
    def test_moisture_high_stress(self):
        """Test high moisture stress (very dry)."""
        level, priority = self.engine.diagnose_moisture_stress(10.0)
        self.assertEqual(level, "High")
        self.assertEqual(priority, 1)
    
    def test_moisture_medium_stress(self):
        """Test medium moisture stress."""
        level, priority = self.engine.diagnose_moisture_stress(35.0)
        self.assertEqual(level, "Medium")
        self.assertEqual(priority, 2)
    
    def test_moisture_low_stress(self):
        """Test low moisture stress."""
        level, priority = self.engine.diagnose_moisture_stress(45.0)
        self.assertEqual(level, "Low")
        self.assertEqual(priority, 3)
    
    def test_moisture_no_stress(self):
        """Test no moisture stress (well-watered)."""
        level, priority = self.engine.diagnose_moisture_stress(70.0)
        self.assertEqual(level, "None")
        self.assertEqual(priority, 0)
    
    def test_moisture_boundary_critical(self):
        """Test moisture at critical threshold."""
        level, _ = self.engine.diagnose_moisture_stress(25.0)
        self.assertEqual(level, "High")
    
    def test_moisture_boundary_healthy(self):
        """Test moisture at healthy threshold."""
        level, _ = self.engine.diagnose_moisture_stress(50.0)
        self.assertEqual(level, "None")


class TestTemperatureStressDetection(unittest.TestCase):
    """Test temperature stress level determination."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = StatusEngine()
    
    def test_temperature_too_cold(self):
        """Test temperature too cold."""
        level, priority = self.engine.diagnose_temperature_stress(10.0)
        self.assertEqual(level, "High")
        self.assertEqual(priority, 1)
    
    def test_temperature_too_hot(self):
        """Test temperature too hot."""
        level, priority = self.engine.diagnose_temperature_stress(32.0)
        self.assertEqual(level, "High")
        self.assertEqual(priority, 1)
    
    def test_temperature_slightly_cold(self):
        """Test temperature slightly below optimal."""
        level, priority = self.engine.diagnose_temperature_stress(17.0)
        self.assertEqual(level, "Medium")
        self.assertEqual(priority, 2)
    
    def test_temperature_optimal(self):
        """Test optimal temperature range."""
        level, priority = self.engine.diagnose_temperature_stress(22.0)
        self.assertEqual(level, "Low")
        self.assertEqual(priority, 3)


class TestLightDeficitDetection(unittest.TestCase):
    """Test light deficit level determination."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = StatusEngine()
    
    def test_light_high_deficit(self):
        """Test high light deficit (dark)."""
        level, priority = self.engine.diagnose_light_deficit(1.0)
        self.assertEqual(level, "High")
        self.assertEqual(priority, 1)
    
    def test_light_medium_deficit(self):
        """Test medium light deficit."""
        level, priority = self.engine.diagnose_light_deficit(3.0)
        self.assertEqual(level, "Medium")
        self.assertEqual(priority, 2)
    
    def test_light_low_deficit(self):
        """Test low light deficit."""
        level, priority = self.engine.diagnose_light_deficit(5.0)
        self.assertEqual(level, "Low")
        self.assertEqual(priority, 3)
    
    def test_light_optimal(self):
        """Test optimal light."""
        level, priority = self.engine.diagnose_light_deficit(7.0)
        self.assertEqual(level, "None")
        self.assertEqual(priority, 0)


class TestDiseaseAssessment(unittest.TestCase):
    """Test disease severity assessment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = StatusEngine()
    
    def test_disease_critical(self):
        """Test critical disease confidence."""
        severity, priority = self.engine.assess_disease_severity(0.85)
        self.assertEqual(severity, "Critical")
        self.assertEqual(priority, 1)
    
    def test_disease_warning(self):
        """Test warning disease confidence."""
        severity, priority = self.engine.assess_disease_severity(0.60)
        self.assertEqual(severity, "Warning")
        self.assertEqual(priority, 2)
    
    def test_disease_minor(self):
        """Test minor disease confidence."""
        severity, priority = self.engine.assess_disease_severity(0.30)
        self.assertEqual(severity, "Minor")
        self.assertEqual(priority, 3)
    
    def test_disease_no_detection(self):
        """Test no disease detection."""
        severity, priority = self.engine.assess_disease_severity(0.05)
        self.assertEqual(severity, "Minor")


class TestCompositeHealthScore(unittest.TestCase):
    """Test composite health score calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = StatusEngine()
    
    def test_thriving_plant(self):
        """Test plant with excellent conditions."""
        score, status = self.engine.calculate_composite_health_score(
            disease_confidence=0.05,
            soil_moisture=75.0,
            temperature=22.0,
            sunlight_hours=6.5
        )
        
        self.assertGreaterEqual(score, 70)
        self.assertEqual(status, "Thriving")
    
    def test_struggling_plant(self):
        """Test plant with moderate stress."""
        score, status = self.engine.calculate_composite_health_score(
            disease_confidence=0.60,
            soil_moisture=35.0,
            temperature=20.0,
            sunlight_hours=3.0
        )
        
        self.assertGreaterEqual(score, 40)
        self.assertLess(score, 70)
        self.assertEqual(status, "Struggling")
    
    def test_critical_plant(self):
        """Test plant in critical condition."""
        score, status = self.engine.calculate_composite_health_score(
            disease_confidence=0.90,
            soil_moisture=10.0,
            temperature=32.0,
            sunlight_hours=1.0
        )
        
        self.assertLess(score, 40)
        self.assertEqual(status, "Critical")
    
    def test_score_bounds(self):
        """Test that health score stays within 0-100."""
        for moisture in [0, 50, 100]:
            for temp in [0, 25, 35]:
                score, _ = self.engine.calculate_composite_health_score(
                    disease_confidence=0.5,
                    soil_moisture=moisture,
                    temperature=temp,
                    sunlight_hours=4.0
                )
                
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 100)


class TestRecommendationGeneration(unittest.TestCase):
    """Test recommendation generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = StatusEngine()
    
    def test_recommendations_disease(self):
        """Test recommendations for diseased plant."""
        recs = self.engine.generate_recommendations(
            disease_confidence=0.85,
            disease_name="Early Blight",
            soil_moisture=60.0,
            temperature=22.0,
            sunlight_hours=6.0
        )
        
        self.assertGreater(len(recs), 0)
        # Should have disease-related recommendation
        self.assertTrue(any("fungicide" in r["action"].lower() for r in recs))
    
    def test_recommendations_low_moisture(self):
        """Test recommendations for dry conditions."""
        recs = self.engine.generate_recommendations(
            disease_confidence=0.10,
            disease_name="Healthy",
            soil_moisture=20.0,
            temperature=22.0,
            sunlight_hours=6.0
        )
        
        # Should have watering recommendation
        self.assertTrue(any("water" in r["action"].lower() for r in recs))
    
    def test_recommendations_hot_temperature(self):
        """Test recommendations for high temperature."""
        recs = self.engine.generate_recommendations(
            disease_confidence=0.10,
            disease_name="Healthy",
            soil_moisture=70.0,
            temperature=32.0,
            sunlight_hours=6.0
        )
        
        # Should suggest shade/cooling
        self.assertTrue(any("shade" in r["action"].lower() for r in recs))
    
    def test_recommendations_low_light(self):
        """Test recommendations for low light."""
        recs = self.engine.generate_recommendations(
            disease_confidence=0.10,
            disease_name="Healthy",
            soil_moisture=70.0,
            temperature=22.0,
            sunlight_hours=1.5
        )
        
        # Should suggest light/relocation
        self.assertTrue(any("sunlight" in r["action"].lower() for r in recs))
    
    def test_max_three_recommendations(self):
        """Test that max 3 recommendations are returned."""
        recs = self.engine.generate_recommendations(
            disease_confidence=0.95,
            disease_name="Multiple Stresses",
            soil_moisture=10.0,
            temperature=35.0,
            sunlight_hours=1.0,
            humidity=30.0
        )
        
        self.assertLessEqual(len(recs), 3)


class TestSystemCommandGeneration(unittest.TestCase):
    """Test system command generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = StatusEngine()
    
    def test_monitor_only_healthy(self):
        """Test healthy plant returns monitoring command."""
        cmd = self.engine.determine_system_command(
            health_score=85,
            disease_confidence=0.05,
            soil_moisture=70.0
        )
        
        self.assertEqual(cmd, "MONITOR_ONLY")
    
    def test_activate_water_pump(self):
        """Test water pump activation for dry conditions."""
        cmd = self.engine.determine_system_command(
            health_score=50,
            disease_confidence=0.10,
            soil_moisture=20.0
        )
        
        self.assertIn("ACTIVATE_WATER_PUMP", cmd)
    
    def test_activate_spray_system(self):
        """Test spray system for disease."""
        cmd = self.engine.determine_system_command(
            health_score=60,
            disease_confidence=0.80,
            soil_moisture=70.0
        )
        
        self.assertIn("ACTIVATE_SPRAY_SYSTEM", cmd)
    
    def test_warning_alert_critical(self):
        """Test critical alert for poor health."""
        cmd = self.engine.determine_system_command(
            health_score=30,
            disease_confidence=0.50,
            soil_moisture=25.0
        )
        
        self.assertIn("SEND_CRITICAL_ALERT", cmd)
    
    def test_multiple_commands(self):
        """Test multiple commands combined."""
        cmd = self.engine.determine_system_command(
            health_score=35,
            disease_confidence=0.85,
            soil_moisture=15.0
        )
        
        # Should have multiple commands
        self.assertIn("|", cmd)


class TestTrajectoryPrediction(unittest.TestCase):
    """Test 7-day trajectory prediction."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = StatusEngine()
    
    def test_trajectory_improving(self):
        """Test trajectory for well-watered plant."""
        trajectory = self.engine.predict_7day_trajectory(
            current_health_score=80,
            growth_prediction=50.0,
            soil_moisture=70.0
        )
        
        self.assertIn(1, trajectory)
        self.assertIn(3, trajectory)
        self.assertIn(5, trajectory)
        self.assertIn(7, trajectory)
    
    def test_trajectory_declining(self):
        """Test trajectory for dry plant."""
        trajectory = self.engine.predict_7day_trajectory(
            current_health_score=50,
            growth_prediction=30.0,
            soil_moisture=15.0  # Very dry
        )
        
        # Later days should be worse
        status_day1 = trajectory.get(1, "Unknown")
        status_day7 = trajectory.get(7, "Unknown")
        
        # Day 7 should be equal or worse than Day 1
        severity_map = {"Thriving": 3, "Fair": 2, "Poor": 1}
        self.assertLessEqual(
            severity_map.get(status_day7, 0),
            severity_map.get(status_day1, 3)
        )


class TestCompleteDiagnosis(unittest.TestCase):
    """Test full diagnosis generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = StatusEngine()
    
    def test_complete_diagnosis_structure(self):
        """Test that complete diagnosis has all required fields."""
        diagnosis = self.engine.generate_full_diagnosis(
            disease_confidence=0.65,
            disease_name="Early Blight",
            soil_moisture=35.0,
            temperature=25.0,
            sunlight_hours=5.0,
            predicted_growth=50.0,
            humidity=60.0
        )
        
        # Check all major sections
        self.assertIn("stress_diagnosis", diagnosis)
        self.assertIn("health_score", diagnosis)
        self.assertIn("overall_status", diagnosis)
        self.assertIn("recommendations", diagnosis)
        self.assertIn("system_command", diagnosis)
        self.assertIn("trajectory_7day", diagnosis)
    
    def test_complete_diagnosis_stressed_plant(self):
        """Test complete diagnosis for stressed plant."""
        diagnosis = self.engine.generate_full_diagnosis(
            disease_confidence=0.80,
            disease_name="Late Blight",
            soil_moisture=20.0,
            temperature=28.0,
            sunlight_hours=2.0,
            predicted_growth=25.0,
            humidity=45.0
        )
        
        self.assertEqual(diagnosis["overall_status"], "Struggling")
        self.assertLess(diagnosis["health_score"], 60)
        self.assertGreater(len(diagnosis["recommendations"]), 0)
        self.assertNotEqual(diagnosis["system_command"], "MONITOR_ONLY")


if __name__ == "__main__":
    unittest.main()
