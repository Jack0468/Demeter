"""
Status Engine for Demeter.
Determines health score, overall status, and stress levels from raw sensor/model data.
"""

class StressThresholds:
    def __init__(self):
        self.moisture_critical = 20.0
        self.moisture_warning = 30.0
        self.moisture_healthy = 50.0
        
        self.temp_too_cold = 10.0
        self.temp_too_hot = 35.0
        self.temp_optimal_min = 18.0
        self.temp_optimal_max = 28.0
        
        self.sunlight_minimal = 2.0
        self.sunlight_moderate = 4.0
        self.sunlight_optimal = 6.0
        
        self.disease_critical = 0.8
        self.disease_warning = 0.5

class StatusEngine:
    def __init__(self):
        self.thresholds = StressThresholds()

    def generate_full_diagnosis(self, disease_confidence, detected_disease, soil_moisture, temperature, sunlight_hours, predicted_growth, humidity):
        moisture_stress = "Low"
        if soil_moisture < self.thresholds.moisture_critical:
            moisture_stress = "High"
        elif soil_moisture < self.thresholds.moisture_warning:
            moisture_stress = "Moderate"

        temp_stress = "Optimal"
        if temperature < self.thresholds.temp_too_cold or temperature > self.thresholds.temp_too_hot:
            temp_stress = "High"
        elif temperature < self.thresholds.temp_optimal_min or temperature > self.thresholds.temp_optimal_max:
            temp_stress = "Moderate"

        light_deficit = "Low"
        if sunlight_hours < self.thresholds.sunlight_minimal:
            light_deficit = "High"
        elif sunlight_hours < self.thresholds.sunlight_moderate:
            light_deficit = "Moderate"

        health_score = 100
        overall_status = "Thriving"
        system_command = "MONITORING"
        
        if moisture_stress == "High" or temp_stress == "High":
            health_score -= 30
            overall_status = "Struggling"
            system_command = "ACTIVATE_WATERING" if moisture_stress == "High" else "ADJUST_CLIMATE"
        elif moisture_stress == "Moderate" or temp_stress == "Moderate":
            health_score -= 15
            overall_status = "Fair"
            
        if disease_confidence > self.thresholds.disease_critical and detected_disease and "Healthy" not in detected_disease:
            health_score -= 40
            overall_status = "Critical"
            system_command = "QUARANTINE"
        elif disease_confidence > self.thresholds.disease_warning and detected_disease and "Healthy" not in detected_disease:
            health_score -= 20
            if overall_status == "Thriving":
                overall_status = "Fair"

        health_score = max(0, min(100, health_score))

        return {
            "stress_diagnosis": {
                "moisture_stress": moisture_stress + " [HEURISTIC]",
                "temperature_stress": temp_stress + " [HEURISTIC]",
                "light_deficit": light_deficit + " [HEURISTIC]",
                "nutrient_status": "Adequate [HEURISTIC]"
            },
            "overall_status": overall_status + " [HEURISTIC]",
            "health_score": f"{health_score} [HEURISTIC]",
            "trajectory_7day": {"3": overall_status, "5": overall_status, "7": overall_status},
            "system_command": system_command + " [HEURISTIC]",
            "is_heuristic": True,
            "recommendations": [
                {"action": "Monitor closely [HEURISTIC]", "urgency": "info", "icon": "ℹ️"}
            ]
        }
