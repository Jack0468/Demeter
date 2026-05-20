"""
Status Engine for Demeter.
Determines health score, overall status, and stress levels from raw sensor/model data.
"""

class StressThresholds:
    def __init__(self, moisture_critical=25.0, moisture_warning=40.0, moisture_healthy=50.0,
                 temp_too_cold=10.0, temp_too_hot=30.0, temp_optimal_min=18.0, temp_optimal_max=28.0,
                 sunlight_minimal=2.0, sunlight_moderate=4.0, sunlight_optimal=6.0,
                 disease_critical=0.75, disease_warning=0.5):
        self.moisture_critical = moisture_critical
        self.moisture_warning = moisture_warning
        self.moisture_healthy = moisture_healthy
        self.temp_too_cold = temp_too_cold
        self.temp_too_hot = temp_too_hot
        self.temp_optimal_min = temp_optimal_min
        self.temp_optimal_max = temp_optimal_max
        self.sunlight_minimal = sunlight_minimal
        self.sunlight_moderate = sunlight_moderate
        self.sunlight_optimal = sunlight_optimal
        self.disease_critical = disease_critical
        self.disease_warning = disease_warning

class StatusEngine:
    def __init__(self, thresholds=None):
        self.thresholds = thresholds or StressThresholds()

    def diagnose_moisture_stress(self, soil_moisture):
        if soil_moisture < self.thresholds.moisture_critical:
            return "High", 1
        elif soil_moisture < self.thresholds.moisture_warning:
            return "Medium", 2
        elif soil_moisture < self.thresholds.moisture_healthy:
            return "Low", 3
        else:
            return "None", 0

    def diagnose_temperature_stress(self, temperature):
        if temperature <= self.thresholds.temp_too_cold or temperature >= self.thresholds.temp_too_hot:
            return "High", 1
        elif temperature < self.thresholds.temp_optimal_min or temperature > self.thresholds.temp_optimal_max:
            return "Medium", 2
        else:
            return "Low", 3

    def diagnose_light_deficit(self, sunlight_hours):
        if sunlight_hours < self.thresholds.sunlight_minimal:
            return "High", 1
        elif sunlight_hours < self.thresholds.sunlight_moderate:
            return "Medium", 2
        elif sunlight_hours < self.thresholds.sunlight_optimal:
            return "Low", 3
        else:
            return "None", 0

    def assess_disease_severity(self, disease_confidence):
        if disease_confidence >= self.thresholds.disease_critical:
            return "Critical", 1
        elif disease_confidence >= self.thresholds.disease_warning:
            return "Warning", 2
        else:
            return "Minor", 3

    def calculate_composite_health_score(self, disease_confidence, soil_moisture, temperature, sunlight_hours):
        score = 100
        
        # Moisture stress penalty
        m_level, _ = self.diagnose_moisture_stress(soil_moisture)
        if m_level == "High":
            score -= 30
        elif m_level == "Medium":
            score -= 15
            
        # Temperature stress penalty
        t_level, _ = self.diagnose_temperature_stress(temperature)
        if t_level == "High":
            score -= 30
        elif t_level == "Medium":
            score -= 15
            
        # Light deficit penalty
        l_level, _ = self.diagnose_light_deficit(sunlight_hours)
        if l_level == "High":
            score -= 20
        elif l_level == "Medium":
            score -= 10
            
        # Disease penalty
        d_level, _ = self.assess_disease_severity(disease_confidence)
        if d_level == "Critical":
            score -= 40
        elif d_level == "Warning":
            score -= 20
            
        score = max(0, min(100, score))
        
        if score >= 70:
            status = "Thriving"
        elif score >= 40:
            status = "Struggling"
        else:
            status = "Critical"
            
        return score, status

    def generate_recommendations(self, disease_confidence, disease_name, soil_moisture, temperature, sunlight_hours, humidity=50.0):
        recs = []
        d_level, _ = self.assess_disease_severity(disease_confidence)
        if d_level in ["Critical", "Warning"] and disease_name and "Healthy" not in disease_name:
            recs.append({
                "action": f"Apply appropriate fungicide to control {disease_name.replace('_', ' ')}",
                "urgency": "critical" if d_level == "Critical" else "warning",
                "icon": "🧪"
            })
        
        m_level, _ = self.diagnose_moisture_stress(soil_moisture)
        if m_level == "High":
            recs.append({
                "action": "Water the plant immediately to relieve moisture stress",
                "urgency": "critical",
                "icon": "💧"
            })
        elif m_level == "Medium":
            recs.append({
                "action": "Water the plant soon to prevent worsening moisture stress",
                "urgency": "warning",
                "icon": "💧"
            })
            
        t_level, _ = self.diagnose_temperature_stress(temperature)
        if t_level == "High":
            recs.append({
                "action": "Provide shade or cooling to protect plant from severe heat",
                "urgency": "critical" if temperature > 30 else "warning",
                "icon": "⛱️"
            })
            
        l_level, _ = self.diagnose_light_deficit(sunlight_hours)
        if l_level in ["High", "Medium"]:
            recs.append({
                "action": "Increase sunlight hours exposure or relocate plant",
                "urgency": "warning",
                "icon": "☀️"
            })
            
        if not recs:
            recs.append({
                "action": "Maintain current watering and monitoring schedule",
                "urgency": "info",
                "icon": "ℹ️"
            })
            
        return recs[:3]

    def determine_system_command(self, health_score, disease_confidence, soil_moisture):
        commands = []
        if soil_moisture < self.thresholds.moisture_critical:
            commands.append("ACTIVATE_WATER_PUMP")
        if disease_confidence >= self.thresholds.disease_critical:
            commands.append("ACTIVATE_SPRAY_SYSTEM")
        if health_score < 40:
            commands.append("SEND_CRITICAL_ALERT")
            
        if not commands:
            return "MONITOR_ONLY"
        return "|".join(commands)

    def predict_7day_trajectory(self, current_health_score, growth_prediction, soil_moisture):
        trajectory = {}
        if soil_moisture < self.thresholds.moisture_critical:
            trajectory[1] = "Fair"
            trajectory[3] = "Fair"
            trajectory[5] = "Poor"
            trajectory[7] = "Poor"
        elif soil_moisture < self.thresholds.moisture_warning:
            trajectory[1] = "Fair"
            trajectory[3] = "Fair"
            trajectory[5] = "Fair"
            trajectory[7] = "Fair"
        else:
            trajectory[1] = "Thriving"
            trajectory[3] = "Thriving"
            trajectory[5] = "Thriving"
            trajectory[7] = "Thriving"
        return trajectory

    def generate_full_diagnosis(self, disease_confidence, detected_disease=None, soil_moisture=50.0,
                               temperature=25.0, sunlight_hours=6.0, predicted_growth=50.0,
                               humidity=50.0, disease_name=None):
        name = disease_name or detected_disease or "Healthy"
        
        # 1. Stress levels
        moisture_stress, _ = self.diagnose_moisture_stress(soil_moisture)
        temp_stress, _ = self.diagnose_temperature_stress(temperature)
        light_deficit, _ = self.diagnose_light_deficit(sunlight_hours)
        
        # 2. Health score
        health_score, overall_status = self.calculate_composite_health_score(
            disease_confidence, soil_moisture, temperature, sunlight_hours
        )
        
        # 3. Recommendations
        recommendations = self.generate_recommendations(
            disease_confidence, name, soil_moisture, temperature, sunlight_hours, humidity
        )
        
        # 4. System command
        system_command = self.determine_system_command(
            health_score, disease_confidence, soil_moisture
        )
        
        # 5. Trajectory
        trajectory = self.predict_7day_trajectory(
            health_score, predicted_growth, soil_moisture
        )
        
        return {
            "stress_diagnosis": {
                "moisture_stress": moisture_stress,
                "temperature_stress": temp_stress,
                "light_deficit": light_deficit,
                "nutrient_status": "Adequate"
            },
            "overall_status": overall_status,
            "health_score": health_score,
            "trajectory_7day": trajectory,
            "system_command": system_command,
            "is_heuristic": False,
            "recommendations": recommendations
        }
