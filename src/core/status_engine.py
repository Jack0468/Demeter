"""
Status Engine: Rule-based health scoring, stress diagnosis, trajectory analysis,
and actionable recommendation generation from model outputs.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class StressThresholds:
    """Configurable thresholds for stress level determination."""
    
    # Moisture stress thresholds (percentage)
    moisture_critical: float = 25.0  # Below this = High stress
    moisture_warning: float = 40.0   # Below this = Medium stress
    moisture_healthy: float = 50.0   # Below this = Low stress
    
    # Temperature stress thresholds (Celsius)
    temp_too_cold: float = 15.0
    temp_too_hot: float = 30.0
    temp_optimal_min: float = 18.0
    temp_optimal_max: float = 26.0
    
    # Sunlight thresholds (hours per day)
    sunlight_minimal: float = 2.0
    sunlight_moderate: float = 4.0
    sunlight_optimal: float = 6.0
    
    # Disease confidence thresholds
    disease_critical: float = 0.75
    disease_warning: float = 0.50
    
    # Growth trajectory thresholds
    growth_decline_critical: float = -20.0  # % decline
    growth_decline_warning: float = -10.0


class StatusEngine:
    """Converts predictions to health status, recommendations, and system commands."""
    
    def __init__(self, thresholds: Optional[StressThresholds] = None):
        """
        Initialize status engine with configurable thresholds.
        
        Args:
            thresholds: Custom StressThresholds object (uses defaults if None)
        """
        self.thresholds = thresholds or StressThresholds()
    
    def diagnose_moisture_stress(self, soil_moisture: float) -> Tuple[str, int]:
        """
        Determine moisture stress level.
        
        Args:
            soil_moisture: Soil moisture percentage (0-100)
            
        Returns:
            Tuple of (stress_level: "High"/"Medium"/"Low", priority: 1-3)
        """
        if soil_moisture < self.thresholds.moisture_critical:
            return ("High", 1)
        elif soil_moisture < self.thresholds.moisture_warning:
            return ("Medium", 2)
        elif soil_moisture < self.thresholds.moisture_healthy:
            return ("Low", 3)
        else:
            return ("None", 0)
    
    def diagnose_temperature_stress(self, temperature: float) -> Tuple[str, int]:
        """
        Determine temperature stress level.
        
        Args:
            temperature: Temperature in Celsius
            
        Returns:
            Tuple of (stress_level: "High"/"Medium"/"Low", priority)
        """
        if temperature < self.thresholds.temp_too_cold:
            return ("High", 1)
        elif temperature < self.thresholds.temp_optimal_min:
            return ("Medium", 2)
        elif temperature > self.thresholds.temp_too_hot:
            return ("High", 1)
        elif temperature > self.thresholds.temp_optimal_max:
            return ("Medium", 2)
        else:
            return ("Low", 3)
    
    def diagnose_light_deficit(self, sunlight_hours: float) -> Tuple[str, int]:
        """
        Determine light deficit level.
        
        Args:
            sunlight_hours: Sunlight hours per day
            
        Returns:
            Tuple of (deficit_level: "High"/"Medium"/"Low", priority)
        """
        if sunlight_hours < self.thresholds.sunlight_minimal:
            return ("High", 1)
        elif sunlight_hours < self.thresholds.sunlight_moderate:
            return ("Medium", 2)
        elif sunlight_hours < self.thresholds.sunlight_optimal:
            return ("Low", 3)
        else:
            return ("None", 0)
    
    def diagnose_nutrient_status(self, humidity: Optional[float] = None) -> str:
        """
        Determine nutrient status (simplified heuristic).
        
        Args:
            humidity: Relative humidity percentage (optional proxy)
            
        Returns:
            Nutrient status string
        """
        # Simplified: Normal unless we have other indicators
        # In production, this would integrate soil NPK sensors
        return "Normal"
    
    def assess_disease_severity(self, disease_confidence: float) -> Tuple[str, int]:
        """
        Map disease confidence to severity level.
        
        Args:
            disease_confidence: Model confidence score (0-1)
            
        Returns:
            Tuple of (severity: "Critical"/"Warning"/"Minor", priority)
        """
        if disease_confidence >= self.thresholds.disease_critical:
            return ("Critical", 1)
        elif disease_confidence >= self.thresholds.disease_warning:
            return ("Warning", 2)
        else:
            return ("Minor", 3)
    
    def calculate_composite_health_score(
        self,
        disease_confidence: float,
        soil_moisture: float,
        temperature: float,
        sunlight_hours: float
    ) -> Tuple[int, str]:
        """
        Calculate composite health score (0-100) and overall status.
        
        Args:
            disease_confidence: Model disease confidence (0-1)
            soil_moisture: Soil moisture percentage (0-100)
            temperature: Temperature in Celsius
            sunlight_hours: Sunlight hours per day
            
        Returns:
            Tuple of (health_score: 0-100, status: "Thriving"/"Struggling"/"Critical")
        """
        score = 100.0
        
        # Disease impact (-40 points max)
        if disease_confidence > self.thresholds.disease_critical:
            score -= 40
        elif disease_confidence > self.thresholds.disease_warning:
            score -= 20
        else:
            score -= disease_confidence * 10
        
        # Moisture impact (-20 points max)
        moisture_status, _ = self.diagnose_moisture_stress(soil_moisture)
        if moisture_status == "High":
            score -= 20
        elif moisture_status == "Medium":
            score -= 10
        
        # Temperature impact (-20 points max)
        temp_status, _ = self.diagnose_temperature_stress(temperature)
        if temp_status == "High":
            score -= 20
        elif temp_status == "Medium":
            score -= 10
        
        # Sunlight impact (-10 points max)
        light_status, _ = self.diagnose_light_deficit(sunlight_hours)
        if light_status == "High":
            score -= 10
        elif light_status == "Medium":
            score -= 5
        
        # Clamp score to 0-100
        score = max(0, min(100, score))
        
        # Determine status
        if score >= 70:
            status = "Thriving"
        elif score >= 40:
            status = "Struggling"
        else:
            status = "Critical"
        
        return (int(score), status)
    
    def generate_recommendations(
        self,
        disease_confidence: float,
        disease_name: str,
        soil_moisture: float,
        temperature: float,
        sunlight_hours: float,
        humidity: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate prioritized action recommendations.
        
        Args:
            disease_confidence: Model disease confidence (0-1)
            disease_name: Detected disease name
            soil_moisture: Soil moisture percentage
            temperature: Temperature in Celsius
            sunlight_hours: Sunlight hours per day
            humidity: Relative humidity percentage (optional)
            
        Returns:
            List of recommendation dicts with priority, action, icon
        """
        recommendations = []
        
        # Disease-based recommendations
        if disease_confidence > self.thresholds.disease_warning:
            if "blight" in disease_name.lower():
                recommendations.append({
                    "priority": 1,
                    "action": "Remove affected leaves and apply fungicide",
                    "icon": "🍂",
                    "urgency": "critical"
                })
                recommendations.append({
                    "priority": 2,
                    "action": "Improve air circulation around plant",
                    "icon": "💨",
                    "urgency": "high"
                })
            elif "rust" in disease_name.lower():
                recommendations.append({
                    "priority": 1,
                    "action": "Apply sulfur-based fungicide",
                    "icon": "🧪",
                    "urgency": "critical"
                })
            elif "mosaic" in disease_name.lower():
                recommendations.append({
                    "priority": 1,
                    "action": "Isolate plant to prevent spread",
                    "icon": "⚠️",
                    "urgency": "critical"
                })
            else:
                recommendations.append({
                    "priority": 1,
                    "action": "Apply appropriate disease treatment",
                    "icon": "💊",
                    "urgency": "high"
                })
        
        # Moisture-based recommendations
        moisture_status, moisture_priority = self.diagnose_moisture_stress(soil_moisture)
        if moisture_status == "High":
            recommendations.append({
                "priority": 1,
                "action": "Increase watering frequency immediately",
                "icon": "💧",
                "urgency": "critical"
            })
        elif moisture_status == "Medium":
            recommendations.append({
                "priority": 2,
                "action": "Water more frequently",
                "icon": "💧",
                "urgency": "high"
            })
        
        # Temperature-based recommendations
        temp_status, _ = self.diagnose_temperature_stress(temperature)
        if temperature < self.thresholds.temp_too_cold:
            recommendations.append({
                "priority": 2,
                "action": "Move to warmer location or provide heating",
                "icon": "🔥",
                "urgency": "high"
            })
        elif temperature > self.thresholds.temp_too_hot:
            recommendations.append({
                "priority": 2,
                "action": "Provide shade and increase air circulation",
                "icon": "☀️",
                "urgency": "high"
            })
        
        # Light-based recommendations
        light_status, _ = self.diagnose_light_deficit(sunlight_hours)
        if light_status == "High":
            recommendations.append({
                "priority": 3,
                "action": "Relocate to area with more sunlight",
                "icon": "☀️",
                "urgency": "medium"
            })
        
        # Sort by priority and limit to top 3
        recommendations = sorted(recommendations, key=lambda x: x["priority"])[:3]
        
        return recommendations
    
    def determine_system_command(
        self,
        health_score: int,
        disease_confidence: float,
        soil_moisture: float
    ) -> str:
        """
        Determine automated system command based on health metrics.
        
        Args:
            health_score: Composite health score (0-100)
            disease_confidence: Disease confidence (0-1)
            soil_moisture: Soil moisture percentage
            
        Returns:
            System command string (e.g., "ACTIVATE_WATER_PUMP")
        """
        commands = []
        
        # Water pump activation
        if soil_moisture < self.thresholds.moisture_warning:
            commands.append("ACTIVATE_WATER_PUMP")
        
        # Pest/disease mitigation (spray system)
        if disease_confidence > self.thresholds.disease_critical:
            commands.append("ACTIVATE_SPRAY_SYSTEM")
        
        # Alert notification
        if health_score < 40:
            commands.append("SEND_CRITICAL_ALERT")
        elif health_score < 60:
            commands.append("SEND_WARNING_ALERT")
        
        # Combine commands
        if not commands:
            return "MONITOR_ONLY"
        
        return " | ".join(commands)
    
    def predict_7day_trajectory(
        self,
        current_health_score: int,
        growth_prediction: float,
        soil_moisture: float
    ) -> Dict[int, str]:
        """
        Generate 7-day health trajectory predictions.
        
        Args:
            current_health_score: Current health score (0-100)
            growth_prediction: RF model growth prediction
            soil_moisture: Current soil moisture
            
        Returns:
            Dict mapping {day: status} for days 1,3,5,7
        """
        trajectory = {}
        
        # Simulate trajectory decline if moisture is low and not intervened
        score = current_health_score
        
        for day in [1, 3, 5, 7]:
            if soil_moisture < self.thresholds.moisture_critical:
                score = max(0, score - (2 * day))  # Aggressive decline
            elif soil_moisture < self.thresholds.moisture_warning:
                score = max(0, score - day)  # Moderate decline
            else:
                score = min(100, score + (day * 0.5))  # Slight improvement
            
            # Map score to status
            if score >= 70:
                trajectory[day] = "Thriving"
            elif score >= 40:
                trajectory[day] = "Fair"
            else:
                trajectory[day] = "Poor"
        
        return trajectory
    
    def generate_full_diagnosis(
        self,
        disease_confidence: float,
        disease_name: str,
        soil_moisture: float,
        temperature: float,
        sunlight_hours: float,
        predicted_growth: float,
        humidity: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate complete diagnostic output with all computed fields.
        
        Args:
            disease_confidence: Model disease confidence (0-1)
            disease_name: Detected disease name
            soil_moisture: Soil moisture percentage
            temperature: Temperature in Celsius
            sunlight_hours: Sunlight hours per day
            predicted_growth: RF predicted growth value
            humidity: Relative humidity percentage (optional)
            
        Returns:
            Complete diagnosis dict ready for dashboard
        """
        # Calculate stress levels
        moisture_stress, _ = self.diagnose_moisture_stress(soil_moisture)
        temp_stress, _ = self.diagnose_temperature_stress(temperature)
        light_deficit, _ = self.diagnose_light_deficit(sunlight_hours)
        nutrient_status = self.diagnose_nutrient_status(humidity)
        disease_severity, _ = self.assess_disease_severity(disease_confidence)
        
        # Calculate health score
        health_score, overall_status = self.calculate_composite_health_score(
            disease_confidence, soil_moisture, temperature, sunlight_hours
        )
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            disease_confidence, disease_name, soil_moisture,
            temperature, sunlight_hours, humidity
        )
        
        # Determine system command
        system_command = self.determine_system_command(
            health_score, disease_confidence, soil_moisture
        )
        
        # Predict trajectory
        trajectory = self.predict_7day_trajectory(
            health_score, predicted_growth, soil_moisture
        )
        
        return {
            "stress_diagnosis": {
                "moisture_stress": moisture_stress,
                "temperature_stress": temp_stress,
                "light_deficit": light_deficit,
                "nutrient_status": nutrient_status,
                "disease_severity": disease_severity
            },
            "health_score": health_score,
            "overall_status": overall_status,
            "recommendations": recommendations,
            "system_command": system_command,
            "trajectory_7day": trajectory
        }
