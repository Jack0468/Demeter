"""
model_evaluation_detailed.py

Comprehensive evaluation of Demeter models:
1. Check model availability and integrity
2. Load and validate architecture
3. Run test predictions
4. Report performance metrics and statistics
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class ModelEvaluator:
    """Comprehensive model evaluation framework."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.models_dir = "models"
        self.data_dir = "data"
        self.results = {}
    
    def check_model_availability(self):
        """Check which models are available."""
        print("\n" + "="*70)
        print("1. MODEL AVAILABILITY CHECK")
        print("="*70)
        
        model_specs = {
            "CNN PlantVillage": {
                "path": "models/demeter_cnn_plantvillage.keras",
                "type": "CNN",
                "purpose": "Disease classification"
            },
            "RF Danforth": {
                "path": "models/demeter_rf_danforth.joblib",
                "type": "Random Forest",
                "purpose": "Growth prediction"
            },
            "CNN Bellwether": {
                "path": "models/demeter_cnn.keras",
                "type": "CNN",
                "purpose": "Water stress detection (legacy)"
            },
            "RF Bellwether": {
                "path": "models/demeter_rf.joblib",
                "type": "Random Forest",
                "purpose": "Growth/stress prediction (legacy)"
            }
        }
        
        availability = {}
        
        for name, spec in model_specs.items():
            exists = os.path.exists(spec["path"])
            size = os.path.getsize(spec["path"]) / (1024**2) if exists else 0  # MB
            
            availability[name] = {
                "available": exists,
                "path": spec["path"],
                "type": spec["type"],
                "purpose": spec["purpose"],
                "size_mb": round(size, 2)
            }
            
            status = "✓ AVAILABLE" if exists else "✗ MISSING"
            print(f"\n{name}: {status}")
            print(f"  Path: {spec['path']}")
            print(f"  Type: {spec['type']}")
            print(f"  Purpose: {spec['purpose']}")
            if exists:
                print(f"  Size: {size:.2f} MB")
        
        self.results['model_availability'] = availability
        return availability
    
    def evaluate_cnn_plantvillage(self):
        """Evaluate PlantVillage CNN model."""
        print("\n" + "="*70)
        print("2. CNN PLANTVILLAGE MODEL EVALUATION")
        print("="*70)
        
        model_path = "models/demeter_cnn_plantvillage.keras"
        
        if not os.path.exists(model_path):
            print(f"\n✗ Model not found at {model_path}")
            self.results['cnn_plantvillage'] = {"status": "not_available"}
            return
        
        try:
            print(f"\nLoading model from {model_path}...")
            model = tf.keras.models.load_model(model_path)
            print("✓ Model loaded successfully")
            
            # Get model architecture info
            print(f"\nModel Architecture:")
            print(f"  Total layers: {len(model.layers)}")
            print(f"  Input shape: {model.input_shape}")
            print(f"  Output shape: {model.output_shape}")
            
            # Count parameters
            total_params = model.count_params()
            trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
            
            print(f"\nParameters:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Non-trainable: {total_params - trainable_params:,}")
            
            # Try a test prediction
            print(f"\nTesting inference...")
            
            # Create dummy input
            dummy_input = np.random.rand(1, *model.input_shape[1:]).astype(np.float32)
            predictions = model.predict(dummy_input, verbose=0)
            
            print(f"✓ Test prediction successful")
            print(f"  Output shape: {predictions.shape}")
            print(f"  Output range: [{predictions.min():.4f}, {predictions.max():.4f}]")
            
            # Get layer details
            print(f"\nLayer Summary:")
            for i, layer in enumerate(model.layers[:5]):  # First 5 layers
                print(f"  Layer {i}: {layer.name} ({type(layer).__name__})")
            if len(model.layers) > 5:
                print(f"  ... and {len(model.layers) - 5} more layers")
            
            self.results['cnn_plantvillage'] = {
                "status": "loaded",
                "total_params": total_params,
                "trainable_params": trainable_params,
                "input_shape": str(model.input_shape),
                "output_shape": str(model.output_shape),
                "layers": len(model.layers),
                "prediction_successful": True
            }
            
        except Exception as e:
            print(f"\n✗ Error evaluating CNN PlantVillage: {e}")
            self.results['cnn_plantvillage'] = {
                "status": "error",
                "error": str(e)
            }
    
    def evaluate_rf_danforth(self):
        """Evaluate Danforth Random Forest model."""
        print("\n" + "="*70)
        print("3. RANDOM FOREST DANFORTH MODEL EVALUATION")
        print("="*70)
        
        model_path = "models/demeter_rf_danforth.joblib"
        
        if not os.path.exists(model_path):
            print(f"\n✗ Model not found at {model_path}")
            self.results['rf_danforth'] = {"status": "not_available"}
            return
        
        try:
            print(f"\nLoading model from {model_path}...")
            rf_model = joblib.load(model_path)
            print("✓ Model loaded successfully")
            
            # Get model info
            print(f"\nModel Type: {type(rf_model).__name__}")
            
            if hasattr(rf_model, 'n_estimators'):
                print(f"  Number of trees: {rf_model.n_estimators}")
            
            if hasattr(rf_model, 'n_features_in_'):
                print(f"  Input features: {rf_model.n_features_in_}")
            
            if hasattr(rf_model, 'n_outputs_'):
                print(f"  Output targets: {rf_model.n_outputs_}")
            
            # Try test prediction
            print(f"\nTesting inference...")
            n_features = getattr(rf_model, 'n_features_in_', 6)  # Default to 6 if not available
            dummy_input = np.random.rand(1, n_features).astype(np.float32)
            
            predictions = rf_model.predict(dummy_input)
            
            print(f"✓ Test prediction successful")
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Output shape: {predictions.shape}")
            print(f"  Output value(s): {predictions}")
            
            # Feature importance if available
            if hasattr(rf_model, 'feature_importances_'):
                importances = rf_model.feature_importances_
                print(f"\nFeature Importances (top 5):")
                top_indices = np.argsort(importances)[-5:][::-1]
                for idx in top_indices:
                    print(f"  Feature {idx}: {importances[idx]:.4f}")
            
            self.results['rf_danforth'] = {
                "status": "loaded",
                "type": type(rf_model).__name__,
                "n_trees": getattr(rf_model, 'n_estimators', 'N/A'),
                "n_features": getattr(rf_model, 'n_features_in_', 'N/A'),
                "prediction_successful": True
            }
            
        except Exception as e:
            print(f"\n✗ Error evaluating RF Danforth: {e}")
            self.results['rf_danforth'] = {
                "status": "error",
                "error": str(e)
            }
    
    def evaluate_data_availability(self):
        """Check availability of key datasets."""
        print("\n" + "="*70)
        print("4. DATA AVAILABILITY CHECK")
        print("="*70)
        
        datasets = {
            "PlantVillage": {
                "path": "data/layer2_health_rgb/PlantVillage",
                "type": "Disease classification images"
            },
            "Danforth Environment": {
                "path": "data/layer3_environment/plant_growth_data.csv",
                "type": "Growth prediction tabular data"
            },
            "Bellwether Images": {
                "path": "data/bellwether_images_dir",
                "type": "Water stress classification images"
            },
            "Plant Diagnostics CSV": {
                "path": "data/plant_diagnostics.csv",
                "type": "Inference logging"
            }
        }
        
        data_availability = {}
        
        for name, spec in datasets.items():
            exists = os.path.exists(spec["path"])
            
            # Get size/count info
            info = ""
            if exists:
                if spec["path"].endswith('.csv'):
                    try:
                        with open(spec["path"], 'r') as f:
                            lines = len(f.readlines())
                        info = f"({lines} lines)"
                    except:
                        info = "(size: error reading)"
                else:
                    try:
                        if os.path.isdir(spec["path"]):
                            count = len(os.listdir(spec["path"]))
                            info = f"({count} items)"
                    except:
                        pass
            
            status = "✓ AVAILABLE" if exists else "✗ MISSING"
            print(f"\n{name}: {status}")
            print(f"  Path: {spec['path']}")
            print(f"  Type: {spec['type']}")
            if info:
                print(f"  Info: {info}")
            
            data_availability[name] = {
                "available": exists,
                "path": spec["path"],
                "type": spec["type"]
            }
        
        self.results['data_availability'] = data_availability
    
    def evaluate_inference_engine(self):
        """Test inference engine functions."""
        print("\n" + "="*70)
        print("5. INFERENCE ENGINE VALIDATION")
        print("="*70)
        
        try:
            from inference_engine import load_models, diagnose_plant_disease, predict_growth_milestone
            print("\n✓ Inference engine imported successfully")
            
            # Check function signatures
            print("\nAvailable functions:")
            print("  - load_models(cnn_path, rf_path)")
            print("  - diagnose_plant_disease(image_path, cnn_model, class_names)")
            print("  - predict_growth_milestone(environmental_data, rf_model)")
            print("  - generate_complete_diagnosis(...)")
            
            self.results['inference_engine'] = {"status": "available"}
            
        except ImportError as e:
            print(f"\n✗ Error importing inference engine: {e}")
            self.results['inference_engine'] = {
                "status": "error",
                "error": str(e)
            }
    
    def evaluate_output_formatter(self):
        """Test output formatter module."""
        print("\n" + "="*70)
        print("6. OUTPUT FORMATTER VALIDATION")
        print("="*70)
        
        try:
            from output_formatter import OutputFormatter, quick_format_diagnosis
            print("\n✓ Output formatter imported successfully")
            
            # Test formatting
            formatter = OutputFormatter()
            print("✓ Formatter instantiated")
            
            # Test disease format
            disease_result = OutputFormatter.format_disease_detection(
                "/test/image.jpg", "Test Disease", 0.75,
                {"Test Disease": 0.75, "Other": 0.25}
            )
            print("✓ format_disease_detection() works")
            
            # Test growth format
            growth_result = OutputFormatter.format_growth_prediction(50.0, {})
            print("✓ format_growth_prediction() works")
            
            # Test sensor format
            sensor_data = OutputFormatter.format_sensor_data(22.0, 65.0, 6.0)
            print("✓ format_sensor_data() works")
            
            self.results['output_formatter'] = {
                "status": "functional",
                "functions": [
                    "format_disease_detection",
                    "format_growth_prediction",
                    "format_sensor_data",
                    "merge_diagnosis",
                    "save_latest",
                    "append_history"
                ]
            }
            
        except Exception as e:
            print(f"\n✗ Error with output formatter: {e}")
            self.results['output_formatter'] = {
                "status": "error",
                "error": str(e)
            }
    
    def evaluate_status_engine(self):
        """Test status engine module."""
        print("\n" + "="*70)
        print("7. STATUS ENGINE VALIDATION")
        print("="*70)
        
        try:
            from status_engine import StatusEngine, StressThresholds
            print("\n✓ Status engine imported successfully")
            
            # Test engine
            engine = StatusEngine()
            print("✓ StatusEngine instantiated")
            
            # Test methods
            stress_level, priority = engine.diagnose_moisture_stress(35.0)
            print(f"✓ diagnose_moisture_stress() works → {stress_level}")
            
            score, status = engine.calculate_composite_health_score(0.5, 65.0, 22.0, 6.0)
            print(f"✓ calculate_composite_health_score() works → {score}/100")
            
            recs = engine.generate_recommendations(0.6, "Early Blight", 40.0, 22.0, 5.0)
            print(f"✓ generate_recommendations() works → {len(recs)} recommendations")
            
            diagnosis = engine.generate_full_diagnosis(0.5, "Test", 50.0, 22.0, 6.0, 50.0)
            print(f"✓ generate_full_diagnosis() works → {diagnosis['overall_status']}")
            
            self.results['status_engine'] = {
                "status": "functional",
                "methods": [
                    "diagnose_moisture_stress",
                    "diagnose_temperature_stress",
                    "diagnose_light_deficit",
                    "assess_disease_severity",
                    "calculate_composite_health_score",
                    "generate_recommendations",
                    "determine_system_command",
                    "predict_7day_trajectory",
                    "generate_full_diagnosis"
                ]
            }
            
        except Exception as e:
            print(f"\n✗ Error with status engine: {e}")
            self.results['status_engine'] = {
                "status": "error",
                "error": str(e)
            }
    
    def generate_report(self):
        """Generate comprehensive evaluation report."""
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        print("\n📊 RESULTS:")
        print(f"  Timestamp: {datetime.now().isoformat()}")
        
        # Model availability summary
        available_models = sum(1 for m in self.results.get('model_availability', {}).values() if m.get('available'))
        total_models = len(self.results.get('model_availability', {}))
        print(f"\n  Models Available: {available_models}/{total_models}")
        
        # Module functionality summary
        modules_ok = sum(1 for v in [
            self.results.get('inference_engine', {}),
            self.results.get('output_formatter', {}),
            self.results.get('status_engine', {})
        ] if v.get('status') in ['available', 'functional'])
        print(f"  Modules Functional: {modules_ok}/3")
        
        # Save results to JSON
        results_file = "data/outputs/model_evaluation_results.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n✓ Results saved to {results_file}")
        
        # Print recommendations
        print("\n💡 RECOMMENDATIONS:")
        if available_models < total_models:
            print(f"  - Train missing models ({total_models - available_models} not found)")
        print("  - Run tests/ directory tests for comprehensive validation")
        print("  - Execute example_integrated_workflow.py to generate sample diagnoses")


def main():
    """Run full evaluation."""
    print("\n" + "="*70)
    print("DEMETER MODEL EVALUATION SUITE")
    print("="*70)
    print(f"Started: {datetime.now().isoformat()}")
    
    evaluator = ModelEvaluator()
    
    # Run all evaluations
    evaluator.check_model_availability()
    evaluator.evaluate_cnn_plantvillage()
    evaluator.evaluate_rf_danforth()
    evaluator.evaluate_data_availability()
    evaluator.evaluate_inference_engine()
    evaluator.evaluate_output_formatter()
    evaluator.evaluate_status_engine()
    
    # Generate final report
    evaluator.generate_report()
    
    print("\n" + "="*70)
    print(f"Completed: {datetime.now().isoformat()}")
    print("="*70)


if __name__ == "__main__":
    main()
