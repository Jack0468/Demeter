"""
Model Evaluation Detailed
Performs detailed structural checks and metrics analysis on the trained Demeter models.
"""

import os
import joblib
import tensorflow as tf
from pathlib import Path

class ModelEvaluator:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.models_dir = self.project_root / "models"
        self.results = {
            "model_availability": {},
            "cnn_plantvillage": {"status": "not_loaded"},
            "rf_danforth": {"status": "not_loaded"},
            "data_availability": {"status": "unknown"},
            "inference_engine": {"status": "unknown"},
            "output_formatter": {"status": "unknown"},
            "status_engine": {"status": "unknown"}
        }

    def check_model_availability(self):
        required_models = [
            "demeter_cnn_plantvillage.keras",
            "demeter_rf_danforth.joblib",
            "demeter_cnn_biomass.keras",
            "demeter_cnn_tiller.keras",
            "demeter_cnn.keras",
            "demeter_rf.joblib",
            "health_clusters.joblib",
            "experimentation/hybrid_full_svm.joblib",
            "experimentation/hybrid_svm_species_identifier.joblib"
        ]
        
        for model_name in required_models:
            model_path = self.models_dir / model_name
            available = model_path.is_file()
            self.results["model_availability"][model_name] = {
                "available": available,
                "path": str(model_path)
            }

    def evaluate_cnn_plantvillage(self):
        cnn_path = self.models_dir / "demeter_cnn_plantvillage.keras"
        if not cnn_path.is_file():
            self.results["cnn_plantvillage"] = {
                "status": "missing",
                "error": "Model file not found."
            }
            return
            
        try:
            # Disable GPU usage for clean test environment runs
            tf.config.set_visible_devices([], 'GPU')
            model = tf.keras.models.load_model(str(cnn_path))
            self.results["cnn_plantvillage"] = {
                "status": "loaded",
                "layers": len(model.layers),
                "total_params": model.count_params(),
                "input_shape": model.input_shape,
                "output_shape": model.output_shape
            }
        except Exception as e:
            self.results["cnn_plantvillage"] = {
                "status": "error",
                "error": str(e)
            }

    def evaluate_rf_danforth(self):
        rf_path = self.models_dir / "demeter_rf_danforth.joblib"
        if not rf_path.is_file():
            self.results["rf_danforth"] = {
                "status": "missing",
                "error": "Model file not found."
            }
            return
            
        try:
            model = joblib.load(str(rf_path))
            self.results["rf_danforth"] = {
                "status": "loaded",
                "type": type(model).__name__,
                "n_trees": len(model.estimators_) if hasattr(model, "estimators_") else 0,
                "n_features": model.n_features_in_ if hasattr(model, "n_features_in_") else 0
            }
        except Exception as e:
            self.results["rf_danforth"] = {
                "status": "error",
                "error": str(e)
            }

    def evaluate_hybrid_svms(self):
        svm_full_path = self.models_dir / "experimentation" / "hybrid_full_svm.joblib"
        svm_ident_path = self.models_dir / "experimentation" / "hybrid_svm_species_identifier.joblib"
        
        if not svm_full_path.is_file() or not svm_ident_path.is_file():
            self.results["hybrid_svms"] = {
                "status": "missing",
                "error": "One or more Hybrid SVM model files not found."
            }
            return
            
        try:
            model_full = joblib.load(str(svm_full_path))
            model_ident = joblib.load(str(svm_ident_path))
            self.results["hybrid_svms"] = {
                "status": "loaded",
                "type_full": type(model_full).__name__,
                "type_ident": type(model_ident).__name__,
                "classes_ident": len(model_ident.classes_) if hasattr(model_ident, "classes_") else 0
            }
        except Exception as e:
            self.results["hybrid_svms"] = {
                "status": "error",
                "error": str(e)
            }

    def evaluate_data_availability(self):
        data_dir = self.project_root / "data"
        evaluation_dir = self.project_root / "evaluation_outputs"
        
        has_data = data_dir.is_dir()
        has_eval = evaluation_dir.is_dir()
        
        self.results["data_availability"] = {
            "status": "functional",
            "has_data_dir": has_data,
            "has_evaluation_outputs": has_eval
        }

    def evaluate_inference_engine(self):
        try:
            from src.core import inference_engine
            self.results["inference_engine"] = {
                "status": "available",
                "functions": [
                    name for name in dir(inference_engine)
                    if callable(getattr(inference_engine, name))
                ]
            }
        except Exception as e:
            self.results["inference_engine"] = {
                "status": "error",
                "error": str(e)
            }

    def evaluate_output_formatter(self):
        try:
            from src.core.output_formatter import OutputFormatter
            formatter = OutputFormatter()
            self.results["output_formatter"] = {
                "status": "available",
                "output_dir": formatter.output_dir
            }
        except Exception as e:
            self.results["output_formatter"] = {
                "status": "error",
                "error": str(e)
            }

    def evaluate_status_engine(self):
        try:
            from src.core.status_engine import StatusEngine
            engine = StatusEngine()
            self.results["status_engine"] = {
                "status": "available",
                "thresholds": vars(engine.thresholds)
            }
        except Exception as e:
            self.results["status_engine"] = {
                "status": "error",
                "error": str(e)
            }
