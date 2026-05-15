"""
DEPRECATED: model_builder.py has been split by technology domain for better 
maintenance, dependency management, and readability.

Please update your imports to use the new files:
- For CNN/TensorFlow models: import from `src.training.vision_models`
- For Scikit-Learn/Tabular models: import from `src.training.tabular_models`

This file is maintained temporarily as a proxy for legacy scripts.
"""
import warnings

warnings.warn(
    "model_builder.py is deprecated. Please import from vision_models.py and tabular_models.py",
    DeprecationWarning,
    stacklevel=2
)

# Proxy exports for legacy support
from .vision_models import (
    train_and_save_cnn,
    train_and_save_cnn_plantvillage,
    train_tiller_cnn_regressor,
    train_custom_baseline_cnn,
    get_augmenter
)

from .tabular_models import (
    train_and_save_rf,
    train_and_save_rf_danforth
)