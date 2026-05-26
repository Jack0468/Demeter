import json
import os
from pathlib import Path

_current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _current_dir.parent.parent if _current_dir.parent.name == "src" else _current_dir.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "data_split_manifest.json"

def update_manifest(model_name, train_keys, test_keys):
    """
    Records which data samples were used for training and testing a base model.
    This guarantees that downstream evaluation models (like KMeans) can be strictly
    evaluated ONLY on the test set to avoid data leakage.
    
    Args:
        model_name: Identifier for the model (e.g. 'demeter_cnn_plantvillage')
        train_keys: List of identifiers (filepaths or indices) used for training
        test_keys: List of identifiers used for testing
    """
    os.makedirs(MANIFEST_PATH.parent, exist_ok=True)
    
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r") as f:
            try:
                manifest = json.load(f)
            except:
                manifest = {}
    else:
        manifest = {}
        
    # Convert numpy arrays to lists if necessary
    manifest[model_name] = {
        "train": [str(k) for k in train_keys],
        "test": [str(k) for k in test_keys]
    }
    
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=4)
    print(f"Updated data split manifest for {model_name}")
