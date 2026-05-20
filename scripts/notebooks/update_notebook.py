import json
import sys

with open('notebooks/interactive_testing.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # 1. Update imports and paths
        if "from src.core.inference_engine import load_models" in source:
            new_source = source.replace(
                "from src.core.inference_engine import load_models, diagnose_plant_disease, predict_growth_milestone, generate_complete_diagnosis",
                "import tensorflow as tf\nimport joblib\nfrom src.core.inference_engine import diagnose_plant_disease, predict_growth_milestone, generate_complete_diagnosis"
            )
            # The try block in the notebook uses exception ModuleNotFoundError.
            new_source = new_source.replace(
                "from inference_engine import load_models",
                "import tensorflow as tf\n    import joblib\n    from inference_engine import diagnose_plant_disease"
            )
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            # Clean up the last newline to match format
            cell['source'][-1] = cell['source'][-1].rstrip('\n')

        # 2. Update model loading cell
        elif "cnn_model, rf_model = load_models" in source:
            new_source = source.replace(
                "cnn_model, rf_model = load_models(plantvillage_cnn_path, danforth_rf_path)",
                "cnn_model = tf.keras.models.load_model(plantvillage_cnn_path)\nrf_model = joblib.load(danforth_rf_path)"
            )
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            cell['source'][-1] = cell['source'][-1].rstrip('\n')
            
        # 3. Update CNN prediction cell to use img_array
        elif "disease_diagnosis = diagnose_plant_disease(image_path" in source:
            new_source = source.replace(
                "disease_diagnosis = diagnose_plant_disease(image_path, cnn_model, class_dirs)",
                "img_tensor = tf.keras.utils.load_img(image_path, target_size=(150, 150))\nimg_array = tf.keras.utils.img_to_array(img_tensor)\nimg_array = tf.expand_dims(img_array, 0)\n\ndisease_diagnosis = diagnose_plant_disease(img_array, image_path, cnn_model, class_dirs)"
            )
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            cell['source'][-1] = cell['source'][-1].rstrip('\n')

with open('notebooks/interactive_testing.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('interactive_testing.ipynb updated successfully!')
