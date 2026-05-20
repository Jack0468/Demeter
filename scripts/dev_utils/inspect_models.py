import pickle
try:
    with open('data/metadata_cache.pkl', 'rb') as f:
        d = pickle.load(f)
    print("Metadata keys:", list(d.keys()))
    if 'class_names' in d: print("class_names:", d['class_names'])
except Exception as e:
    print("Error reading metadata:", e)

import sys
try:
    sys.path.append('src')
    from tensorflow.keras.models import load_model
    m = load_model('models/demeter_cnn_plantvillage.keras')
    print("PlantVillage Output shape:", m.output_shape)
    m2 = load_model('models/demeter_cnn.keras')
    print("Bellwether Output shape:", m2.output_shape)
except Exception as e:
    print("Error reading models:", e)
