import os
import shutil
#import kagglehub
import subprocess
import tensorflow_datasets as tfds

## Written by JMC for ENGG2112 2024, this script automates the setup of the 
# Demeter datasets for our plant health classification project. It handles 
# downloading from Kaggle, organizing local directories, and provides instructions 
# for manual data integration where necessary.


def download_demeter_datasets():
    print("Initializing Demeter Dataset Integration...\n")
    
    # Define local project directories mapped to our 3 layers
    base_dir = os.path.join(os.getcwd(), 'data')
    dirs = {
        'layer1_id': os.path.join(base_dir, 'layer1_id_species'),
        'layer2_health_rgb': os.path.join(base_dir, 'layer2_health_rgb'),
        'layer2_health_thermal': os.path.join(base_dir, 'layer2_health_thermal'),
        'layer3_environment': os.path.join(base_dir, 'layer3_environment')
    }
    
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    # ==========================================
    # LAYER 1: Species Classification
    # ==========================================
    print("1. Setting up Layer 1 (Species ID)...")
    
    # Pl@ntNet Dataset (Kaggle)
    try:
        print("   Downloading plantnet 300k images from Kaggle...")
        plantnet_path = kagglehub.dataset_download("noahbadoa/plantnet-300k-images")
        shutil.copytree(plantnet_path, os.path.join(dirs['layer1_id'], 'plantnet'), dirs_exist_ok=True)
    except Exception as e:
        print(f"   [!] Error downloading plantnet 300k images: {e}")

    # TensorFlow Plant Leaves
    try:
        print("   Downloading TF Plant Leaves...")
        ds, info = tfds.load('plant_leaves', split='train', with_info=True, data_dir=os.path.join(dirs['layer1_id'], 'tf_plant_leaves'))
        print(f"   Successfully initialized {info.splits['train'].num_examples} TF Plant Leaves images.")
    except Exception as e:
        print(f"   [!] Error downloading TF Plant Leaves: {e}")

    # ==========================================
    # LAYER 2: Health & Stress
    # ==========================================
    print("\n2. Setting up Layer 2 (Health & Thermal)...")
    
    # PlantVillage / Plant Disease Dataset (Kaggle)
    try:
        print("   Downloading PlantDisease dataset...")
        disease_path = kagglehub.dataset_download("emmarex/plantdisease")
        shutil.copytree(disease_path, dirs['layer2_health_rgb'], dirs_exist_ok=True)
    except Exception as e:
        print(f"   [!] Error downloading PlantDisease: {e}")

    # AI4EOSC Thermal
    print(f"   [!] ACTION REQUIRED: AI4EOSC Thermal Dashboard data requires manual authentication. Extract downloaded thermal images to: {dirs['layer2_health_thermal']}")

    # ==========================================
    # LAYER 3: Environment & Growth
    # ==========================================
    print("\n3. Setting up Layer 3 (Environment Metrics)...")
    
    # Plant Growth Data Classification (Kaggle)
    try:
        print("   Downloading Plant Growth Tabular Data...")
        growth_path = kagglehub.dataset_download("gorororororo23/plant-growth-data-classification")
        shutil.copytree(growth_path, dirs['layer3_environment'], dirs_exist_ok=True)
    except Exception as e:
        print(f"   [!] Error downloading Plant Growth Data: {e}")

    # Danforth Center
    print(f"   [!] ACTION REQUIRED: Danforth Center phenotypic data requires portal access. Extract data to: {os.path.join(dirs['layer3_environment'], 'danforth')}")

    print("\n==========================================")
    print("Demeter Data setup complete! Ready for ENGG2112 training pipelines.")
    print("==========================================\n")

def init_demeter_data():
    print("Initializing Demeter Dataset Ingestion...\n")
    
    # Base data directory (one level up from the src folder)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    
    # Define layer directories
    dirs = {
        'layer1_tfds': os.path.join(base_dir, 'layer1_tf_leaves'),
        'layer3_tabular': os.path.join(base_dir, 'layer3_kaggle_tabular'),
        'layer2_thermal': os.path.join(base_dir, 'layer2_manual_thermal'),
        'layer3_danforth': os.path.join(base_dir, 'layer3_manual_danforth')
    }
    
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    # =====================================================================
    # PROGRAMMATIC DOWNLOADS
    # =====================================================================
    print("1. Downloading TensorFlow Plant Leaves (TFDS)...")
    try:
        ds, info = tfds.load('plant_leaves', split='train', data_dir=dirs['layer1_tfds'])
        print("   -> Success.")
    except Exception as e:
        print(f"   -> Failed: {e}")

    print("2. Downloading Plant Growth Tabular Data (via Kaggle CLI)...")
    try:
        # Triggering the native CLI directly, completely bypassing kagglehub
        subprocess.run([
            "kaggle", "datasets", "download", 
            "-d", "gorororororo23/plant-growth-data-classification", 
            "-p", dirs['layer3_tabular'], 
            "--unzip"
        ], check=True, capture_output=True, text=True)
        print("   -> Success.")
    except subprocess.CalledProcessError as e:
        print(f"   -> Failed. Kaggle CLI Error: {e.stderr}")
    except FileNotFoundError:
        print("   -> Failed. Kaggle CLI not found. Ensure 'kaggle' is installed.")

    # =====================================================================
    # MANUAL DOWNLOAD INSTRUCTIONS
    # =====================================================================
    print("\n[ACTION REQUIRED] Please manually download the following datasets:")
    
    print(f"\n-> AI4EOSC Thermal directory ready at: {dirs['layer2_thermal']}")
    print("   Link: https://dashboard.cloud.ai4eosc.eu/catalog/modules/plants-classification")
    
    print(f"\n-> Danforth Center directory ready at: {dirs['layer3_danforth']}")
    print("   Link: https://datasci.danforthcenter.org/data/")
    
    print("\nNote: PlantNet-300K is massive. We will attach it directly in Kaggle rather than downloading locally.")
    
if __name__ == "__main__":
    #download_demeter_datasets()
    init_demeter_data()