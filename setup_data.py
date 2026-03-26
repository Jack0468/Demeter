import os
import shutil
import kagglehub
import tensorflow_datasets as tfds

def download_and_organize_data():
    print("Starting Demeter Data Initialization...\n")
    
    # Define local project directories
    base_data_dir = os.path.join(os.getcwd(), 'data')
    image_target_dir = os.path.join(base_data_dir, 'raw_images')
    tabular_target_dir = os.path.join(base_data_dir, 'tabular')
    
    # Create the folders if they don't exist
    os.makedirs(image_target_dir, exist_ok=True)
    os.makedirs(tabular_target_dir, exist_ok=True)

    # ==========================================
    # 1. DOWNLOAD IMAGE DATASET (CNN)
    # ==========================================
    print("Downloading Plant Types Image Dataset via Kaggle...")
    try:
        # Downloads to system cache and returns the path
        image_cache_path = kagglehub.dataset_download("yudhaislamisulistya/plants-type-datasets")
        print(f"Downloaded to cache: {image_cache_path}")
        
        # Copy from cache to our local project folder
        print("Moving images to local data/raw_images folder...")
        shutil.copytree(image_cache_path, image_target_dir, dirs_exist_ok=True)
        print("Image dataset ready.\n")
    except Exception as e:
        print(f"Failed to download image dataset: {e}\n")

    # ==========================================
    # 2. DOWNLOAD TABULAR DATASET (Random Forest)
    # ==========================================
    print("Downloading Plant Growth Tabular Dataset via Kaggle...")
    try:
        tabular_cache_path = kagglehub.dataset_download("gorororororo23/plant-growth-data-classification")
        print(f"Downloaded to cache: {tabular_cache_path}")
        
        # Copy from cache to our local project folder
        print("Moving CSV to local data/tabular folder...")
        shutil.copytree(tabular_cache_path, tabular_target_dir, dirs_exist_ok=True)
        print("Tabular dataset ready.\n")
    except Exception as e:
        print(f"Failed to download tabular dataset: {e}\n")

    print("==========================================")
    print("Data setup complete! Your folder structure is ready for training.")
    print(f"Images located at: {image_target_dir}")
    print(f"Tabular data located at: {tabular_target_dir}")
    print("==========================================")

def download_advanced_datasets():
    print("Initializing Demeter Advanced Data Setup...\n")
    
    base_dir = os.path.join(os.getcwd(), 'data')
    dirs = {
        'rgb_species': os.path.join(base_dir, 'rgb_species'),
        'rgb_health': os.path.join(base_dir, 'rgb_health'),
        'thermal_stress': os.path.join(base_dir, 'thermal_stress')
    }
    
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    # 1. Base Species Dataset (Kaggle)
    print("1. Downloading RGB Species Dataset...")
    try:
        species_path = kagglehub.dataset_download("yudhaislamisulistya/plants-type-datasets")
        shutil.copytree(species_path, dirs['rgb_species'], dirs_exist_ok=True)
    except Exception as e:
        print(f"Error: {e}")

    # 2. Plant Health/Disease Dataset (TensorFlow Datasets)
    print("\n2. Downloading Plant Health Dataset (TFDS)...")
    try:
        # plant_village contains images of healthy and unhealthy leaves
        ds, info = tfds.load('plant_village', split='train', with_info=True, data_dir=dirs['rgb_health'])
        print(f"Successfully downloaded {info.splits['train'].num_examples} health images.")
    except Exception as e:
        print(f"Error downloading TFDS: {e}")

    # 3. Thermal Imaging Data
    print("\n3. Thermal Imaging Data Structure Created.")
    print(f"NOTE: Please manually download the Danforth/AI4EOSC thermal datasets and extract them to: {dirs['thermal_stress']}")

if __name__ == "__main__":
    #download_and_organize_data()
    download_advanced_datasets()