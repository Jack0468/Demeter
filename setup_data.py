import os
import shutil
import kagglehub

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

if __name__ == "__main__":
    download_and_organize_data()
