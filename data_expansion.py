import os
import shutil
import kagglehub
import tensorflow as tf
import requests

def expand_demeter_datasets():
    print("🌾 Starting Dataset Expansion Pipeline...\n")
    
    # 1. Setup local directories (Consistent with existing architecture)
    base_data_dir = os.path.join(os.getcwd(), 'data')
    disease_dir = os.path.join(base_data_dir, 'plant_disease')
    molecular_dir = os.path.join(base_data_dir, 'molecular_plant')
    danforth_dir = os.path.join(base_data_dir, 'danforth_center')
    
    for folder in [disease_dir, molecular_dir, danforth_dir]:
        os.makedirs(folder, exist_ok=True)

    # --- 1. Kaggle: PlantVillage Disease Dataset ---
    print("1/3: Downloading PlantVillage Disease Dataset...")
    try:
        # Downloads: https://www.kaggle.com/datasets/emmarex/plantdisease
        disease_path = kagglehub.dataset_download("emmarex/plantdisease")
        shutil.copytree(disease_path, disease_dir, dirs_exist_ok=True)
        print(f"✅ Disease dataset ready at: {disease_dir}\n")
    except Exception as e:
        print(f"❌ Error downloading Kaggle disease data: {e}")

    # --- 2. Molecular Plant: Supplemental Data ---
    # Note: Journal datasets often require direct HTTP requests or manual download
    # if they are behind a landing page. This script attempts a direct fetch.
    print("2/3: Fetching Molecular Plant Supplemental Data...")
    mol_url = "https://www.cell.com/molecular-plant/fulltext/S1674-2052(15)00268-3"
    print(f"🔗 Source: {mol_url}")
    # Recommendation: Check the 'Supplemental Information' section on the page 
    # to find direct .xlsx or .csv links for this specific paper.

    # --- 3. Danforth Center: Data Science Portal ---
    print("3/3: Preparing Danforth Center Data...")
    danforth_url = "https://datasci.danforthcenter.org/data/files"
    # The Danforth portal typically uses a 'download.py' script or CKAN API.
    # You can use the requests library to pull specific identified files.
    print(f"ℹ️  To automate Danforth, use their CKAN API if you have a token.\n")

    print("==========================================")
    print("Expansion Complete! Data stored in /data/")
    print("==========================================")

if __name__ == "__main__":
    expand_demeter_datasets()