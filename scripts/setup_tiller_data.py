"""
=============================================================================
Demeter Project: Manual Phenotypic Data Loader (Dataset 3)
=============================================================================

Purpose:
    This module processes the manual tiller count, tiller angle, and leaf 
    angle measurements from the Fahlgren et al. (2015) Bellwether dataset. 
    It reads the tabular data and dynamically matches each plant's physical 
    measurements to its corresponding PNG image file using the plant_id.

Use Case (Demeter Architecture):
    While the main pipeline (main.py) handles the 118 GB dataset for 
    stress classification and water need prediction, this module is designed 
    to isolate Dataset 3. It generates a linked dataframe that can be passed 
    to a secondary CNN pipeline to predict continuous physical architecture 
    metrics (like tiller_count) directly from a visual input.

=============================================================================
"""
import os
import glob
import pandas as pd

def load_manual_tiller_data(txt_filepath, images_dir):
    """
    Loads Dataset 3 (Manual measurements) and links them to their PNG images.
    """
    print(f"Loading Tiller Measurement data from {txt_filepath}...")
    
    # Check if the file exists
    if not os.path.exists(txt_filepath):
        print(f"[!] ERROR: Tabular file not found at {txt_filepath}")
        return None

    # Load the text file. 
    try:
        df = pd.read_csv(txt_filepath, sep='\t') 
    except Exception as e:
        print(f"[!] Error reading file: {e}")
        return None

    # Clean column names just in case
    df.columns = df.columns.str.strip()

    # Define a helper function to find the image path
    def find_image_path(plant_id):
        # Search for any .png file starting with the plant_id in the target folder
        search_pattern = os.path.join(images_dir, f"{plant_id}*.png")
        matches = glob.glob(search_pattern)
        
        # If a match is found, return the first one. Otherwise, return None.
        if matches:
            return matches[0]
        return None

    # Apply the helper function to create a new 'filepath' column
    df['filepath'] = df['plant_id'].apply(find_image_path)

    # Filter out any rows where the image couldn't be found
    matched_df = df.dropna(subset=['filepath']).copy()
    
    print(f"Successfully linked {len(matched_df)} records with their images.")
    
    return matched_df