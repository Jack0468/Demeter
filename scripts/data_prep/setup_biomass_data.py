"""
=============================================================================
Demeter Project: Manual Biomass Data Loader (Dataset 6)
=============================================================================

Purpose:
    Loads the manual biomass measurements CSV (fresh_weight, dry_weight)
    and links each plant to its PNG images in the biomass image directory.

    Each plant has up to 5 images: 4 side-views (0/90/180/270 degrees) and
    1 top-view. This loader expands the dataset to include all side-view
    angles as separate training rows (multi-angle strategy), effectively
    multiplying usable training data by up to 4x.

Dataset 6 (Danforth):
    205 PNG images, 1 CSV file.
    41 plants with fresh_weight and dry_weight measurements.

=============================================================================
"""
import os
import glob
import pandas as pd


def load_biomass_data(csv_filepath: str, images_dir: str, multi_angle: bool = True) -> pd.DataFrame:
    """
    Loads Dataset 6 (Manual biomass measurements) and links them to PNG images.

    Args:
        csv_filepath: Path to manual_biomass_samples.csv
        images_dir:   Path to the image directory containing biomass PNGs
        multi_angle:  If True, expands each plant to one row per side-view
                      angle (0/90/180/270), quadrupling the effective dataset
                      size. If False, uses only the 0-degree canonical image.

    Returns:
        DataFrame with columns: plant_id, datetime, fresh_weight, dry_weight,
        filepath, angle
    """
    print(f"Loading Biomass Measurement data from {csv_filepath}...")

    if not os.path.exists(csv_filepath):
        print(f"[!] ERROR: CSV file not found at {csv_filepath}")
        return None

    if not os.path.isdir(images_dir):
        print(f"[!] ERROR: Image directory not found at {images_dir}")
        return None

    try:
        df = pd.read_csv(csv_filepath)
    except Exception as e:
        print(f"[!] Error reading CSV: {e}")
        return None

    df.columns = df.columns.str.strip()

    # Validate required columns
    required_cols = ['plant_id', 'fresh_weight', 'dry_weight']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[!] ERROR: Missing columns in CSV: {missing}")
        return None

    # Side-view angles available in the dataset
    sv_angles = [0, 90, 180, 270]

    rows = []
    unmatched = []

    for _, row in df.iterrows():
        plant_id = str(row['plant_id']).strip()

        if multi_angle:
            # Search for each of the 4 side-view angles
            for angle in sv_angles:
                pattern = os.path.join(images_dir, f"{plant_id}_vis_sv_{angle}_*.png")
                matches = glob.glob(pattern)
                if matches:
                    rows.append({
                        'plant_id': plant_id,
                        'datetime': row.get('datetime'),
                        'fresh_weight': float(row['fresh_weight']),
                        'dry_weight': float(row['dry_weight']),
                        'filepath': matches[0],  # Take first if multiple exist
                        'angle': angle
                    })
        else:
            # Single canonical image: side-view at 0 degrees
            pattern = os.path.join(images_dir, f"{plant_id}_vis_sv_0_*.png")
            matches = glob.glob(pattern)
            if matches:
                rows.append({
                    'plant_id': plant_id,
                    'datetime': row.get('datetime'),
                    'fresh_weight': float(row['fresh_weight']),
                    'dry_weight': float(row['dry_weight']),
                    'filepath': matches[0],
                    'angle': 0
                })
            else:
                unmatched.append(plant_id)

    if unmatched:
        print(f"[!] Warning: No images found for {len(unmatched)} plants: {unmatched}")

    if not rows:
        print("[!] ERROR: No image-linked records could be built.")
        return None

    result_df = pd.DataFrame(rows)
    n_plants = result_df['plant_id'].nunique()
    print(f"Successfully linked {len(result_df)} image records for {n_plants} plants.")
    if multi_angle:
        print(f"  (Multi-angle mode: up to {len(sv_angles)} rows per plant)")

    return result_df
