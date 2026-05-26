import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ==========================================
# RANDOM FOREST TRAINING PIPELINE - DANFORTH GROWTH PREDICTION
# ==========================================
def train_and_save_rf_danforth(danforth_csv_path, save_path):
    """Trains a Random Forest Regressor on Danforth growth data to predict growth milestones."""
    print("Initializing Random Forest Regressor Training Pipeline (Danforth)...")
    
    # Load the environmental sensor data
    if not os.path.exists(danforth_csv_path):
        print(f"[!] ERROR: Danforth data not found at {danforth_csv_path}")
        return False
    
    df = pd.read_csv(danforth_csv_path)
    print(f"Loaded {len(df)} records from Danforth dataset")
    
    # Drop rows with NaN values
    df_clean = df.dropna().copy()
    print(f"After removing NaN: {len(df_clean)} records")
    
    # Features: Environmental conditions (excluding Growth_Milestone target)
    feature_cols = [col for col in df_clean.columns if col != 'Growth_Milestone']
    X = df_clean[feature_cols]
    
    # Target: Growth milestone (binary or continuous)
    y = df_clean['Growth_Milestone']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    try:
        from src.training.split_tracker import update_manifest
        update_manifest("demeter_rf_danforth", X_train.index.tolist(), X_test.index.tolist())
    except Exception as e:
        print(f"Failed to save manifest: {e}")
    
    # We will use a Scikit-Learn Pipeline to handle categorical features automatically
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    # Train as a Regressor inside a Pipeline
    rf_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    rf_model.fit(X_train, y_train)
    
    # Evaluate using RMSE
    predictions = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print(f"Random Forest Regressor RMSE: {rmse:.4f}")
    
    # Feature importances are ordered by the ColumnTransformer (categorical first, then remainder)
    ordered_features = categorical_cols + [c for c in X.columns if c not in categorical_cols]
    regressor = rf_model.named_steps['regressor']
    print(f"Feature importance: {dict(zip(ordered_features, regressor.feature_importances_))}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(rf_model, save_path)
    print(f"Random Forest Regressor successfully saved to {save_path}")
    
    return True


# ==========================================
# RANDOM FOREST TRAINING PIPELINE - BELLWETHER WATER STRESS (Original)
# ==========================================
def train_and_save_rf(df, save_path):
    """Trains a Random Forest Regressor on tabular sensor data to predict growth trajectory."""
    print("Initializing Random Forest Regressor Training Pipeline...")
    
    # Drop rows with NaN values in our target features
    df_clean = df.dropna(subset=['weight before', 'water amount', 'weight after']).copy()
    
    # Features: Current state (Weight) + Intervention (Water Applied)
    X = df_clean[['weight before', 'water amount']] 
    
    # Target: Continuous future growth metric (The resulting weight of the plant/soil)
    y = df_clean['weight after']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train as a Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate using RMSE (As promised in the proposal)
    predictions = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print(f"Random Forest Regressor RMSE: {rmse:.2f}g")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(rf_model, save_path)
    print(f"Random Forest Regressor successfully saved to {save_path}")
