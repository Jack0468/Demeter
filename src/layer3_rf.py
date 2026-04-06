import os
import pandas as pd
import joblib 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_environment_model(csv_path, save_path):
    print(f"Training Layer 3 Random Forest from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    X = df[['Temp', 'Moisture', 'Light']] # Update with actual Danforth/Kaggle columns
    y = df['Needs_Water'] 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train, y_train)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(rf_model, save_path)