import pandas as pd
import numpy as np
import joblib  # For saving models
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

# Import shared functions
from utils import load_and_preprocess_data, decompose_signal

# SETTINGS
TRAIN_FILE = r'C:\Users\erenf\Feke_80.csv'
MODEL_DIR = 'models'

# Create model directory if not exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print("--- TRAINING PROCESS STARTED ---")

# 1. Load Data
train_df, target_col = load_and_preprocess_data(TRAIN_FILE)
if train_df is None: exit()

X_train = train_df[['Lag_1', 'Lag_3']]
y_train = train_df[target_col]

# Scale Data for ANN (and save the scaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, f'{MODEL_DIR}/scaler.pkl')

# 2. Define Models
models = {
    'RF': RandomForestRegressor(n_estimators=100, random_state=42),
    'DT': DecisionTreeRegressor(random_state=42),
    'ANN': MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42)
}

# ---------------------------------------------------------
# PHASE 1: TRAIN BASELINE MODELS
# ---------------------------------------------------------
print("\n>>> Training Baseline Models...")
for name, model in models.items():
    print(f" -> Training {name}...")
    
    if name == 'ANN':
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)
    
    # Save Model
    joblib.dump(model, f'{MODEL_DIR}/baseline_{name}.pkl')

# ---------------------------------------------------------
# PHASE 2: TRAIN HYBRID WAVELET MODELS
# ---------------------------------------------------------
print("\n>>> Training Hybrid Wavelet Models...")

# Decompose Target Signal
train_components = decompose_signal(y_train)

for name, base_model in models.items():
    print(f" -> Building Hybrid {name} System...")
    
    # Dictionary to hold sub-models for each component
    hybrid_sub_models = {}
    
    # Train a separate model for each component (Trend, Noise, etc.)
    for comp_name in train_components.columns:
        y_part = train_components[comp_name]
        
        # Clone the base model architecture
        model = clone(base_model)
        
        if name == 'ANN':
            model.fit(X_train_scaled, y_part)
        else:
            model.fit(X_train, y_part)
            
        hybrid_sub_models[comp_name] = model
    
    # Save the dictionary of sub-models
    joblib.dump(hybrid_sub_models, f'{MODEL_DIR}/hybrid_{name}.pkl')

print("\n--- ALL MODELS SAVED SUCCESSFULLY! ---")
print(f"Location: '{MODEL_DIR}/'")