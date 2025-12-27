import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Import shared functions
from utils import load_and_preprocess_data

# SETTINGS
TEST_FILE = r'C:\Users\erenf\feke_20.csv'
MODEL_DIR = 'models'

print("--- TESTING PROCESS STARTED ---")

# 1. Load Test Data
test_df, target_col = load_and_preprocess_data(TEST_FILE)
if test_df is None: exit()

X_test = test_df[['Lag_1', 'Lag_3']]
y_test = test_df[target_col]

# Load Scaler and Transform Data
scaler = joblib.load(f'{MODEL_DIR}/scaler.pkl')
X_test_scaled = scaler.transform(X_test)

model_names = ['RF', 'DT', 'ANN']
results = []
predictions = {'Actual': y_test}

# ---------------------------------------------------------
# PHASE 1: TEST BASELINE MODELS
# ---------------------------------------------------------
print("\n>>> Testing Baseline Models...")

for name in model_names:
    # Load Model
    model = joblib.load(f'{MODEL_DIR}/baseline_{name}.pkl')
    
    # Predict
    if name == 'ANN':
        pred = model.predict(X_test_scaled)
    else:
        pred = model.predict(X_test)
        
    predictions[f'Baseline_{name}'] = pred
    
    # Metrics Calculation
    mse = mean_squared_error(y_test, pred)           # Mean Squared Error
    rmse = np.sqrt(mse)                              # Root Mean Squared Error
    mae = mean_absolute_error(y_test, pred)          # Mean Absolute Error
    r2 = r2_score(y_test, pred)                      # R-Squared
    
    results.append({
        'Type': 'Baseline', 
        'Model': name, 
        'RMSE': rmse, 
        'R2': r2, 
        'MAE': mae, 
        'MSE': mse
    })

# ---------------------------------------------------------
# PHASE 2: TEST HYBRID WAVELET MODELS
# ---------------------------------------------------------
print("\n>>> Testing Hybrid Wavelet Models...")

for name in model_names:
    # Load Sub-Models Dictionary
    sub_models = joblib.load(f'{MODEL_DIR}/hybrid_{name}.pkl')
    
    y_pred_total = np.zeros(len(y_test))
    
    # Predict each component and sum them up
    for comp_name, model in sub_models.items():
        if name == 'ANN':
            pred_part = model.predict(X_test_scaled)
        else:
            pred_part = model.predict(X_test)
        
        y_pred_total += pred_part
        
    predictions[f'Hybrid_{name}'] = y_pred_total
    
    # Metrics Calculation
    mse = mean_squared_error(y_test, y_pred_total)   # Mean Squared Error
    rmse = np.sqrt(mse)                              # Root Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred_total)  # Mean Absolute Error
    r2 = r2_score(y_test, y_pred_total)              # R-Squared
    
    results.append({
        'Type': 'Hybrid', 
        'Model': name, 
        'RMSE': rmse, 
        'R2': r2, 
        'MAE': mae, 
        'MSE': mse
    })

# ---------------------------------------------------------
# REPORTING & VISUALIZATION
# ---------------------------------------------------------
# Create DataFrame
df_results = pd.DataFrame(results)

# Reorder columns for better readability
df_results = df_results[['Type', 'Model', 'RMSE', 'MAE', 'MSE', 'R2']]
df_results = df_results.sort_values(by='RMSE')

print("\n" + "="*60)
print("FULL PERFORMANCE METRICS TABLE")
print("="*60)
print(df_results.to_string(index=False))

# --- Save Comparison Table to CSV ---
csv_filename = 'model_performance_comparison.csv'
df_results.to_csv(csv_filename, index=False)
print(f"\n[SUCCESS] Comparison table saved as '{csv_filename}'.")

# --- PLOTTING ---
fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

# 1. Random Forest Comparison
axes[0].plot(y_test.index, y_test, label='Actual', color='black', alpha=0.4)
axes[0].plot(y_test.index, predictions['Baseline_RF'], label='Baseline RF', color='orange', linestyle='--')
axes[0].plot(y_test.index, predictions['Hybrid_RF'], label='Hybrid Wavelet RF', color='red')
axes[0].set_title(f"Random Forest (RMSE: Baseline={df_results[(df_results['Model']=='RF') & (df_results['Type']=='Baseline')]['RMSE'].values[0]:.2f} vs Hybrid={df_results[(df_results['Model']=='RF') & (df_results['Type']=='Hybrid')]['RMSE'].values[0]:.2f})")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Decision Tree Comparison
axes[1].plot(y_test.index, y_test, label='Actual', color='black', alpha=0.4)
axes[1].plot(y_test.index, predictions['Baseline_DT'], label='Baseline DT', color='orange', linestyle='--')
axes[1].plot(y_test.index, predictions['Hybrid_DT'], label='Hybrid Wavelet DT', color='red')
axes[1].set_title(f"Decision Tree (RMSE: Baseline={df_results[(df_results['Model']=='DT') & (df_results['Type']=='Baseline')]['RMSE'].values[0]:.2f} vs Hybrid={df_results[(df_results['Model']=='DT') & (df_results['Type']=='Hybrid')]['RMSE'].values[0]:.2f})")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. ANN Comparison
axes[2].plot(y_test.index, y_test, label='Actual', color='black', alpha=0.4)
axes[2].plot(y_test.index, predictions['Baseline_ANN'], label='Baseline ANN', color='orange', linestyle='--')
axes[2].plot(y_test.index, predictions['Hybrid_ANN'], label='Hybrid Wavelet ANN', color='red')
axes[2].set_title(f"ANN (RMSE: Baseline={df_results[(df_results['Model']=='ANN') & (df_results['Type']=='Baseline')]['RMSE'].values[0]:.2f} vs Hybrid={df_results[(df_results['Model']=='ANN') & (df_results['Type']=='Hybrid')]['RMSE'].values[0]:.2f})")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.xlabel("Date / Time")
plt.tight_layout()
plt.savefig('all_models_comparison.png')
plt.show()

print("\n[SUCCESS] Plot saved as 'all_models_comparison.png'.")