import pandas as pd
import numpy as np
import pywt
import os

# Configuration
DATE_COL = 'DateTime'
TARGET_COL = 'Speed'  # Wind Speed

def load_and_preprocess_data(file_path):
    """
    Loads the CSV file, handles missing values, and creates lag features.
    """
    if not os.path.exists(file_path):
        print(f"ERROR: File not found -> {file_path}")
        return None, None
    
    df = pd.read_csv(file_path)
    
    # Parse Dates
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df.set_index(DATE_COL, inplace=True)
    
    # Handle Target Column
    col = TARGET_COL
    if TARGET_COL not in df.columns:
        # If specific column not found, take the first numeric column
        col = df.select_dtypes(include=[np.number]).columns[0]

    # Impute Missing Values (Median)
    df[col] = df[col].fillna(df[col].median())
    
    # Feature Engineering: Lag Features (t-1, t-3)
    df['Lag_1'] = df[col].shift(1)
    df['Lag_3'] = df[col].shift(3)
    df.dropna(inplace=True)
    
    return df, col

def decompose_signal(signal, wavelet='db4', level=3):
    """
    Decomposes the signal using DWT and applies MRA (reconstruction) 
    to match original length.
    """
    try:
        coeffs = pywt.wavedec(signal, wavelet, level=level)
    except:
        # Reduce level if signal is too short
        coeffs = pywt.wavedec(signal, wavelet, level=1)
        
    components = {}
    for i, coeff in enumerate(coeffs):
        # Create a list of zeros and insert only the current coefficient
        temp_coeffs = [np.zeros_like(c) for c in coeffs]
        temp_coeffs[i] = coeff
        
        # Reconstruct
        recon = pywt.waverec(temp_coeffs, wavelet)
        
        # Match dimensions (trim padding)
        if len(recon) > len(signal): 
            recon = recon[:len(signal)]
            
        components[f'Comp_{i}'] = recon
        
    return pd.DataFrame(components, index=signal.index)