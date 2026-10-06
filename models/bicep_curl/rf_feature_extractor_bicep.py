# File: models/bicep_curl/rf_feature_extractor_bicep.py

import pandas as pd
import numpy as np
import os
import glob

# --- KONFIGURASI MEDIAPIPE INDEKS KEYPOINT ---
KP_INDICES = {
    'RShoulder': 12, # Bahu Kanan
    'RElbow': 14, # Siku Kanan
    'RWrist': 16, # Pergelangan Tangan Kanan
    'RHip': 24,  # Pinggul Kanan
    'LHip': 23,  # Pinggul Kiri
    'LShoulder': 11, # Bahu Kiri
}

# --- DEFINISI GLOBAL FEATURE COLUMNS ---
FEATURE_COLUMNS = (
    # Keypoint yang dinormalisasi (X dan Y)
    [f'{name}_{axis}_NORM' for name in KP_INDICES.keys() for axis in ['X', 'Y']] +
    # Sudut-sudut fitur spesifik
    ['R_ELBOW_ANGLE', 'R_SHOULDER_ANGLE'] 
)

# --- DIREKTORI ---
CSV_DATA_DIR_CORRECT = 'form_benar'
CSV_DATA_DIR_INCORRECT = 'form_salah'
OUTPUT_MODEL_DIR = 'rf_model_data'
OUTPUT_DATA_FILE = os.path.join(OUTPUT_MODEL_DIR, 'rf_features.csv')

# FUNGSI BANTUAN VITAL UNTUK PEMROSESAN SERVER
def get_col_name(index, axis):
    """Mendapatkan nama kolom sesuai format LandmarkX_Y"""
    return f"Landmark{index}_{axis}"


def calculate_angle(row, p1_idx, p2_idx, p3_idx):
    """Menghitung sudut 3 keypoint (p1-p2-p3) menggunakan indeks Landmark."""
    try:
        p1 = np.array([row[get_col_name(p1_idx, 'X')], row[get_col_name(p1_idx, 'Y')]])
        p2 = np.array([row[get_col_name(p2_idx, 'X')], row[get_col_name(p2_idx, 'Y')]])
        p3 = np.array([row[get_col_name(p3_idx, 'X')], row[get_col_name(p3_idx, 'Y')]])
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle
    except Exception:
        return np.nan # Mengembalikan NaN jika ada error perhitungan


def normalize_keypoints(df):
    """Normalisasi keypoint relatif terhadap Bahu Kanan dan skala panjang Torso Rata-rata."""
    
    # 1. Hitung panjang Torso Rata-rata (untuk skala)
    try:
        # Jarak Bahu ke Pinggul Kanan
        dist_r = np.sqrt(
            (df[get_col_name(KP_INDICES['RShoulder'], 'X')] - df[get_col_name(KP_INDICES['RHip'], 'X')])**2 + 
            (df[get_col_name(KP_INDICES['RShoulder'], 'Y')] - df[get_col_name(KP_INDICES['RHip'], 'Y')])**2
        )
        # Jarak Bahu ke Pinggul Kiri
        dist_l = np.sqrt(
            (df[get_col_name(KP_INDICES['LShoulder'], 'X')] - df[get_col_name(KP_INDICES['LHip'], 'X')])**2 + 
            (df[get_col_name(KP_INDICES['LShoulder'], 'Y')] - df[get_col_name(KP_INDICES['LHip'], 'Y')])**2
        )
        torso_length = (dist_r + dist_l) / 2
        
        # Cek validitas torso
        if torso_length.iloc[0] == 0 or np.isnan(torso_length.iloc[0]):
            return None
        
        avg_torso = torso_length.iloc[0] # Ambil nilai torso untuk frame ini

    except KeyError:
        return None
    
    if avg_torso == 0 or np.isnan(avg_torso):
        return None 
        
    # Ambil koordinat bahu kanan sebagai titik referensi (origin 0,0)
    shoulder_x = df[get_col_name(KP_INDICES['RShoulder'], 'X')].iloc[0]
    shoulder_y = df[get_col_name(KP_INDICES['RShoulder'], 'Y')].iloc[0]
        
    # 2. Normalisasi dan Ekstraksi Keypoint yang Relevan (X dan Y)
    features_to_normalize = {}
    for name, index in KP_INDICES.items():
        col_x = get_col_name(index, 'X')
        new_col_x = f'{name}_X_NORM' 
        features_to_normalize[new_col_x] = (df[col_x] - shoulder_x) / avg_torso
        
        col_y = get_col_name(index, 'Y')
        new_col_y = f'{name}_Y_NORM'
        features_to_normalize[new_col_y] = (df[col_y] - shoulder_y) / avg_torso

    return pd.DataFrame(features_to_normalize)


# Definisikan fungsi extract_features() jika Anda ingin menjadikannya executable untuk training
def extract_features():
    # ... (Isi seperti skrip lama Anda)
    pass
    
if __name__ == '__main__':
    # extract_features() 
    pass