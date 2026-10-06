import pandas as pd
import numpy as np
import os
import glob

# --- KONFIGURASI INDEKS MEDIAPIPE UNTUK PUSH-UP ---
KP_INDICES = {
    'RShoulder': 12, # Bahu Kanan
    'RElbow': 14,    # Siku Kanan
    'RWrist': 16,    # Pergelangan Tangan Kanan
    'RHip': 24,      # Pinggul Kanan
    'RKnee': 26,     # Lutut Kanan
    'RAnkle': 28,    # Pergelangan Kaki Kanan
    # Untuk normalisasi skala & Torso stability
    'LShoulder': 11, # Bahu Kiri
    'LHip': 23,      # Pinggul Kiri
    'LAnkle': 27,    # Pergelangan Kaki Kiri
}

# --- DEFINISI GLOBAL FEATURE COLUMNS (BARIS BARU YANG DITAMBAHKAN) ---
# Diperlukan agar variabel ini dapat diimpor oleh app_server.py
FEATURE_COLUMNS = (
    [f'{name}_{axis}_NORM' for name in KP_INDICES.keys() for axis in ['X', 'Y']] +
    ['R_ELBOW_ANGLE', 'TORSO_ANGLE', 'R_SHOULDER_TUCK_ANGLE']
)
# --- AKHIR BARIS BARU ---


# --- DIREKTORI OUTPUT ---
CSV_DATA_DIR_CORRECT = 'form_benar'
CSV_DATA_DIR_INCORRECT = 'form_salah'
OUTPUT_MODEL_DIR = 'rf_model_data' # Harus ada di folder push_up
OUTPUT_DATA_FILE = os.path.join(OUTPUT_MODEL_DIR, 'rf_features_pushup.csv')

# Fungsi bantuan untuk mendapatkan nama kolom penuh dari indeks
def get_col_name(index, axis):
    return f"Landmark{index}_{axis}"

def calculate_angle(row, p1_idx, p2_idx, p3_idx):
    """Menghitung sudut 3 keypoint (p1-p2-p3) dari baris DataFrame menggunakan indeks."""
    try:
        p1 = np.array([row[get_col_name(p1_idx, 'X')], row[get_col_name(p1_idx, 'Y')]])
        p2 = np.array([row[get_col_name(p2_idx, 'X')], row[get_col_name(p2_idx, 'Y')]])
        p3 = np.array([row[get_col_name(p3_idx, 'X')], row[get_col_name(p3_idx, 'Y')]])
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle
    except KeyError as e:
        raise KeyError(f"Kolom keypoint tidak ditemukan untuk index {e}.")
    except Exception:
        return np.nan 

def normalize_keypoints(df):
    """Normalisasi keypoint relatif terhadap Bahu Kanan dan skala panjang Torso Rata-rata."""
    
    # 1. Hitung panjang Torso Rata-rata (untuk skala)
    try:
        # Gunakan jarak Bahu-Hip untuk normalisasi
        dist_r = np.sqrt(
            (df[get_col_name(KP_INDICES['RShoulder'], 'X')] - df[get_col_name(KP_INDICES['RHip'], 'X')])**2 + 
            (df[get_col_name(KP_INDICES['RShoulder'], 'Y')] - df[get_col_name(KP_INDICES['RHip'], 'Y')])**2
        )
        dist_l = np.sqrt(
            (df[get_col_name(KP_INDICES['LShoulder'], 'X')] - df[get_col_name(KP_INDICES['LHip'], 'X')])**2 + 
            (df[get_col_name(KP_INDICES['LShoulder'], 'Y')] - df[get_col_name(KP_INDICES['LHip'], 'Y')])**2
        )
        torso_length = (dist_r + dist_l) / 2
        avg_torso = torso_length.mean()
    except KeyError as e:
        raise KeyError(f"Kolom Torso tidak ditemukan. Error: {e}")
    
    if avg_torso == 0 or avg_torso == np.nan:
        return None 
        
    # Ambil koordinat bahu kanan sebagai titik referensi (origin 0,0)
    shoulder_x = df[get_col_name(KP_INDICES['RShoulder'], 'X')]
    shoulder_y = df[get_col_name(KP_INDICES['RShoulder'], 'Y')]
        
    # 2. Normalisasi dan Ekstraksi Keypoint yang Relevan (X dan Y)
    features_to_normalize = {}
    for name, index in KP_INDICES.items():
        # NORM: (KP - R_SHOULDER) / AVG_TORSO 
        col_x = get_col_name(index, 'X')
        new_col_x = f'{name}_X_NORM'
        features_to_normalize[new_col_x] = (df[col_x] - shoulder_x) / avg_torso
        
        col_y = get_col_name(index, 'Y')
        new_col_y = f'{name}_Y_NORM'
        features_to_normalize[new_col_y] = (df[col_y] - shoulder_y) / avg_torso

    df_norm = pd.DataFrame(features_to_normalize)
    return df_norm

def extract_features():
    all_features = []
    
    if not os.path.exists(OUTPUT_MODEL_DIR):
        os.makedirs(OUTPUT_MODEL_DIR)

    required_cols_check = [get_col_name(idx, 'X') for idx in KP_INDICES.values()]

    for label, data_dir in [('correct', CSV_DATA_DIR_CORRECT), ('incorrect', CSV_DATA_DIR_INCORRECT)]:
        for filename in glob.glob(os.path.join(data_dir, '*.csv')):
            try:
                df = pd.read_csv(filename)
                
                # Cek apakah kolom-kolom penting ada
                if not all(col in df.columns for col in required_cols_check):
                    print(f"Skip: {filename} Kolom utama keypoint Push-up tidak lengkap.")
                    continue

                # 1. Normalisasi Keypoint
                df_norm = normalize_keypoints(df)

                if df_norm is None or df_norm.empty:
                    print(f"Skip: {filename} karena gagal dinormalisasi atau kosong.")
                    continue

                # 2. Hitung Fitur Sudut (menggunakan data asli)
                
                # Sudut Siku Kanan: Shoulder-Elbow-Wrist (12-14-16) -> Untuk kedalaman
                df_norm['R_ELBOW_ANGLE'] = df.apply(
                    lambda row: calculate_angle(row, KP_INDICES['RShoulder'], KP_INDICES['RElbow'], KP_INDICES['RWrist']), axis=1
                )
                
                # Sudut Torso/Tubuh: Shoulder-Hip-Knee (12-24-26) -> Harus mendekati 180° untuk punggung lurus
                df_norm['TORSO_ANGLE'] = df.apply(
                    lambda row: calculate_angle(row, KP_INDICES['RShoulder'], KP_INDICES['RHip'], KP_INDICES['RKnee']), axis=1
                )
                
                # Sudut Bahu/Armpit: Elbow-Shoulder-Hip (14-12-24) -> Untuk Tuck Siku
                df_norm['R_SHOULDER_TUCK_ANGLE'] = df.apply(
                    lambda row: calculate_angle(row, KP_INDICES['RElbow'], KP_INDICES['RShoulder'], KP_INDICES['RHip']), axis=1
                )
                
                # 3. Kumpulkan dan Bersihkan
                # Kita akan drop baris dengan NaN (misal: visibility keypoint rendah)
                features_df = df_norm.dropna()
                
                if features_df.empty:
                    print(f"Skip: {filename} karena datanya kosong setelah dibersihkan.")
                    continue
                
                features_df['LABEL'] = label
                all_features.append(features_df)
                
                print(f"Processed: {filename} ({len(features_df)} frames)")

            except KeyError as e:
                print(f"FATAL ERROR processing {filename}: Pastikan semua kolom keypoint MediaPipe ada. Error: {e}")
                continue
            except Exception as e:
                print(f"ERROR processing {filename}: {e}")

    # 2. Menggabungkan dan Menyimpan data
    if all_features:
        final_df = pd.concat(all_features, ignore_index=True)
        
        final_df.to_csv(OUTPUT_DATA_FILE, index=False)
        print(f"\nTotal {len(final_df)} frame disimpan ke: {OUTPUT_DATA_FILE} (Siap untuk Training)")
        print("\n--- LANJUTKAN KE TAHAP PELATIHAN ---")
        print("Jalankan: 'python rf_trainer.py' (Gunakan kode trainer Anda, tapi pastikan ia membaca rf_features_pushup.csv)")
        return final_df
    else:
        print("Tidak ada data yang berhasil diproses.")
        return None

if __name__ == '__main__':
    extract_features()