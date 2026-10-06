# File: ProjectAI/app_server.py (VERSI FINAL DENGAN LOGIKA BICEP CURL TERSINKRONISASI)

from flask import Flask, render_template, request, jsonify
import joblib
import os
import numpy as np
import pandas as pd
from typing import Dict, Any

# --- IMPORT FEATURE EXTRACTOR UTAMA (Push-up) ---
from models.push_up.rf_feature_extractor_pushup import (
 calculate_angle, 
 normalize_keypoints, 
 KP_INDICES as PUSHUP_KP_INDICES, 
 FEATURE_COLUMNS as PUSHUP_FEATURE_COLUMNS
)

# --- FUNGSI BANTUAN UNTUK MEMPROSES KEYPOINT ---

def get_keypoint_data_from_list(keypoints_list_of_lists):
    """
    Mengubah list of lists MediaPipe mentah ([x, y, z, visibility]) 
    menjadi list of dict yang mudah diakses dan menambahkan index landmark.
    """
    processed_keypoints = []
    
    for i, kp in enumerate(keypoints_list_of_lists):
        if len(kp) >= 4:
            processed_keypoints.append({
                'x': kp[0],
                'y': kp[1],
                'z': kp[2],
                'visibility': kp[3],
                'index': i
            })
        else:
            processed_keypoints.append({
                'x': np.nan, 'y': np.nan, 'z': np.nan, 'visibility': np.nan, 'index': i
            })
            
    return processed_keypoints

# --- Fungsi pemrosesan Pushup ---
def process_keypoints_for_pushup(keypoints_list):
    keypoint_data = get_keypoint_data_from_list(keypoints_list)
    raw_data = {}
    
    for name, index in PUSHUP_KP_INDICES.items():
        kp = keypoint_data[index] 
        # Di sini, Anda mungkin perlu get_col_name dari push_up.rf_feature_extractor_pushup
        # Tetapi karena ini adalah kode yang berfungsi di Pushup, kita biarkan saja.
        raw_data[f'Landmark{index}_X'] = kp['x']
        raw_data[f'Landmark{index}_Y'] = kp['y']
        
    df_raw = pd.DataFrame([raw_data])
    
    try:
        df_norm = normalize_keypoints(df_raw)
    except Exception:
        return None
    if df_norm is None or df_norm.empty:
        return None

    df_norm['R_ELBOW_ANGLE'] = df_raw.apply(
        lambda row: calculate_angle(row, PUSHUP_KP_INDICES['RShoulder'], PUSHUP_KP_INDICES['RElbow'], PUSHUP_KP_INDICES['RWrist']), axis=1
    )
    df_norm['TORSO_ANGLE'] = df_raw.apply(
        lambda row: calculate_angle(row, PUSHUP_KP_INDICES['RShoulder'], PUSHUP_KP_INDICES['RHip'], PUSHUP_KP_INDICES['RKnee']), axis=1
    )
    df_norm['R_SHOULDER_TUCK_ANGLE'] = df_raw.apply(
        lambda row: calculate_angle(row, PUSHUP_KP_INDICES['RElbow'], PUSHUP_KP_INDICES['RShoulder'], PUSHUP_KP_INDICES['RHip']), axis=1
    )
    
    features_df = df_norm.filter(items=PUSHUP_FEATURE_COLUMNS, axis=1)
    if features_df.isnull().values.any():
        return None
        
    return features_df


# --- IMPORT & SINKRONISASI BICEP CURL DARI rf_prediction.py ---

# Konfigurasi Keypoint dan Kolom Fitur yang sama
BICEP_KP_INDICES = {'RShoulder': 12, 'RElbow': 14, 'RWrist': 16, 'RHip': 24, 'LHip': 23, 'LShoulder': 11}
BICEP_FEATURE_COLUMNS = (
    [f'{name}_{axis}_NORM' for name in BICEP_KP_INDICES.keys() for axis in ['X', 'Y']] +
    ['R_ELBOW_ANGLE', 'R_SHOULDER_ANGLE']
)

def bicep_calculate_angle_dict(keypoint_data, p1_idx, p2_idx, p3_idx):
    """Menghitung sudut 3 keypoint (p1-p2-p3) menggunakan List of Dict keypoint_data."""
    try:
        p1 = np.array([keypoint_data[p1_idx]['x'], keypoint_data[p1_idx]['y']])
        p2 = np.array([keypoint_data[p2_idx]['x'], keypoint_data[p2_idx]['y']])
        p3 = np.array([keypoint_data[p3_idx]['x'], keypoint_data[p3_idx]['y']])
        v1 = p1 - p2
        v2 = p3 - p2
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    except Exception:
        return np.nan

def process_keypoints_for_bicep(keypoints_list):
    """
    Memproses keypoint untuk model Bicep Curl menggunakan logika 
    extract_and_normalize_features dari rf_prediction.py yang berhasil.
    """
    
    keypoint_data = get_keypoint_data_from_list(keypoints_list)
    
    # Ambil keypoint vital
    try:
        RShoulder = keypoint_data[BICEP_KP_INDICES['RShoulder']]
        RHip = keypoint_data[BICEP_KP_INDICES['RHip']]
        LShoulder = keypoint_data[BICEP_KP_INDICES['LShoulder']]
        LHip = keypoint_data[BICEP_KP_INDICES['LHip']]
    except IndexError:
        return None

    # 1. Hitung Panjang Torso (untuk skala)
    dist_r = np.sqrt((RShoulder['x'] - RHip['x'])**2 + (RShoulder['y'] - RHip['y'])**2)
    dist_l = np.sqrt((LShoulder['x'] - LHip['x'])**2 + (LShoulder['y'] - LHip['y'])**2)
    avg_torso = (dist_r + dist_l) / 2
    
    if avg_torso == 0 or np.isnan(avg_torso): return None
        
    shoulder_x, shoulder_y = RShoulder['x'], RShoulder['y']
    feature_row = {}
    
    # 2. Normalisasi
    for name, index in BICEP_KP_INDICES.items():
        if index >= len(keypoint_data): return None
        kp = keypoint_data[index]
        
        # Pengecekan Visibilitas (Ambang batas rendah 0.2)
        if np.isnan(kp['x']) or kp['visibility'] < 0.2:
             return None 
             
        feature_row[f'{name}_X_NORM'] = (kp['x'] - shoulder_x) / avg_torso
        feature_row[f'{name}_Y_NORM'] = (kp['y'] - shoulder_y) / avg_torso

    # 3. Hitung Sudut menggunakan fungsi bicep_calculate_angle_dict (sama seperti rf_prediction.py)
    feature_row['R_ELBOW_ANGLE'] = bicep_calculate_angle_dict(
        keypoint_data, BICEP_KP_INDICES['RShoulder'], BICEP_KP_INDICES['RElbow'], BICEP_KP_INDICES['RWrist']
    )
    feature_row['R_SHOULDER_ANGLE'] = bicep_calculate_angle_dict(
        keypoint_data, BICEP_KP_INDICES['RHip'], BICEP_KP_INDICES['RShoulder'], BICEP_KP_INDICES['RElbow']
    )
    
    # 4. Filter dan Return
    features_df = pd.DataFrame([feature_row])[BICEP_FEATURE_COLUMNS]
    
    if features_df.isnull().values.any() or features_df.shape[1] != len(BICEP_FEATURE_COLUMNS):
        return None
        
    return features_df


# --- KONFIGURASI PATH MODEL ---
MODELS_CONFIG: Dict[str, Any] = {
 'pushup': {
  'model': None,
  'model_path': os.path.join('models', 'push_up', 'rf_model_data', 'random_forest_model.pkl'),
  'process_func': process_keypoints_for_pushup,
  'feature_columns': PUSHUP_FEATURE_COLUMNS
 },
 'bicep': {
  'model': None,
  'model_path': os.path.join('models', 'bicep_curl', 'rf_model_data', 'random_forest_model.pkl'),
  'process_func': process_keypoints_for_bicep, # <-- Sekarang menggunakan fungsi yang disinkronisasi
  'feature_columns': BICEP_FEATURE_COLUMNS
 }
}


# --- INISIALISASI FLASK, LOAD MODEL, dan ROUTING ---

app = Flask(__name__, 
   template_folder='ai_frontend/templates', 
   static_folder='ai_frontend/static')


def load_all_models():
 global MODELS_CONFIG
 
 for name, config in MODELS_CONFIG.items():
  try:
   if os.path.exists(config['model_path']):
    config['model'] = joblib.load(config['model_path'])
    print(f"✅ Model {name.title()} berhasil dimuat.")
   else:
    print(f"❌ PERINGATAN: Model {name.title()} tidak ditemukan.")
  except Exception as e:
   print(f"❌ GAGAL memuat model {name.title()}: {e}")

load_all_models()


# --- ROUTE UTAMA ---

@app.route('/')
def index(): return render_template('index.html')
@app.route('/bicep')
def bicep(): return render_template('bicep.html')
@app.route('/pushup')
def pushup(): return render_template('push_up.html')
@app.route('/tricep')
def tricep(): return render_template('tricep.html')
@app.route('/chestpress')
def chestpress(): return render_template('chestpress.html')
@app.route('/shoulderpress')
def shoulderpress(): return render_template('shoulderpress.html')
@app.route('/incline')
def incline(): return render_template('incline.html')


# --- ROUTE API UNTUK PREDIKSI AI (LOGIKA FEEDBACK FLEKSIBEL) ---
@app.route('/predict_frame', methods=['POST'])
def predict_frame():
 
    data = request.json
    keypoints_list = data.get('keypoints') 
    exercise_name = data.get('exercise', '').lower()
    
    config = MODELS_CONFIG.get(exercise_name)

    if not config or not config.get('model'):
        return jsonify({'error': f'Model untuk latihan "{exercise_name}" tidak ditemukan/dimuat.'}), 500
    
    if not keypoints_list:
        return jsonify({'prediction': 'INCORRECT', 'feedback': 'Gagal Mendeteksi Pose (Keypoints Hilang)', 'accuracy': 0.0}), 200

    # 1. Konversi Keypoint ke DataFrame Fitur
    features_df = config['process_func'](keypoints_list)
    
    if features_df is None:
        return jsonify({'prediction': 'INCORRECT', 'feedback': 'Keypoints Tidak Cukup Jelas atau Gagal Normalisasi', 'accuracy': 0.0}), 200

    # 2. Validasi & Urutkan Kolom
    required_cols = config['feature_columns']
    
    if features_df.shape[1] != len(required_cols):
        return jsonify({'prediction': 'INCORRECT', 'feedback': 'Kolom fitur tidak lengkap.', 'accuracy': 0.0}), 500
    
    features_df = features_df[required_cols]

    # Ambil Nilai Sudut
    elbow_angle, shoulder_angle = np.nan, np.nan
    
    try:
        # PENTING: Kolom ini sekarang terjamin ada karena BICEP_FEATURE_COLUMNS sama
        elbow_angle = features_df['R_ELBOW_ANGLE'].iloc[0]
        shoulder_angle = features_df['R_SHOULDER_ANGLE'].iloc[0]
    except KeyError:
        pass 

    # 3. Prediksi Menggunakan Model Random Forest
    try:
        model = config['model']
        prediction_proba = model.predict_proba(features_df)[0]
        correct_proba = prediction_proba[1] 
        
        prediction_code = model.predict(features_df)[0]
        prediction_label = 'CORRECT' if prediction_code == 1 else 'INCORRECT'
        
        accuracy = correct_proba * 100
        
    except Exception:
        return jsonify({'prediction': 'INCORRECT', 'feedback': 'Error Internal Model', 'accuracy': 0.0}), 500


    # 4. Logika Feedback Fleksibel (Hanya Fokus pada Gerakan Utama Siku)
    
    feedback = ""
    is_form_incorrect_by_rule = False

    if exercise_name == 'bicep' and not np.isnan(elbow_angle):
        
        # --- Parameter Fleksibel Bicep Curl ---
        # Rule A: Anti-Swinging (SANGAT Longgar)
        SHOULDER_SWING_MIN = 70 
        SHOULDER_SWING_MAX = 120
        
        # Rule C: Rentang Gerak Siku (SANGAT Longgar)
        ELBOW_MAX_FLEXION = 35  # Asal ada tekukan (di atas)
        ELBOW_MIN_EXTENSION = 170 # Asal ada pelurusan (di bawah)
        
        # Rule B (Anti-Lateral Movement) DINONAKTIFKAN TOTAL

        # Rule A: Periksa Swinging (gerakan bahu)
        if shoulder_angle < SHOULDER_SWING_MIN or shoulder_angle > SHOULDER_SWING_MAX:
            feedback = "⚠️ Gerakan bahu terlalu ekstrem. Fokus pada siku."
            is_form_incorrect_by_rule = True
            
        # Rule C: Periksa Rentang Gerak Siku (Deteksi Gerakan Naik-Turun)
        elif elbow_angle > ELBOW_MIN_EXTENSION:
            # Siku sangat lurus (di bawah)
            feedback = "Gerakan Siku Terdeteksi: Ekstensi Bawah."
        elif elbow_angle < ELBOW_MAX_FLEXION:
            # Siku sangat menekuk (di atas)
            feedback = "Gerakan Siku Terdeteksi: Fleksi Atas."
        else:
            # Jika berada di antara rentang fleksibel, utamakan prediksi AI
            if prediction_label == 'CORRECT':
                feedback = "Gerakan naik turun terdeteksi (OK)."
            else:
                feedback = "Form perlu ditingkatkan (Prediksi AI)."

        # Jika ada kesalahan ekstrem pada Rule A, set akurasi rendah.
        if is_form_incorrect_by_rule:
            prediction_label = 'INCORRECT'
            accuracy = min(accuracy, 40.0)
    
    
    # 5. Jika tidak ada aturan spesifik yang berlaku atau bukan Bicep Curl, gunakan prediksi AI
    if not feedback or exercise_name != 'bicep':
        if prediction_label == 'CORRECT' and accuracy > 80:
            feedback = "Gerakan terdeteksi benar."
        elif accuracy > 50:
            feedback = "Form perlu ditingkatkan."
        else:
            feedback = "Form sangat tidak tepat."

    # 6. Kirim Hasil
    return jsonify({
        'prediction': prediction_label,
        'feedback': feedback,
        'accuracy': accuracy,
        'exercise': exercise_name
    })


if __name__ == '__main__':
 app.run(debug=True)