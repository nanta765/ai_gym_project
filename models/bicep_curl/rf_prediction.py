# File: rf_prediction.py (BICEP CURL LIVE COACH - VERSI FLEKSIBEL)

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import os

# --- KONFIGURASI MODEL & FITUR ---
KP_INDICES = {
    'RShoulder': 12, 'RElbow': 14, 'RWrist': 16,  
    'RHip': 24,   'LHip': 23, 'LShoulder': 11, 
}
OUTPUT_MODEL_DIR = 'rf_model_data'
MODEL_FILENAME = 'random_forest_model.pkl'
MODEL_PATH = os.path.join(OUTPUT_MODEL_DIR, MODEL_FILENAME)
FEATURE_COLUMNS = (
    [f'{name}_{axis}_NORM' for name in KP_INDICES.keys() for axis in ['X', 'Y']] +
    ['R_ELBOW_ANGLE', 'R_SHOULDER_ANGLE']
)
# Ambang batas Sudut Siku (FLEKSIBEL - Gerakan Naik Turun Saja)
MIN_ELBOW_ANGLE_CURL = 35 
MAX_ELBOW_ANGLE_CURL = 170 

# Inisialisasi MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- STATE MACHINE ---
STATE_START = 0
STATE_DOWN = 1
STATE_UP = 2
current_state = STATE_START
rep_count = 0
current_rep_scores = [] 

# --- FUNGSI BANTUAN EKSTRAKSI FITUR ---

def calculate_angle(keypoint_data, p1_idx, p2_idx, p3_idx):
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

def extract_and_normalize_features(keypoint_data):
    # Mengubah logika ini agar sesuai dengan normalize_keypoints di rf_feature_extractor_bicep.py
    
    RShoulder = keypoint_data[KP_INDICES['RShoulder']]
    RHip = keypoint_data[KP_INDICES['RHip']]
    LShoulder = keypoint_data[KP_INDICES['LShoulder']]
    LHip = keypoint_data[KP_INDICES['LHip']]
    
    # 1. Hitung Panjang Torso
    dist_r = np.sqrt((RShoulder['x'] - RHip['x'])**2 + (RShoulder['y'] - RHip['y'])**2)
    dist_l = np.sqrt((LShoulder['x'] - LHip['x'])**2 + (LShoulder['y'] - LHip['y'])**2)
    avg_torso = (dist_r + dist_l) / 2
    
    if avg_torso == 0 or np.isnan(avg_torso): return None
        
    shoulder_x, shoulder_y = RShoulder['x'], RShoulder['y']
    feature_row = {}
    
    # 2. Normalisasi
    for name, index in KP_INDICES.items():
        kp = keypoint_data[index]
        feature_row[f'{name}_X_NORM'] = (kp['x'] - shoulder_x) / avg_torso
        feature_row[f'{name}_Y_NORM'] = (kp['y'] - shoulder_y) / avg_torso

    # 3. Hitung Sudut
    feature_row['R_ELBOW_ANGLE'] = calculate_angle(keypoint_data, KP_INDICES['RShoulder'], KP_INDICES['RElbow'], KP_INDICES['RWrist'])
    feature_row['R_SHOULDER_ANGLE'] = calculate_angle(keypoint_data, KP_INDICES['RHip'], KP_INDICES['RShoulder'], KP_INDICES['RElbow'])
    
    df_features = pd.DataFrame([feature_row])[FEATURE_COLUMNS]
    return df_features


# --- FUNGSI STATE MACHINE DAN HITUNG REPS (MENGGUNAKAN AMBANG BATAS FLEKSIBEL) ---
def update_rep_state(elbow_angle):
    global current_state, rep_count
    
    if np.isnan(elbow_angle):
        return
    
    # STATE_START: Siap memulai di posisi bawah, atau di tengah (asal tidak di ekstrem)
    if current_state == STATE_START and elbow_angle < MAX_ELBOW_ANGLE_CURL and elbow_angle > MIN_ELBOW_ANGLE_CURL:
        current_state = STATE_DOWN 
        
    # STATE_DOWN: Bergerak ke atas (menekuk siku)
    elif current_state == STATE_DOWN and elbow_angle < MIN_ELBOW_ANGLE_CURL:
        current_state = STATE_UP 
        
    # STATE_UP: Bergerak ke bawah (meluruskan siku), Rep selesai
    elif current_state == STATE_UP and elbow_angle > MAX_ELBOW_ANGLE_CURL:
        rep_count += 1
        current_state = STATE_START
        global current_rep_scores
        current_rep_scores = []


# --- FUNGSI LOGIKA FEEDBACK SPESIFIK (DILONGGARKAN) ---

def get_feedback(features_df, model, label_map):
    global current_state
    
    prediction_code = model.predict(features_df)[0]
    prediction_label = label_map.get(prediction_code, 'UNKNOWN')
    
    elbow_angle = features_df['R_ELBOW_ANGLE'].iloc[0]
    shoulder_angle = features_df['R_SHOULDER_ANGLE'].iloc[0]
    
    # Aturan Swinging (Sangat longgar, 70-110 derajat)
    if shoulder_angle > 110 or shoulder_angle < 70:
        return "⚠️ Gerakan bahu terlalu ekstrem! (Swinging/angkat bahu).", (0, 165, 255) # Oranye

    # Jika model memprediksi CORRECT, utamakan feedback positif
    if prediction_label == 'CORRECT':
        if current_state == STATE_DOWN:
            return "Gerakan ke atas (Tekuk Siku) OK.", (0, 255, 0) # Hijau
        elif current_state == STATE_UP:
            return "Gerakan ke bawah (Lurus Siku) OK.", (0, 255, 0) # Hijau
        else:
             return "Form Terdeteksi Benar.", (0, 255, 0)

    # Jika model memprediksi INCORRECT, berikan feedback berdasarkan rentang gerakan
    
    # Aturan Rentang Gerak (Longgar)
    # Ini adalah "hard coded" feedback, terlepas dari state machine
    if elbow_angle < 45: 
        return "Angkat lebih tinggi! Siku perlu menekuk maksimal.", (0, 165, 255)
    
    if elbow_angle > 150:
        return "Luruskan lengan sepenuhnya!", (0, 165, 255) 

    # Default jika model INCORRECT dan tidak ada aturan rentang gerak yang spesifik
    return "Form tidak tepat. Cek posisi siku dan bahu.", (0, 0, 255) # Merah


def run_live_coach():
    """Fungsi utama untuk prediksi, hitung rep, dan feedback."""
    global rep_count, current_rep_scores
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model tidak ditemukan di {MODEL_PATH}.")
        return

    rf_model = joblib.load(MODEL_PATH)
    print(f"✅ Model Random Forest berhasil dimuat dari {MODEL_PATH}")

    # Menggunakan CAP_DSHOW untuk Windows agar lebih cepat
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): 
        print("ERROR: Tidak dapat membuka kamera.")
        return

    label_map = {0: 'INCORRECT', 1: 'CORRECT'}
    last_rep_accuracy = 0.0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            prediction_text = ""
            feedback_text = "Siap Mulai..."
            box_color = (100, 100, 100) # Abu-abu
            
            try:
                if results.pose_landmarks:
                    keypoint_data = [
                        {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
                        for lm in results.pose_landmarks.landmark
                    ]
                    
                    features_df = extract_and_normalize_features(keypoint_data)
                    
                    if features_df is not None and not features_df.isnull().values.any():
                        
                        elbow_angle = features_df['R_ELBOW_ANGLE'].iloc[0]
                        
                        # Cek apakah rep baru selesai *sebelum* update state
                        is_rep_complete = (current_state == STATE_UP and elbow_angle > MAX_ELBOW_ANGLE_CURL)
                        
                        # Update state machine
                        update_rep_state(elbow_angle)

                        if is_rep_complete:
                            # Hitung akurasi rep yang baru selesai
                            if current_rep_scores:
                                last_rep_accuracy = np.mean(current_rep_scores) * 100
                            else:
                                last_rep_accuracy = 0.0
                            current_rep_scores = []
                        
                        
                        prediction_proba = rf_model.predict_proba(features_df)[0]
                        correct_proba = prediction_proba[1] 

                        feedback_text, box_color = get_feedback(features_df, rf_model, label_map)
                        
                        if current_state != STATE_START:
                            # Rekam skor akurasi selama rep berlangsung
                            current_rep_scores.append(correct_proba)
                        
                        prediction_text = f"Ketepatan Form: {correct_proba*100:.1f}%"
                        
                    # Gambar Keypoint
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                             mp_drawing.DrawingSpec(color=box_color, thickness=2, circle_radius=2),
                                             mp_drawing.DrawingSpec(color=box_color, thickness=2, circle_radius=2)
                    )

            except Exception as e:
                feedback_text = f"ERROR PEMROSESAN: {e}"

            # --- DISPLAY UTAMA ---
            cv2.putText(image, f"REPS: {rep_count}", (image.shape[1] - 200, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
            
            cv2.putText(image, feedback_text, (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2, cv2.LINE_AA)

            cv2.putText(image, prediction_text, (10, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            if last_rep_accuracy > 0:
                cv2.putText(image, f"LAST REP ACCURACY: {last_rep_accuracy:.1f}%", (10, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Bicep Curl Live Coach', image)

            if cv2.waitKey(5) & 0xFF == 27: 
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_live_coach()