import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import os
import time

# --- KONFIGURASI PUSH-UP (HARUS SAMA DENGAN EXTRACTOR) ---
KP_INDICES = {
    'RShoulder': 12, 'RElbow': 14, 'RWrist': 16,    
    'RHip': 24,      'RKnee': 26, 'RAnkle': 28,
    'LShoulder': 11, 'LHip': 23,  'LAnkle': 27,
}

OUTPUT_MODEL_DIR = 'rf_model_data'
MODEL_FILENAME = 'random_forest_model.pkl' # Asumsi nama model sama
MODEL_PATH = os.path.join(OUTPUT_MODEL_DIR, MODEL_FILENAME)
FEATURE_COLUMNS = (
    [f'{name}_{axis}_NORM' for name in KP_INDICES.keys() for axis in ['X', 'Y']] +
    ['R_ELBOW_ANGLE', 'TORSO_ANGLE', 'R_SHOULDER_TUCK_ANGLE']
)

# Ambang Batas Gerakan Push-up (Nilai Perkiraan Ideal)
MAX_ELBOW_ANGLE_UP = 160  # Posisi Puncak Atas (Siku Lurus)
MIN_ELBOW_ANGLE_DOWN = 100 # Posisi Puncak Bawah (90 derajat atau lebih rendah)
IDEAL_TORSO_ANGLE = 175    # Sudut Torso Sempurna (garis lurus)
TORSO_TOLERANCE = 10       # Toleransi garis lurus (175 +/- 10)
IDEAL_TUCK_ANGLE = 45      # Siku masuk (nilai rendah = siku masuk)

# Inisialisasi MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 

# --- STATE MACHINE ---
STATE_UP = 0
STATE_DOWN_MOVEMENT = 1
STATE_DOWN_POS = 2
STATE_UP_MOVEMENT = 3

current_state = STATE_UP
rep_count = 0
current_rep_scores = [] 
last_rep_accuracy = 0.0

# --- FUNGSI BANTUAN EKSTRAKSI FITUR (HARUS SAMA PERSIS DENGAN EXTRACTOR) ---

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
    # Logika normalisasi yang sama persis
    RShoulder = keypoint_data[KP_INDICES['RShoulder']]
    RHip = keypoint_data[KP_INDICES['RHip']]
    LShoulder = keypoint_data[KP_INDICES['LShoulder']]
    LHip = keypoint_data[KP_INDICES['LHip']]
    
    dist_r = np.sqrt((RShoulder['x'] - RHip['x'])**2 + (RShoulder['y'] - RHip['y'])**2)
    dist_l = np.sqrt((LShoulder['x'] - LHip['x'])**2 + (LShoulder['y'] - LHip['y'])**2)
    avg_torso = (dist_r + dist_l) / 2
    
    if avg_torso == 0 or np.isnan(avg_torso): return None
        
    shoulder_x, shoulder_y = RShoulder['x'], RShoulder['y']
    feature_row = {}
    
    for name, index in KP_INDICES.items():
        kp = keypoint_data[index]
        feature_row[f'{name}_X_NORM'] = (kp['x'] - shoulder_x) / avg_torso
        feature_row[f'{name}_Y_NORM'] = (kp['y'] - shoulder_y) / avg_torso

    # Fitur Sudut
    feature_row['R_ELBOW_ANGLE'] = calculate_angle(keypoint_data, KP_INDICES['RShoulder'], KP_INDICES['RElbow'], KP_INDICES['RWrist'])
    feature_row['TORSO_ANGLE'] = calculate_angle(keypoint_data, KP_INDICES['RShoulder'], KP_INDICES['RHip'], KP_INDICES['RKnee'])
    feature_row['R_SHOULDER_TUCK_ANGLE'] = calculate_angle(keypoint_data, KP_INDICES['RElbow'], KP_INDICES['RShoulder'], KP_INDICES['RHip'])
    
    df_features = pd.DataFrame([feature_row])[FEATURE_COLUMNS]
    return df_features

# --- LOGIKA PENGHITUNG REPETISI PUSH-UP ---

def update_rep_state(elbow_angle):
    global current_state, rep_count, current_rep_scores, last_rep_accuracy
    
    if np.isnan(elbow_angle): return

    if current_state == STATE_UP and elbow_angle < MAX_ELBOW_ANGLE_UP:
        # Mulai turun (dari posisi lengan lurus)
        current_state = STATE_DOWN_MOVEMENT
        
    elif current_state == STATE_DOWN_MOVEMENT and elbow_angle < MIN_ELBOW_ANGLE_DOWN:
        # Mencapai puncak bawah
        current_state = STATE_DOWN_POS
        
    elif current_state == STATE_DOWN_POS and elbow_angle > MIN_ELBOW_ANGLE_DOWN:
        # Mulai naik
        current_state = STATE_UP_MOVEMENT
        
    elif current_state == STATE_UP_MOVEMENT and elbow_angle > MAX_ELBOW_ANGLE_UP:
        # Rep selesai! Kembali ke posisi atas.
        rep_count += 1
        
        # Hitung Akurasi Rep Terakhir
        if current_rep_scores:
           last_rep_accuracy = np.mean(current_rep_scores) * 100
        else:
           last_rep_accuracy = 0.0
           
        current_rep_scores = [] # Reset skor
        current_state = STATE_UP
        
# --- LOGIKA FEEDBACK PUSH-UP SPESIFIK ---

def get_feedback(features_df, model, label_map):
    
    prediction_code = model.predict(features_df)[0]
    prediction_label = label_map.get(prediction_code, 'UNKNOWN')
    
    # Ambil nilai fitur
    elbow_angle = features_df['R_ELBOW_ANGLE'].iloc[0]
    torso_angle = features_df['TORSO_ANGLE'].iloc[0]
    tuck_angle = features_df['R_SHOULDER_TUCK_ANGLE'].iloc[0]
    
    # 1. Cek Torso (Form Sempurna)
    if torso_angle < (IDEAL_TORSO_ANGLE - TORSO_TOLERANCE):
         return "❌ Jaga Torso! Pinggul terlalu rendah (kendur). Kencangkan inti!", (0, 0, 255) # Merah

    # 2. Cek Torso (Form Sempurna)
    if torso_angle > (IDEAL_TORSO_ANGLE + TORSO_TOLERANCE):
         return "❌ Jaga Torso! Pinggul terlalu tinggi (gunung). Luruskan!", (0, 0, 255) # Merah

    # 3. Cek Kedalaman (Elbow Angle)
    if current_state == STATE_DOWN_MOVEMENT and elbow_angle > 120:
         return "Gerakan Turun: Dorong dada lebih dekat ke lantai!", (0, 165, 255) # Oranye
    
    # 4. Cek Siku (Tuck Angle)
    if tuck_angle > 80:
         return "❌ Siku Melebar! Tuck siku Anda mendekati tubuh.", (0, 0, 255) # Merah

    # Prediksi ML
    if prediction_label == 'CORRECT':
        return "Pertahankan form! Sempurna.", (0, 255, 0) # Hijau
    
    return "Form tidak tepat. Cek posisi siku dan bahu.", (0, 0, 255) # Merah


def run_live_coach():
    global rep_count, current_rep_scores, last_rep_accuracy
    
    # Muat Model (asumsi model_pkl ada)
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model tidak ditemukan di {MODEL_PATH}.")
        print("Pastikan Anda menjalankan 'rf_feature_extractor_pushup.py' dan 'rf_trainer.py' terlebih dahulu.")
        return

    rf_model = joblib.load(MODEL_PATH)
    print(f"✅ Model Random Forest berhasil dimuat dari {MODEL_PATH}")

    # Solusi Kamera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("ERROR: Tidak dapat membuka kamera.")
            return

    label_map = {0: 'INCORRECT', 1: 'CORRECT'}
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            feedback_text = "Mencari Pose..."
            prediction_text = ""
            box_color = (100, 100, 100)
            
            try:
                if results.pose_landmarks:
                    keypoint_data = [
                        {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
                        for lm in results.pose_landmarks.landmark
                    ]
                    
                    features_df = extract_and_normalize_features(keypoint_data)
                    
                    if features_df is not None and not features_df.isnull().values.any():
                        
                        elbow_angle = features_df['R_ELBOW_ANGLE'].iloc[0]
                        
                        # 1. Update Repetition State
                        update_rep_state(elbow_angle)

                        # 2. Prediksi dan Feedback
                        prediction_proba = rf_model.predict_proba(features_df)[0]
                        correct_proba = prediction_proba[1] 

                        feedback_text, box_color = get_feedback(features_df, rf_model, label_map)
                        
                        if current_state != STATE_UP:
                            current_rep_scores.append(correct_proba)
                        
                        prediction_text = f"Ketepatan Form: {correct_proba*100:.1f}%"
                        
                    # Gambar Keypoint
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=box_color, thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=box_color, thickness=2, circle_radius=2)
                    )

            except Exception as e:
                feedback_text = "ERROR PEMROSESAN"

            # --- DISPLAY UTAMA ---
            
            # Rep Counter
            cv2.putText(image, f"REPS: {rep_count}", (image.shape[1] - 200, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
            
            # Feedback (Baris Atas)
            cv2.putText(image, feedback_text, (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2, cv2.LINE_AA)

            # Persentase Ketepatan (Baris Bawah)
            cv2.putText(image, prediction_text, (10, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Akurasi Rep Terakhir
            cv2.putText(image, f"LAST REP ACCURACY: {last_rep_accuracy:.1f}%", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Push-up Live Coach', image)

            # Stop Camera (ESC atau q)
            key = cv2.waitKey(5) & 0xFF
            if key == 27 or key == ord('q'): 
                break 

    cap.release()
    cv2.destroyAllWindows()
    print("Prediksi Selesai.")

if __name__ == '__main__':
    run_live_coach()