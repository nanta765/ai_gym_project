import cv2
import mediapipe as mp
import pandas as pd
import os
import time
import numpy as np

# --- INISIALISASI MEDIAPIPE ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# Pose detector diinisialisasi di fungsi utama
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- SOLUSI KAMERA: Coba beberapa indeks ---
# Coba indeks 0, lalu 1 jika 0 gagal. Anda bisa menambahkan indeks lain jika perlu.
CAMERA_INDICES = [0, 1] 

# --- FUNGSI EKSTRAKSI DATA (DISEDERHANAKAN & KONSISTEN) ---

def extract_keypoints_to_dataframe(results):
    """
    Mengubah hasil MediaPipe menjadi DataFrame yang hanya berisi
    kolom Landmark[Index]_[Axis] (Data mentah) untuk konsistensi.
    """
    if not results.pose_landmarks:
        return pd.DataFrame()
        
    landmarks = results.pose_landmarks.landmark
    data = {}
    
    # Ekstraksi SEMUA 33 keypoint (x, y, z, visibility)
    for i, landmark in enumerate(landmarks):
        # Format kolom: Landmark[Index]_[Axis]
        data[f'Landmark{i}_X'] = landmark.x
        data[f'Landmark{i}_Y'] = landmark.y
        data[f'Landmark{i}_Z'] = landmark.z
        data[f'Landmark{i}_V'] = landmark.visibility
        
    # CATATAN: Kami MENGHAPUS keypoint kustom seperti RShoulderY17, HipY1, dll.,
    # karena tugas perhitungan fitur sudah dipindahkan ke rf_feature_extractor.py.
    
    return pd.DataFrame([data])


# --- FUNGSI UTAMA UNTUK MENJALANKAN KAMERA & DETEKSI POSE ---

def run_pose_detector(camera_index):
    """Mencoba membuka kamera pada indeks tertentu dan menampilkan deteksi pose."""
    
    # 1. Coba buka kamera dengan backend DSHOW (sering membantu di Windows)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"❌ GAGAL: Tidak dapat membuka kamera pada indeks {camera_index}.")
        return False

    print(f"✅ BERHASIL: Kamera dibuka pada indeks {camera_index}. Tekan 'q' untuk keluar.")
    
    # 2. Main Loop
    with pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            status_text = "Mencari Pose..."
            
            if results.pose_landmarks:
                # Gambar Keypoint
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                status_text = "POSE DITEMUKAN"
            
            # Tampilkan status
            cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Pose Detector Test', image)

            if cv2.waitKey(5) & 0xFF == ord('q'): break # Tekan 'q' untuk berhenti

    # 3. Cleanup
    cap.release()
    cv2.destroyAllWindows()
    return True # Berhasil dibuka

# --- BLOK UTAMA EKSEKUSI ---
if __name__ == '__main__':
    print("--- MEMULAI UJI COBA KAMERA ---")
    
    success = False
    # Loop melalui indeks kamera yang dikonfigurasi
    for index in CAMERA_INDICES:
        if run_pose_detector(index):
            success = True
            break
            
    if not success:
        print("\n❌ GAGAL TOTAL: Tidak ada kamera yang berhasil dibuka.")
        print("Solusi: Coba tambahkan indeks kamera lain (misalnya 2) ke list CAMERA_INDICES.")