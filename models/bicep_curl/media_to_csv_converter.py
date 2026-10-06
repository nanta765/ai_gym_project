import cv2
import mediapipe as mp
import csv
import os
import glob
import numpy as np

# --- KONFIGURASI ---
TARGET_FOLDERS = ['form_benar', 'form_salah']
VIDEO_EXTENSIONS = ['*.mp4', '*.mov', '*.avi']
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png']
KEYPOINT_COUNT = 33 # Jumlah landmark MediaPipe

# Inisialisasi MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def get_csv_header():
    """Membuat header CSV (Landmark0_X, Landmark0_Y, ..., Landmark32_V)"""
    landmarks_names = []
    for i in range(KEYPOINT_COUNT):
        landmarks_names += [f'Landmark{i}_X', f'Landmark{i}_Y', f'Landmark{i}_Z', f'Landmark{i}_V']
    return landmarks_names

def process_file(file_path, pose_processor):
    """Memproses file media (video atau gambar) dan menghasilkan data keypoint."""
    
    # Tentukan path output CSV
    base_name = os.path.basename(file_path)
    file_name, file_ext = os.path.splitext(base_name)
    output_csv_path = os.path.join(os.path.dirname(file_path), f"{file_name}_processed.csv")
    
    data_rows = []
    
    # Header CSV
    header = get_csv_header()

    if file_ext.lower() in ['.mp4', '.mov', '.avi']:
        # --- PROSES VIDEO ---
        cap = cv2.VideoCapture(file_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"   -> Processing Video ({frame_count} frames)...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose_processor.process(image)

            if results.pose_landmarks:
                row = []
                for landmark in results.pose_landmarks.landmark:
                    row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                data_rows.append(row)
            
            # Opsional: Tampilkan visualisasi (memperlambat proses, nonaktifkan saat produksi)
            # image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # cv2.imshow('Video Processing', image)
            # if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()
        
    elif file_ext.lower() in ['.jpg', '.jpeg', '.png']:
        # --- PROSES GAMBAR ---
        print("   -> Processing Image (1 frame)...")
        image = cv2.imread(file_path)
        if image is None:
            print("Failed to load image.")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose_processor.process(image)

        if results.pose_landmarks:
            row = []
            for landmark in results.pose_landmarks.landmark:
                row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
            data_rows.append(row)
        
    # Tulis hasil ke CSV
    if data_rows:
        with open(output_csv_path, mode='w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(header)
            csv_writer.writerows(data_rows)
        print(f"SUCCESS: {len(data_rows)} frames saved to {output_csv_path}")
    else:
        print(f"FAILED: No pose detected or file skipped.")


def run_converter():
    print("--- STARTING MEDIA TO CSV CONVERTER ---")
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for folder in TARGET_FOLDERS:
            print(f"\nProcessing folder: {folder}")
            
            # Cari semua file media
            media_files = []
            for ext in VIDEO_EXTENSIONS + IMAGE_EXTENSIONS:
                media_files.extend(glob.glob(os.path.join(folder, ext)))
            
            if not media_files:
                print(f"Warning: No media files found in {folder}. Skipping.")
                continue

            for file_path in media_files:
                print(f"File: {os.path.basename(file_path)}")
                process_file(file_path, pose)

    print("\n--- CONVERSION COMPLETE ---")
    print("Semua file CSV pose telah dibuat. Sekarang lanjutkan ke ekstraksi fitur.")
    print("Jalankan: 'python rf_feature_extractor.py'")

if __name__ == '__main__':
    run_converter()