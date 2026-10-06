import cv2
import mediapipe as mp
import csv
import os
import glob
import numpy as np

# --- KONFIGURASI ---
# Ganti TARGET_FOLDERS jika Anda menggunakan nama folder yang berbeda
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
    # Keypoint 0 sampai 32
    for i in range(KEYPOINT_COUNT):
        landmarks_names += [f'Landmark{i}_X', f'Landmark{i}_Y', f'Landmark{i}_Z', f'Landmark{i}_V']
    return landmarks_names

def process_file(file_path, pose_processor):
    """Memproses file media (video atau gambar) dan menghasilkan data keypoint."""
    
    # Tentukan path output CSV
    base_name = os.path.basename(file_path)
    file_name, file_ext = os.path.splitext(base_name)
    # File CSV akan bernama [nama_video]_processed.csv
    output_csv_path = os.path.join(os.path.dirname(file_path), f"{file_name}_processed.csv")
    
    data_rows = []
    header = get_csv_header()

    is_video = file_ext.lower() in ['.mp4', '.mov', '.avi']

    if is_video:
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
            # image.flags.writeable = True # Tidak perlu ditampilkan
            
            if results.pose_landmarks:
                row = []
                for landmark in results.pose_landmarks.landmark:
                    row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                data_rows.append(row)

        cap.release()
        cv2.destroyAllWindows()
        
    elif file_ext.lower() in ['.jpg', '.jpeg', '.png']:
        # --- PROSES GAMBAR ---
        print("   -> Processing Image (1 frame)...")
        image = cv2.imread(file_path)
        if image is None:
            print("   ⚠️ Failed to load image.")
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
        print(f"   ✅ SUCCESS: {len(data_rows)} frames saved to {output_csv_path}")
    else:
        print(f"   ❌ FAILED: No pose detected or file skipped.")


def run_converter():
    print("--- STARTING MEDIA TO CSV CONVERTER FOR PUSH-UP ---")
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for folder in TARGET_FOLDERS:
            # Mengatur path relatif ke folder push_up saat ini
            folder_path = os.path.join(os.getcwd(), folder) 
            print(f"\nProcessing folder: {folder_path}")
            
            if not os.path.exists(folder_path):
                 print(f"   ⚠️ Folder tidak ditemukan: {folder_path}. Lewati.")
                 continue

            # Cari semua file media
            media_files = []
            for ext in VIDEO_EXTENSIONS + IMAGE_EXTENSIONS:
                media_files.extend(glob.glob(os.path.join(folder_path, ext)))
            
            if not media_files:
                print(f"   ⚠️ Warning: No media files found in {folder}. Skipping.")
                continue

            for file_path in media_files:
                print(f"File: {os.path.basename(file_path)}")
                # Menggunakan fungsi process_file
                process_file(file_path, pose)

    print("\n--- CONVERSION COMPLETE ---")
    print("Sekarang lanjutkan ke ekstraksi fitur Push-up.")
    print("Jalankan: 'python rf_feature_extractor_pushup.py'")

if __name__ == '__main__':
    run_converter()