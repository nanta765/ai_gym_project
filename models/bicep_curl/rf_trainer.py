import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib # Untuk menyimpan model
import os

# --- KONFIGURASI ---
OUTPUT_MODEL_DIR = 'rf_model_data'
OUTPUT_DATA_FILE = os.path.join(OUTPUT_MODEL_DIR, 'rf_features.csv')
MODEL_FILENAME = 'random_forest_model.pkl'
MODEL_PATH = os.path.join(OUTPUT_MODEL_DIR, MODEL_FILENAME)

def train_and_save_rf():
    if not os.path.exists(OUTPUT_DATA_FILE):
        print(f"ERROR: File fitur tidak ditemukan di {OUTPUT_DATA_FILE}. Jalankan rf_feature_extractor.py terlebih dahulu.")
        return

    # Muat data fitur
    df = pd.read_csv(OUTPUT_DATA_FILE)
    
    # Pisahkan Fitur (X) dan Label (y)
    X = df.drop(columns=['LABEL'])
    y = df['LABEL']

    # Konversi label string ke integer (0 atau 1)
    y_encoded = y.astype('category').cat.codes
    
    # Pisahkan data Training dan Testing (20% untuk pengujian)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"Total Frames: {len(df)}")
    print(f"Training Frames: {len(X_train)} | Testing Frames: {len(X_test)}")
    print("Memulai Training Random Forest...")
    
    # Inisialisasi dan Latih Model Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,      # Jumlah pohon
        max_depth=10,          # Kedalaman maksimum pohon
        random_state=42,
        class_weight='balanced' # Penting jika jumlah form_benar dan form_salah tidak seimbang
    )
    
    rf_model.fit(X_train, y_train)
    
    print("Training Selesai.")

    # Evaluasi Model
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Konversi label integer kembali ke string untuk laporan
    target_names = df['LABEL'].unique()
    
    print("\n--- Hasil Evaluasi pada Test Set (20%) ---")
    print(f"Akurasi Total: {accuracy:.4f}")
    print("\nLaporan Klasifikasi:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Simpan Model
    joblib.dump(rf_model, MODEL_PATH)
    print(f"\nModel Random Forest disimpan ke: {MODEL_PATH}")

if __name__ == '__main__':
    train_and_save_rf()