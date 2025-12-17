import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path, output_path):
    print(f"Memulai proses data dari: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} tidak ditemukan!")
        return

    df = pd.read_csv(input_path)
    print(f"Data awal dimuat. Ukuran: {df.shape}")

    target_col = 'Potability'
    
    if target_col not in df.columns:
        print(f"Error: Kolom target '{target_col}' tidak ada di dataset.")
        return

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    feature_names = X.columns

    print("Mengisi missing values (Median Imputation)...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    print("Melakukan Scaling (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    X_final = pd.DataFrame(X_scaled, columns=feature_names)
    final_df = pd.concat([X_final, y], axis=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    final_df.to_csv(output_path, index=False)
    print(f"Data bersih disimpan di: {output_path}")
    print(f"Ukuran akhir: {final_df.shape}")

if __name__ == "__main__":
    # Pengaturan Path (Asumsi script dijalankan dari root repository)
    # Input: data mentah di folder data/
    input_csv = 'data/water_potability.csv'
    
    # Output: data bersih di folder data/
    output_csv = 'data/water_potability_processed.csv'
    
    # Jalankan fungsi
    preprocess_data(input_csv, output_csv)