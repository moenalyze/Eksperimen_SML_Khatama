import pandas as pd
import mlflow
import dagshub
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_PASSWORD = os.getenv("DAGSHUB_TOKEN") 
DAGSHUB_REPO_NAME = "Eksperimen_SML_Khatama"

def train_basic():
    print("Memulai Training (Autolog)...")
    
    if DAGSHUB_PASSWORD:
        dagshub.auth.add_app_token(DAGSHUB_PASSWORD)

    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow")
    # mlflow.set_experiment("Water_Quality_Basic")
    
    file_name = 'water_potability_processed.csv'

    if os.path.exists(os.path.join('Membangun_model', file_name)):
        data_path = os.path.join('Membangun_model', file_name)
    elif os.path.exists(file_name):
        data_path = file_name
    else:
        data_path = os.path.join('data', file_name)
        
    print(f"Membaca dataset dari: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: File {file_name} tidak ditemukan di mana-mana!")
        return
    
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.autolog()

    with mlflow.start_run(run_name="Basic_RandomForest"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Training Selesai. Akurasi: {acc}")

if __name__ == "__main__":
    train_basic()