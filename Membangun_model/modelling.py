import pandas as pd
import mlflow
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DAGSHUB_USERNAME = "moenalyze"
DAGSHUB_REPO_NAME = "Eksperimen_SML_Khatama"

def train_basic():
    print("Memulai Training (Autolog)...")

    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow")
    mlflow.set_experiment("Water_Quality_Basic")

    df = pd.read_csv('water_potability_processed.csv')
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