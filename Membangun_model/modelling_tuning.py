import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DAGSHUB_USERNAME = "moenalyze"
DAGSHUB_REPO_NAME = "Eksperimen_SML_Khatama"

def train_and_tune():
    print("Memulai proses training dengan Hyperparameter Tuning...")

    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    
    mlflow.set_experiment("Water_Quality_Experiment_Tuning")

    data_path = 'data/water_potability_processed.csv'
    if not os.path.exists(data_path):
        data_path = '../data/water_potability_processed.csv'
        
    print(f"Loading data dari: {data_path}")
    df = pd.read_csv(data_path)

    X = df.drop('Potability', axis=1)
    y = df['Potability']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }

    rf = RandomForestClassifier(random_state=42)
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)

    with mlflow.start_run(run_name="RandomForest_GridSearch"):
        print("Sedang melakukan GridSearch...")
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Model Terbaik ditemukan: {best_params}")

        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Metrik: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
        
        mlflow.log_params(best_params)

        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        mlflow.sklearn.log_model(best_model, "best_model")
        
        # Artefak
        plt.figure(figsize=(6,5))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png") 
        print("Confusion Matrix tersimpan.")

        # Artefak
        importances = best_model.feature_importances_
        feature_names = X.columns
        indices = importances.argsort()[::-1]

        plt.figure(figsize=(10,6))
        plt.title("Feature Importances")
        plt.bar(range(X.shape[1]), importances[indices], align="center")
        plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        print("Feature Importance tersimpan.")
        
        os.remove("confusion_matrix.png")
        os.remove("feature_importance.png")

if __name__ == "__main__":
    train_and_tune()