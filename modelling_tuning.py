import os
import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
import joblib

from dotenv import load_dotenv
from dagshub import dagshub_logger
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# MLflow + DagsHub Integration
# -----------------------------
load_dotenv()

# Ambil credentials dari .env
username = os.getenv("DAGSHUB_USERNAME")
token = os.getenv("DAGSHUB_TOKEN")
repo = "labibaadinda/Membangun_model"

# Validasi terlebih dahulu sebelum lanjut
assert username is not None, "DAGSHUB_USERNAME tidak ditemukan di .env"
assert token is not None, "DAGSHUB_TOKEN tidak ditemukan di .env"

# Set MLflow Tracking URI
mlflow.set_tracking_uri(f"https://dagshub.com/{repo}.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = token

# Set experiment
mlflow.set_experiment("Sleep Disorder Prediction with XGBoost Modelling - GridSearchCV")

print("MLflow tracking berhasil dikonfigurasi.")

# Autolog
mlflow.xgboost.autolog(log_models=True)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("sleep-health_life-style_preprocessing.csv")
X = df.drop(['Sleep Disorder'], axis=1)
y = df['Sleep Disorder']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# GridSearchCV Parameters
# -----------------------------
param_grid = {
    'objective': ['multi:softmax'],
    'num_class': [3],
    'eval_metric': ['mlogloss'],
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_lambda': [1],  # formerly 'lambda'
    'reg_alpha': [1],   # formerly 'alpha'
    'gamma': [0.1]
}


xgb_model = xgb.XGBClassifier(use_label_encoder=False)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    scoring='accuracy',
    verbose=2
)

# -----------------------------
# Training & Logging
# -----------------------------
with mlflow.start_run() as run:
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_precision", prec)
    mlflow.log_metric("test_recall", rec)
    mlflow.log_metric("test_f1_score", f1)

    # Simpan model
    model_path = "xgboost_best_model.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path)

    print("\n--- Evaluation Results ---")
    print(f"Best CV Accuracy: {best_score:.4f}")
    print(f"Test Accuracy   : {acc:.4f}")
    print(f"Precision       : {prec:.4f}")
    print(f"Recall          : {rec:.4f}")
    print(f"F1 Score        : {f1:.4f}")

    # -----------------------------
    # Logging ke DagsHub 
    # -----------------------------
    with dagshub_logger() as d_logger:
        d_logger.log_metrics({
            "best_cv_accuracy": best_score,
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec,
            "test_f1_score": f1
        })
