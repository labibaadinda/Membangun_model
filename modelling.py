import mlflow
import mlflow.xgboost  
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib  

# Load dataset
file_path = 'sleep-health_life-style_preprocessing.csv'
df = pd.read_csv(file_path)

# Split data into features (X) and target (y)
X = df.drop(['Sleep Disorder'], axis=1)
y = df['Sleep Disorder']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLflow experiment
mlflow.set_experiment("sleep_disorder_prediction")

# ENABLE AUTOLOGGING
mlflow.xgboost.autolog()

# Start MLflow run
with mlflow.start_run():

    # Define and train model
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        n_estimators=200,
        max_depth=10
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log metrics (manual)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Simpan ke Model Registry
    mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        registered_model_name="ModelSML"
    )

    # (Opsional) Simpan manual
    joblib.dump(model, "xgboost_model.joblib")
    mlflow.log_artifact("xgboost_model.joblib")

    # Print metrics
    print(f"Model Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

