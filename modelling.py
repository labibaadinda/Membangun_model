import mlflow
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib  # For saving the model manually if needed

# Load dataset
file_path = 'sleep-health_life-style_preprocessing.csv'

# Load the CSV file
df = pd.read_csv(file_path)

# Split data into features (X) and target (y)
X = df.drop(['Sleep Disorder'], axis=1)
y = df['Sleep Disorder']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set the experiment name for MLflow
mlflow.set_experiment("sleep_disorder_prediction")

# Start MLflow run
with mlflow.start_run():

    # Define the XGBoost model with parameters
    n_estimators = 200
    max_depth = 10
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, 
                              eval_metric='mlogloss', n_estimators=n_estimators, max_depth=max_depth)

    # Fit the model
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log hyperparameters (manual)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Log metrics (manual)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # ✅ Log ke Model Registry
    mlflow.xgboost.log_model(
        model,
        artifact_path="model",
        registered_model_name="ModelSML"  # <- Nama model di registry
    )

    # Opsional: juga simpan model secara manual
    joblib.dump(model, "xgboost_model.joblib")
    mlflow.log_artifact("xgboost_model.joblib")
    # Print metrics to console
    print(f"Model Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
