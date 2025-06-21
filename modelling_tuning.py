import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import pandas as pd
import joblib
import os

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
mlflow.set_experiment("sleep_disorder_prediction_gridsearch")

# Define the hyperparameter grid for GridSearchCV with regularization
param_grid = {
    'objective': ['multi:softmax'],
    'num_class': [3],
    'eval_metric': ['mlogloss'],
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'lambda': [0.1, 1, 10],  # L2 regularization term
    'alpha': [0.1, 1, 10],    # L1 regularization term
    'gamma': [0, 0.1, 0.5]    # Minimum loss reduction
}

# Initialize the XGBoost model
xgb_model = xgb.XGBClassifier()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit GridSearchCV to the data
with mlflow.start_run() as run:
    # Train the model using GridSearchCV
    grid_search.fit(X_train, y_train)
    
    # Log the best hyperparameters and score
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_accuracy", grid_search.best_score_)
    
    # Log the best model found by GridSearchCV
    best_model = grid_search.best_estimator_
    
    # Save the model
    model_filename = "xgboost_best_model.joblib"
    joblib.dump(best_model, model_filename)
    
    # Check if the file is being saved correctly
    if os.path.exists(model_filename):
        print(f"Model saved to {model_filename}")
    else:
        print("Error: Model was not saved.")
    
    # Log the artifact (save the model)
    mlflow.log_artifact(model_filename)
    print(f"Artifact logged: {model_filename}")

    # Log the model to MLflow
    mlflow.xgboost.log_model(best_model, "model")
    
    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
    print(f"Test Set Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Log metrics (manual)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)