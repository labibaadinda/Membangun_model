import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('sleep-health_life-style_preprocessing.csv')
X = df.drop(['Sleep Disorder'], axis=1)
y = df['Sleep Disorder']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set experiment
mlflow.set_experiment("sleep_disorder_prediction_basic")

# ENABLE AUTOLOG
mlflow.xgboost.autolog()

# Start run
with mlflow.start_run():
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        eval_metric='mlogloss',
        n_estimators=100,
        max_depth=6
    )
    model.fit(X_train, y_train)
