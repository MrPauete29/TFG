import xgboost as xgb
import pandas as pd
from sklearn.metrics import classification_report
import joblib
class XGBoost:
    def __init__(self, **params):
        self.model = xgb.XGBClassifier(**params)
        self.trained = False

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        self.trained = True
        return self.model.fit(X_train, y_train)

    def predict(self, X_val):
        if not self.trained:
            raise Exception("Model not trained")
        return self.model.predict(X_val)

    def predict_proba(self, X_val):
        if not self.trained:
            raise Exception("Model not trained")
        return self.model.predict_proba(X_val)[:, 1]

    def evaluate(self, X: pd.Dataframe = None, y: pd.DataFrame = None, y_pred: pd.DataFrame = None):
        if not self.trained:
            raise Exception("Model not trained")
        if y is None:
            raise ValueError("y set must be provided")
        if y_pred is None:
            if X is None:
                raise ValueError("y_pred set or X set must be provided")
            y_pred = self.predict(X)
        return classification_report(y, y_pred)

    def save_model(self, filename: str):
        joblib.dump(self.model, f"trained_models/random_forest/{filename}")

    def load_model(self, filename: str):
        self.model = joblib.load(f"trained_models/random_forest/{filename}")