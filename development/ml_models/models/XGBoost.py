import xgboost as xgb
import pandas as pd
from sklearn.metrics import classification_report, make_scorer, recall_score
from development.utils import model_best_parameters, create_visualize_confusion_matrix,create_visualize_classification_report
from typing import Dict
import joblib

class XGBoost:
    def __init__(self, pos_label):
        self.pos_label = pos_label
        self.model = xgb.XGBClassifier(random_state = 50)
        self.trained = False

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame, param_distributions: Dict):
        self.trained = True
        scoring = make_scorer(recall_score,pos_label=self.pos_label)
        self.model = model_best_parameters(self.model,scoring = scoring, param_distributions = param_distributions)
        self.model.fit(X_train, y_train)

    def predict(self, X_val):
        if not self.trained:
            raise Exception("Model not trained")
        return self.model.predict(X_val)

    def predict_proba(self, X_val):
        if not self.trained:
            raise Exception("Model not trained")
        return self.model.predict_proba(X_val)[:,1]

    def evaluate(self, X: pd.DataFrame = None, y_true: pd.DataFrame = None, y_pred: pd.DataFrame = None):
        if not self.trained:
            raise Exception("Model not trained")
        if y_true is None:
            raise ValueError("y set must be provided")
        if y_pred is None:
            if X is None:
                raise ValueError("y_pred set or X set must be provided")
            y_pred = self.predict(X)
        create_visualize_confusion_matrix(y_true, y_pred)
        create_visualize_classification_report(y_true, y_pred)

    def save_model(self, filename: str) -> None:
        joblib.dump(self.model,f"trained_models/random_forest/{filename}.pkl")

    def load_model(self, filename: str) -> None:
        self.model = joblib.load(f"trained_models/random_forest/{filename}.pkl")
        self.trained = True