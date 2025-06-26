from development.ml_models.models.LogisticRegression import LogisticRegression
from development.ml_models.models.RandomForest import RandomForest
from development.ml_models.models.XGBoost import XGBoost
from development.utils import read_csv_file, train_test_validation_split, factorize
from development.clean_datasets.Stroke.clean import clean as clean_stroke
from development.clean_datasets.HeartDisease.clean import clean as clean_heart
from development.clean_datasets.Cancer.clean import clean as clean_cancer
from imblearn.over_sampling import SMOTE
from typing import Dict
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import TomekLinks
file = "e"
model = LogisticRegression
if model == LogisticRegression:
    param_distributions = {
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200, 300, 500],
    'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]
}
elif model == RandomForest:
    param_distributions = {
        "n_estimators": [100, 300, 500, 700, 1000],
        "max_depth": [None, 10, 20, 30, 40, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True]
    }
elif model == XGBoost:
    param_distributions = {
        "n_estimators": [100, 300, 500, 700],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 10],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 1, 3, 5],
        "reg_alpha": [0, 0.1, 0.5, 1],
        "reg_lambda": [0.5, 1, 2]
    }
if file == "s":
    print("Using Strokes dataset...")
    target_column = "stroke"
    clean = clean_stroke
    file = "C:/Users/paumo/PycharmProjects/TFG/datasets/Stroke.csv"
    pos_label = 1
elif file == "e":
    print("Using Heart Disease dataset...")
    target_column = "HeartDisease"
    clean = clean_heart
    file = "C:/Users/paumo/PycharmProjects/TFG/datasets/EnfermedadCorazon.csv"
    pos_label = 1
elif file == "c":
    print("Using Cancer dataset...")
    target_column = "Diagnosis"
    clean = clean_cancer
    pos_label = 1
    file = "C:/Users/paumo/PycharmProjects/TFG/datasets/Cancer.csv"
else:
    print("File not found, using Cancer Dataset by default...")
    file = "C:/Users/paumo/PycharmProjects/TFG/datasets/Cancer.csv"
    target_column = "Diagnosis"
    clean = clean_cancer
    pos_label = 1
def main(file, target_column, pos_label, model, clean, param_distributions: Dict | None = None):
    model = model(pos_label = pos_label)
    df = read_csv_file(file)
    df = clean(df)
    X_train, y_train, X_test, y_test, X_validation, y_validation = train_test_validation_split(df,target_column)
    undersampler = RandomUnderSampler(sampling_strategy={0: 21970, 1: 21970}, random_state=50)
    oversampler = RandomOverSampler(sampling_strategy={0: 43940, 1: 43940}, random_state=50)
    smote = SMOTE(random_state=50)
    tl = TomekLinks(sampling_strategy='auto')
    X_train, X_validation = factorize(X_train),factorize(X_validation)
    X_train, y_train = tl.fit_resample(X_train, y_train)
    X_train, y_train = undersampler.fit_resample(X_train, y_train)
    #X_train, y_train = oversampler.fit_resample(X_train, y_train)
    #X_train, y_train= smote.fit_resample(X_train, y_train)
    model.train(X_train, y_train, param_distributions)
    return model, X_test, X_validation, y_test, y_validation

if __name__ == "__main__":
    trained_model, X_test, X_validation, y_test, y_validation = main(file = file, target_column = target_column, pos_label = pos_label, model = model, clean = clean, param_distributions = param_distributions)
    print(trained_model.feature_importance(X_test.columns))
    trained_model.evaluate(X_validation, y_validation)