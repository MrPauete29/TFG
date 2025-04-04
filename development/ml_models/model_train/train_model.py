from development.ml_models.LogisticRegression import LogisticRegression
from development.utils import read_csv_file, train_test_validation_split, factorize
from imblearn.over_sampling import SMOTE
from typing import Dict
def main(file, target_column, pos_label, model, clean, param_distributions: Dict | None = None):

    model = model(pos_label = pos_label)
    df = read_csv_file(file)
    df = clean(df)
    X_train, y_train, X_test, y_test, X_validation, y_validation = train_test_validation_split(df,target_column)
    X_train, X_validation = factorize(X_train),factorize(X_validation)
    smote = SMOTE(random_state=50)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    model.train(X_train_resampled, y_train_resampled, param_distributions)
    return model, X_test, X_validation, y_test, y_validation