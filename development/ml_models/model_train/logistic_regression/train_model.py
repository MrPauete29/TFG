from development.ml_models.LogisticRegression import LogisticRegression
from development.utils import read_csv_file, train_test_validation_split, factorize
from imblearn.over_sampling import SMOTE,RandomOverSampler

if __name__ == "__main__":
    pos_label = "Yes"

    target_column = "HeartDisease"

    file = f"C:/Users/paumo/PycharmProjects/TFG/datasets/EnfermedadCorazon.csv"

    param_distributions = {
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga'],
        'max_iter': [100, 200, 300, 500],
        'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]
    }

    model = LogisticRegression(pos_label = pos_label)
    df = read_csv_file(file)
    X_train, y_train, X_test, y_test, X_validation, y_validation = train_test_validation_split(df,target_column)
    X_train, X_validation = factorize(X_train),factorize(X_validation)
    print(X_train.shape)
    smote = SMOTE(random_state=50)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(X_train_resampled.shape)
    model.train(X_train_resampled, y_train_resampled, param_distributions)
    #model.train(X_train, y_train)
    print(model.evaluate(X_validation,y_validation))