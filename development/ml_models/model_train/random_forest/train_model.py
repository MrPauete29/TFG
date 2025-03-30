from development.ml_models.RandomForest import RandomForest
from development.utils import read_csv_file, train_test_validation_split, factorize
from imblearn.over_sampling import SMOTE,RandomOverSampler

if __name__ == "__main__":
    file = f"C:/Users/paumo/PycharmProjects/TFG/datasets/Cancer.csv"
    param_distributions = {
        'n_estimators': [100, 200, 300, 500, 800],
        'max_depth': [10, 20, 30, 40, 50, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True]
    }
    model = RandomForest(pos_label="M", beta = 2)
    df = read_csv_file(file)
    X_train, y_train, X_test, y_test, X_validation, y_validation = train_test_validation_split(df,"Diagnosis")
    X_train, X_validation = factorize(X_train),factorize(X_validation)
    print(X_train.shape)
    smote = SMOTE(random_state=50)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(X_train_resampled.shape)
    model.train(X_train_resampled, y_train_resampled, param_distributions)
    #model.train(X_train, y_train)
    print(model.evaluate(X_validation,y_validation))

