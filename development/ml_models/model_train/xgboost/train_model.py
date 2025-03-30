from development.ml_models.RandomForest import RandomForest
from development.utils import read_csv_file, train_test_validation_split, factorize
from imblearn.over_sampling import SMOTE,RandomOverSampler

if __name__ == "__main__":
    file = f"C:/Users/paumo/PycharmProjects/TFG/datasets/Cancer.csv"
    model = RandomForest(pos_label="M", beta = 2)
    df = read_csv_file(file)
    X_train, y_train, X_test, y_test, X_validation, y_validation = train_test_validation_split(df,"Diagnosis")
    X_train, X_validation = factorize(X_train),factorize(X_validation)
    print(X_train.shape)
    ros = RandomOverSampler(random_state=50)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    print(X_train_resampled.shape)
    model.train(X_train_resampled, y_train_resampled)
    #model.train(X_train, y_train)
    print(model.evaluate(X_validation,y_validation))