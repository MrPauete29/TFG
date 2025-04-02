from development.ml_models.model_train.train_model import main
from development.ml_models.LogisticRegression import LogisticRegression

param_distributions = {
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga'],
        'max_iter': [100, 200, 300, 500],
        'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]
    }
file = "c"
model = LogisticRegression
if file == "s":
    print("Using Strokes dataset...")
    target_column = "stroke"
    file = "C:/Users/paumo/PycharmProjects/TFG/datasets/Stroke.csv"
    pos_label = "1"
elif file == "e":
    print("Using Heart Disease dataset...")
    target_column = "HeartDisease"
    file = "C:/Users/paumo/PycharmProjects/TFG/datasets/EnfermedadCorazon.csv"
    pos_label = "Yes"
elif file == "c":
    print("Using Cancer dataset...")
    target_column = "Diagnosis"
    pos_label = "M"
    file = "C:/Users/paumo/PycharmProjects/TFG/datasets/Cancer.csv"
else:
    print("File not found, using Cancer Dataset by default...")
    file = "C:/Users/paumo/PycharmProjects/TFG/datasets/Cancer.csv"
if __name__ == "__main__":
    trained_model, X_test, X_validation, y_test, y_validation = main(file = file, target_column = target_column, pos_label = pos_label, model = model, param_distributions = param_distributions)
    print(trained_model.evaluate(X_validation, y_validation))