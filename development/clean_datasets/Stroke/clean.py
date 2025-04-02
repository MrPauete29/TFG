from development.utils import eliminate_row_na, eliminate_column_na, factorize

def clean(df):
    smoke_map = {
    "never smoked": 0,
    "formerly smoked": 1,
    "smokes": 2,
    "Unknown": -1
    }
    married_map = {
        "Yes": 1,
        "No": 0
    }
    df = df.drop("id", axis = 1)
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    print(df.columns)
    df["ever_married"] = df["ever_married"].map(married_map)
    df["smoking_status"] = df["smoking_status"].map(smoke_map)
    df = factorize(df, ["gender", "Residence_type"])
    return df