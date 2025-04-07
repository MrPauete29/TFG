from development.utils import factorize

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
    df["ever_married"] = df["ever_married"].map(married_map)
    df["smoking_status"] = df["smoking_status"].map(smoke_map)
    df = factorize(df, ["gender", "Residence_type","work_type"])
    return df