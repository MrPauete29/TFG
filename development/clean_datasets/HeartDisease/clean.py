from development.utils import eliminate_row_na, eliminate_column_na, factorize

def clean(df):
    map_edades = {
        "18-24": 21,
        "25-29": 27,
        "30-34": 32,
        "35-39": 37,
        "40-44": 42,
        "45-49": 47,
        "50-54": 52,
        "55-59": 57,
        "60-64": 62,
        "65-69": 67,
        "70-74": 72,
        "75-79": 77,
        "80 or older": 82
    }
    map_diabetes = {
        "No": 0,
        "No, borderline diabetes": 1,
        "Yes (during pregnancy)": 2,
        "Yes": 3
    }
    map_genhealth = {
        "Excellent": 5,
        "Very good": 4,
        "Good": 3,
        "Fair": 2,
        "Poor": 1
    }
    map_yes_no = {
        "Yes": 1,
        "No": 0
    }
    columns = ["Sex", "Race"]
    df = eliminate_column_na(df, 30)
    df = eliminate_row_na(df, 20)
    df = df.drop(["PhysicalHealth", "MentalHealth"], axis=1)
    df["AgeCategory"] = df["AgeCategory"].map(map_edades)
    df["Diabetic"] = df["Diabetic"].map(map_diabetes)
    df["GenHealth"] = df["GenHealth"].map(map_genhealth)
    df["Smoking"] = df["Smoking"].map(map_yes_no)
    df["AlcoholDrinking"] = df["AlcoholDrinking"].map(map_yes_no)
    df["HeartDisease"] = df["HeartDisease"].map(map_yes_no)
    df["Stroke"] = df["Stroke"].map(map_yes_no)
    df["SkinCancer"] = df["SkinCancer"].map(map_yes_no)
    df["KidneyDisease"] = df["KidneyDisease"].map(map_yes_no)
    df["Asthma"] = df["Asthma"].map(map_yes_no)
    df["DiffWalking"] = df["DiffWalking"].map(map_yes_no)
    df = factorize(df, columns)
    return df


