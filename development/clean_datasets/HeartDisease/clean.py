from development.utils import eliminate_row_na, eliminate_column_na, factorize

def clean(df):
    map_edades = {
        "18-24": 0,
        "25-29": 0,
        "30-34": 0,
        "35-39": 1,
        "40-44": 1,
        "45-49": 1,
        "50-54": 2,
        "55-59": 2,
        "60-64": 2,
        "65-69": 3,
        "70-74": 3,
        "75-79": 3,
        "80 or older": 3
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


