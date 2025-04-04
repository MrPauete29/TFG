from development.utils import eliminate_row_na, eliminate_column_na, factorize

def clean(df):
    map_diagnosis ={
        "M": 1,
        "B": 0
    }
    df = eliminate_column_na(df, 30)
    df = eliminate_row_na(df, 20)
    df["Diagnosis"] = df["Diagnosis"].map(map_diagnosis)
    return df