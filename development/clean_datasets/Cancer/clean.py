import numpy as np


def clean(df):
    map_diagnosis ={
        "M": 1,
        "B": 0
    }

    columns_na= [
        "concavity1", "concave_points1",
        "concavity2", "concave_points2",
        "concavity3", "concave_points3"
    ]

    df[columns_na] = df[columns_na].replace(0.0, np.nan)
    df = df.dropna()
    df["Diagnosis"] = df["Diagnosis"].map(map_diagnosis)

    return df