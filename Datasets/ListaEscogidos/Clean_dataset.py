import pandas as pd
import numpy as np


def factorize(df):
    for column in df.columns:
        if df[column].dtype not in ["int64","float64"]:
            df[column] = df[column].factorize()[0]
    return df

def eliminate_column_na(df, threshold):
    threshold = threshold / 100
    threshold = threshold * len(df)
    df = df.dropna(axis=1, thresh=threshold)
    return df
def eliminate_row_na(df, threshold):
    threshold = 100 - threshold
    threshold = threshold / 100
    threshold = threshold * len(df.columns)
    df = df.dropna(axis=0, thresh=threshold)
    return df






