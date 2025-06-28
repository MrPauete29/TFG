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

if __name__ == "__main__"
    document = "Stroke.csv"
    if document == None:
        document = input("Document (Without extension: ")
        document = f"{document}.csv"

    try:
        df = pd.read_csv(document)
        print(df.shape)
    except FileNotFoundError:
        raise FileNotFoundError("Archivo no encontrado")
    except:
        raise ValueError("Error en la lectura del archivo")
    if document == "Stroke.csv":
        df['smoking_status'] = df['smoking_status'].replace('Unknown', np.nan)
    df = eliminate_column_na(df, 70)
    df = eliminate_row_na(df, 0)
    print(df.shape)




