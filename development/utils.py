import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
def factorize(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if df[column].dtype not in ["int64","float64"]:
            df[column] = df[column].factorize()[0]
    return df

def eliminate_column_na(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    threshold = threshold / 100
    threshold = threshold * len(df)
    df = df.dropna(axis=1, thresh=int(threshold))
    return df

def eliminate_row_na(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    threshold = 100 - threshold
    threshold = threshold / 100
    threshold = threshold * len(df.columns)
    df = df.dropna(axis=0, thresh=int(threshold))
    return df

def split_dataset(df: pd.DataFrame,target_column: str) -> (pd.DataFrame,pd.DataFrame):
    target_df = df[target_column]
    features_df = df.drop(columns=[target_column])
    return features_df,target_df

def join_dataframe_columns(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    return df1.join(df2)

def join_dataframe_rows(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df1, df2], ignore_index=True)

def read_csv_file(file) -> pd.DataFrame:
    try:
        return pd.read_csv(file)
    except FileNotFoundError:
        raise FileNotFoundError("Archivo no encontrado")
    except:
        raise ValueError("Error en la lectura del archivo")

def write_csv_file(df: pd.DataFrame, nombre_archivo: str = None, ruta: str = None):
    if nombre_archivo is None:
        raise TypeError("Nombre de archivo no especificado")
    if ruta is None:
        df.to_csv(nombre_archivo, index = False)
        return
    df.to_csv(f"{ruta}/{nombre_archivo}")

def train_test_validation_split(df: pd.DataFrame, target_column: str, test_size: float = 0.1, validation_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X, y = split_dataset(df, target_column)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + validation_size, random_state=19)
    validation_size_proportioned = validation_size / (test_size + validation_size)
    X_test, X_validation, y_test, y_validation = train_test_split(X_temp, y_temp, test_size=validation_size_proportioned, random_state=19)

    return X_train, y_train, X_test, y_test, X_validation, y_validation
