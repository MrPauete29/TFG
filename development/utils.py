import pandas as pd


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
def split_dataset(df: pd.DataFrame,target_column) -> (pd.DataFrame,pd.DataFrame):
    target_df = df[target_column]
    features_df = df.drop(columns=[target_column])
    return target_df,features_df
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