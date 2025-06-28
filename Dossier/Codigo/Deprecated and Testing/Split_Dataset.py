import pandas as pd

try:
    df = pd.read_csv(document)
    print(df)
except FileNotFoundError:
    raise FileNotFoundError("Archivo no encontrado")
except:
    raise ValueError("Error en la lectura del archivo")

def split_dataset(target_column):
    target_df = df[target_column]
    features_df = df.drop(columns=[target_column])
    return target_df,features_df

if __name__ = "__main__":
    document = input("Document (Without extension: ")
    document = f"{document}.csv"
    target_column = input("Name of the column you want to predict")
    target_df,features_df = split_dataset(target_column)
    target_df.to_csv(f"Target/{document}")
    features_df.to_csv(f"Features/{document}")
