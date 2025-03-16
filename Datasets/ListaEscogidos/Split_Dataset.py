import pandas as pd


def split_dataset(df,target_column):
    target_df = df[target_column]
    features_df = df.drop(columns=[target_column])
    return target_df,features_df


