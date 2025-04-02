from development.utils import eliminate_row_na, eliminate_column_na, factorize

def clean(df):
    df = eliminate_column_na(df, 30)
    df = eliminate_row_na(df, 20)