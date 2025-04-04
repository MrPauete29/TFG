
from development.utils import read_csv_file
from development.clean_datasets.Stroke.clean import clean
#file = "C:/Users/paumo/PycharmProjects/TFG/datasets/Cancer.csv"

#file = "C:/Users/paumo/PycharmProjects/TFG/datasets/EnfermedadCorazon.csv"
file = "C:/Users/paumo/PycharmProjects/TFG/datasets/Stroke.csv"

df = read_csv_file(file)


print(df["work_type"].unique())