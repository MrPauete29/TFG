from pyspark.sql import SparkSession
from development.utils import read_csv_file
from development.clean_datasets.Stroke.clean import clean
#file = "C:/Users/paumo/PycharmProjects/TFG/datasets/Cancer.csv"

file = "C:/Users/paumo/PycharmProjects/TFG/datasets/EnfermedadCorazon.csv"
file_b = "C:/Users/paumo/PycharmProjects/TFG/datasets/EnfermedadCorazon_Balanceado.csv"
#file = "C:/Users/paumo/PycharmProjects/TFG/datasets/Stroke.csv"

spark = SparkSession.builder.appName("RestarFilas").getOrCreate()

spark_df_no_balanceado = spark.read.csv(file)
spark_df_balanceado = spark.read.csv(file_b)


spark_df_resultado = spark_df_no_balanceado.subtract(spark_df_balanceado)


filas_no_balanceado_spark = spark_df_no_balanceado.count()
filas_balanceado_spark = spark_df_balanceado.count()
filas_resultado_spark = spark_df_resultado.count()



