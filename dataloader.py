from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp

FILE_PATH = "weather_data.csv"

class DataFrame(SparkSession):
    def __init__(self):
        self.spark = SparkSession.builder.appName("WeatherAnalysis").getOrCreate()
        self.df = self.spark.read.csv(FILE_PATH, header=True, inferSchema=True)
        self.df = self.df.withColumn("Date_Time", to_timestamp(col("Date_Time"), "yyyy-MM-dd HH:mm:ss"))

    def get_rows_by_location(self, location):
         filtered_df = self.df.filter(col("Location") == location)
         if filtered_df.count() == 0:
             return "No rows found"
         return filtered_df

        

    
