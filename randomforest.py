from dataloader import DataFrame
from collections import Counter

class randomForest:
    def __init__(self):
        self.df_obj = DataFrame() 
        self.predicted_temperature = 0
        self.predicted_humidity = 0
        self.predicted_precipitation = 0
        self.predicted_windspeed = 0