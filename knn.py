from dataloader import DataFrame
from collections import Counter

class Knn:
    def __init__(self):
        self.df_obj = DataFrame() 
        self.predicted_temperature = 0
        self.predicted_humidity = 0
        self.predicted_precipitation = 0
        self.predicted_windspeed = 0

    def calculateManhattanDistance(self, location, time, date):
        result = self.df_obj.get_rows_by_location(location)  
        hours, minutes, seconds = map(int, time.split(':'))
        X = hours
        year, month, day = map(int, date.split('-'))
        Y = month * 31 + day

        distances = []

        for row in result.collect():
            row_time = int(row['Date_Time'].hour)
            row_date = row['Date_Time'].month * 31 + row['Date_Time'].day

            distance = abs(X - row_time) + abs(Y - row_date)
            distances.append((row, distance))

        return sorted(distances, key=lambda x: x[1])

    def kNearestNeighbors(self, kval, location, time, date):
        distances = self.calculateManhattanDistance(location, time, date)
        return distances[:kval]

    def calculateMode(self, k_neighbors):
        temperature_values = [neighbor[0]['Temperature_C'] for neighbor in k_neighbors]
        humidity_values = [neighbor[0]['Humidity_pct'] for neighbor in k_neighbors]
        precipitation_values = [neighbor[0]['Precipitation_mm'] for neighbor in k_neighbors]
        windspeed_values = [neighbor[0]['Wind_Speed_kmh'] for neighbor in k_neighbors]
        
        self.predicted_temperature = Counter(temperature_values).most_common(1)[0][0]
        self.predicted_humidity = Counter(humidity_values).most_common(1)[0][0]
        self.predicted_precipitation = Counter(precipitation_values).most_common(1)[0][0]
        self.predicted_windspeed = Counter(windspeed_values).most_common(1)[0][0]
    
        return {
            "Temperature": self.predicted_temperature,
            "Humidity": self.predicted_humidity,
            "Precipitation": self.predicted_precipitation,
            "Wind Speed": self.predicted_windspeed
        }
