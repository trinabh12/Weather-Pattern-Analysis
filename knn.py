from dataloader import DataFrame
from collections import Counter
from datetime import datetime
import numpy as np

class Knn:
    def __init__(self):
        self.df_obj = DataFrame()

    def convert_time_to_hours(self, time_str):
        hours, minutes, _ = map(int, time_str.split(':'))
        return hours + minutes / 60

    def convert_date_to_days(self, date_str):
        current_date = datetime.strptime(date_str, "%Y-%m-%d")
        start_of_year = datetime(current_date.year, 1, 1)
        return (current_date - start_of_year).days

    def calculateManhattanDistance(self, location, time, date):
        result = self.df_obj.get_rows_by_location(location)
        X = self.convert_time_to_hours(time)
        Y = self.convert_date_to_days(date)

        distances = []

        for row in result.collect():
            row_time = self.convert_time_to_hours(row['Date_Time'].strftime('%H:%M:%S'))
            row_date = self.convert_date_to_days(row['Date_Time'].strftime('%Y-%m-%d'))

            distance = abs(X - row_time) + abs(Y - row_date)
            distances.append((row, distance))

        return sorted(distances, key=lambda x: x[1])

    def kNearestNeighbors(self, kval, location, time, date):
        distances = self.calculateManhattanDistance(location, time, date)
        return distances[:kval]

    def calculateWeightedAverage(self, k_neighbors):
        distances = np.array([dist for _, dist in k_neighbors])
        weights = np.exp(-distances) / np.sum(np.exp(-distances))

        temperature_values = np.dot(weights, [neighbor['Temperature_C'] for neighbor, _ in k_neighbors])
        humidity_values = np.dot(weights, [neighbor['Humidity_pct'] for neighbor, _ in k_neighbors])
        precipitation_values = np.dot(weights, [neighbor['Precipitation_mm'] for neighbor, _ in k_neighbors])
        windspeed_values = np.dot(weights, [neighbor['Wind_Speed_kmh'] for neighbor, _ in k_neighbors])

        return {
            "Temperature": temperature_values,
            "Humidity": humidity_values,
            "Precipitation": precipitation_values,
            "Wind Speed": windspeed_values
        }

    def predict(self, kval, location, time, date):
        k_neighbors = self.kNearestNeighbors(kval, location, time, date)
        return self.calculateWeightedAverage(k_neighbors)
