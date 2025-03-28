from knn import Knn

knn_model = Knn()

location = input("Enter location: ")
time = input("Enter time (HH:MM:SS): ")
date = input("Enter date (YYYY-MM-DD): ")



prediction = knn_model.predict(15, location, time, date)

print("\nPredicted Weather Conditions:")
print(prediction)
