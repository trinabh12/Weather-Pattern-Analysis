from knn import Knn

# Create Knn object
knn_model = Knn()

# Get user inputs
location = input("Enter location: ")
time = input("Enter time (HH:MM:SS): ")
date = input("Enter date (YYYY-MM-DD): ")

# Run KNN
k_neighbors = knn_model.kNearestNeighbors(10, location, time, date)
prediction = knn_model.calculateMode(k_neighbors)

# Print the result
print("\nPredicted Weather Conditions:")
print(prediction)
