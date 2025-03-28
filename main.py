from flask import Flask, render_template, request
from knn import Knn

app = Flask(__name__)
knn_model = Knn()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        location = request.form['location']
        date = request.form['date']
        time = request.form['time']
        
        prediction = knn_model.predict(15, location, time, date)
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)


#knn_model = Knn()

#location = input("Enter location: ")
#time = input("Enter time (HH:MM:SS): ")
#date = input("Enter date (YYYY-MM-DD): ")

#k_neighbors = knn_model.kNearestNeighbors(10, location, time, date)
#prediction = knn_model.calculateMode(k_neighbors)


#print("\nPredicted Weather Conditions:")
#print(prediction)
