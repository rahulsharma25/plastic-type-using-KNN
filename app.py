from flask import Flask, url_for, request, render_template
import numpy as np
import csv
import math

app = Flask(__name__)

def compute_ln_norm_distance(vector1, vector2, n):
    vector_len = len(vector1)
    distance = 0
    for i in range(vector_len):
        distance += (abs(vector1[i] - vector2[i])) ** n
    
    return distance ** (1/n)

def find_k_nearest_neighbors(train_X, test_example, k, n):
    dist_indices_pairs = []
    for i in range(len(train_X)):
        distance = compute_ln_norm_distance(train_X[i], test_example, n)
        dist_indices_pairs.append([i,distance])
    
    dist_indices_pairs.sort(key = lambda x :(x[1],x[0]))
    k_nearest_list = [i[0] for i in dist_indices_pairs]
    k_nearest_list = k_nearest_list[:k]
    return k_nearest_list

def classify_points_using_knn(train_X, train_Y, test_X, k, n):
    predicted_classes = []
    for test_example in test_X:
        k_nearest_indices = find_k_nearest_neighbors(train_X, test_example, k, n)
        k_nearest_classes = []
        
        for index in k_nearest_indices:
            k_nearest_classes.append(train_Y[index])
        
        classes = list(set(k_nearest_classes))
        max_count = 0
        mode_class = -1
        for certain_class in classes:
            count = k_nearest_classes.count(certain_class)
            if count > max_count:
                max_count = count
                mode_class = certain_class
        
        predicted_classes.append(mode_class)

    return predicted_classes

def calculate_accuracy(pred_Y, actual_Y):
    total_cases = len(actual_Y)
    true_prediction = 0

    for i in range(total_cases):
        if pred_Y[i] == actual_Y[i]:
            true_prediction += 1
    
    accuracy = true_prediction/total_cases
    return accuracy

def get_best_k(train_X, train_Y, validation_split_percent, n):
    training_data_fraction = math.floor(((100 - validation_split_percent)/100) * len(train_X))
    
    training_data = train_X[0:training_data_fraction]
    validation_data = train_X[training_data_fraction : ]
    actual_Y = train_Y[training_data_fraction : ]

    accuracy_k_pairs = []
    for k in range(1, len(training_data) + 1):
        predicted_Y = classify_points_using_knn(training_data, train_Y, validation_data, k, n)
        accuracy = calculate_accuracy(predicted_Y, actual_Y)
        accuracy_k_pairs.append([accuracy, k])
    
    accuracy_k_pairs.sort(key = lambda x : [-x[0], x[1]])

    return accuracy_k_pairs[0][1]


train_X = np.genfromtxt("train_X_knn.csv", delimiter=",", dtype=np.float64, skip_header=1)
train_Y = np.genfromtxt("train_Y_knn.csv", delimiter=",", dtype=np.int32)
validation_split_percent = 25
n = 2
k = get_best_k(train_X, train_Y, validation_split_percent, n)

def predict(X):
    prediction = classify_points_using_knn(train_X, train_Y, X, k, n)
    
    return prediction[0]

@app.route("/")
def home_page():
    return render_template("index.html")

@app.route("/submit", methods=["POST", "GET"])
def submit():
    if request.method == "POST":
        X = np.array([float(x) for x in request.form.values()])
        X = np.reshape(X, (1, len(X)))
        
        prediction = predict(X)

        return render_template("index.html", prediction = f'The plastic is classified as type: {prediction}')

if __name__ == "__main__":
    app.run(debug=True)