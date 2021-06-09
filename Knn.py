from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import sklearn.model_selection


def read_data_from_file():
    """
    read data from sample file
    :return:
    """
    return pd.read_csv('rectangle.txt', delim_whitespace=True, names=['x_val', 'y_val', 'label'])


def split_data(data):
    """
    Split dataset into training set and test set
    :param data: pandas data frame contains the data
    :return: train set, test set data frames
    """
    return sklearn.model_selection.train_test_split(data, data['label'], test_size=0.5)


# calculate the Euclidean distance between two vectors wwith given p
def euclidean_distance(row1, row2, p):

    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**p
    return distance**(1/p)


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors, p):

    distances = list()
    for idx, train_row in train.iterrows():
        dist = euclidean_distance(test_row, train_row, p)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors, p):
    neighbors = get_neighbors(train, test_row, num_neighbors, p)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors, p):
    predictions = list()
    for idx, row in test.iterrows():
        output = predict_classification(train, row, num_neighbors, p)
        predictions.append(output)
    return(predictions)

# Calculate accuracy percentage


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

 # Convert string column to float




def runner():
    data = read_data_from_file()

    X_train, X_test, y_train, y_test = split_data(data)
    
    for k in range(1, 10, 2):

        for p in [1, 2, float('inf')]:

            predicted = k_nearest_neighbors(X_train, X_test, k, p)
            accuracy = accuracy_metric(y_test, predicted)
            print("Accuracy:", 1 - accuracy)


if __name__ == "__main__":
    runner()
