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


def runner():
    data = read_data_from_file()
    for i in range(101):

        X_train, X_test, y_train, y_test = split_data(data)

        for k in range(1, 10, 2):
            # Create KNN Classifier
            knn = KNeighborsClassifier(n_neighbors=k)
            # Train the model using the training sets
            knn.fit(X_train, y_train)

            # Predict the response for test dataset
            y_pred = knn.predict(X_test)

            print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# calculate the Euclidean distance between two vectors wwith given p
def euclidean_distance(row1, row2, p):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**p
    return distance**(1/p)



# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


if __name__ == "__main__":
    runner()
