import pandas as pd
import pytest
import sklearn.model_selection


def read_data_from_file():
    """
    read data from sample file
    :return:
    """
    return pd.read_csv('rectangle.txt', delim_whitespace=True, names=['x_val', 'y_val', 'label'])


def split_data(data, target):
    """
    Split dataset into training set and test set
    :param data: pandas data frame contains the data
    :return: train set, test set data frames
    """
    return sklearn.model_selection.train_test_split(data, target, test_size=0.5)


def euclidean_distance(row1, row2, p):
    """
    calculate the Euclidean distance between two vectors with given p
    :param row1:
    :param row2:
    :param p:
    :return:
    """

    distance = 0.0
    for i in range(len(row1)):
        distance += (abs(row1[i] - row2[i]))**p
    return distance**(1/p)


def get_neighbors(train, test_row, num_neighbors, p):
    """
    Locate the most similar neighbors
    :param train:
    :param test_row:
    :param num_neighbors:
    :param p:
    :return:
    """

    distances = list()

    for train_row in train.itertuples():
        dist = euclidean_distance(test_row, train_row, p)
        distances.append((train_row, dist))
    distances = sorted(distances, key=lambda t: t [ 1 ])

    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances [ i ] [ 0 ])
    return neighbors


"""
Make a classification prediction with neighbors
return: -1 or 1 (label)
"""


def predict_classification(train, test_row, num_neighbors, p):
    """
    Make a classification prediction with neighbors
    :param train:
    :param test_row:
    :param num_neighbors:
    :param p:
    :return:
    """

    neighbors = get_neighbors(train, test_row, num_neighbors, p)
    output_values = [row.label for row in neighbors]
    # return the most represented class among the neighbors.
    prediction = max(set(output_values), key=output_values.count)
    return prediction


"""
KNN Algorithm
return: list of the predictions of all test rows.
"""


def k_nearest_neighbors(train_set, test_set, num_neighbors, p):
    """
    kNN Algorithm
    :param train:
    :param test:
    :param num_neighbors:
    :param p:
    :return:
    """

    predictions = {}

    for row in test_set.itertuples():
        output = predict_classification(train_set, row, num_neighbors, p)
        predictions[row.Index] = output
    return predictions


def accuracy_metric(actual, predicted):
    """
    Calculate accuracy percentage
    :param actual:
    :param predicted_dict:
    :return:
    """
    correct = 0

    for i, v in actual.items():
        if v == predicted[i]:
            correct += 1
    return correct / len(actual)


def print_results(true_error_dict, empirical_error_dict):
    for x in true_error_dict:
        print("(", x, ")", "True error avg:", true_error_dict.get(x)/100,
              "Empirical error avg:", empirical_error_dict.get(x)/100)


def runner():
    data = read_data_from_file()
    true_error_average = {}
    empirical_error_average = {}

    for i in range(100):
        X_train, X_test, y_train, y_test = split_data(data, data [ 'label' ])

        for k in range(1, 10, 2):

            for p in [ 1, 2, float('inf') ]:
                predicted_for_train = k_nearest_neighbors(
                    X_train, X_train, k, p)

                predicted_for_test = k_nearest_neighbors(X_train, X_test, k, p)

                empirical_error = 1 - \
                    accuracy_metric(y_train, predicted_for_train)
                true_error = 1 - accuracy_metric(y_test, predicted_for_test)
                if (i == 0):
                    true_error_average [ "{},{}".format(p, k) ] = true_error
                    empirical_error_average [ "{},{}".format(
                        p, k) ] = empirical_error

                else:

                    res_t = true_error_average.get(
                        "{},{}".format(p, k)) + true_error
                    res_e = empirical_error_average.get(
                        "{},{}".format(p, k)) + empirical_error
                    true_error_average.update({"{},{}".format(p, k): res_t})
                    empirical_error_average.update(
                        {"{},{}".format(p, k): res_e})

    print_results(true_error_average, empirical_error_average)


if __name__ == "__main__":
    runner()


def test_distance():
    assert euclidean_distance([ 2, 4 ], [ -3, 8 ], 2) == pytest.approx(6.403, 0.001)
