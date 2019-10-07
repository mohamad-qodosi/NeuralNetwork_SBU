import numpy as np
import matplotlib.pyplot as plt


def get_data_and():
    X = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]

    y = [[0],
         [0],
         [0],
         [1]]

    X = np.array(X)
    y = np.array(y)

    return X, y


def get_data_or():
    X = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]

    y = [[0],
         [1],
         [1],
         [1]]

    X = np.array(X)
    y = np.array(y)

    return X, y


def get_normal_distribution():
    X1 = np.random.multivariate_normal((-3, -3), cov=[[5, 0], [0, 5]], size=(100))
    y1 = np.zeros(100)
    X2 = np.random.multivariate_normal((4, 4), cov=[[3, 0], [0, 3]], size=(100))
    y2 = np.ones(100)

    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))

    return X, y

get_normal_distribution()