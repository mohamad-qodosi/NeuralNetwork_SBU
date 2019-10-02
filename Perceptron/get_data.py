import numpy as np

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