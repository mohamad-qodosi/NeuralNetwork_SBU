import numpy as np
import matplotlib.pyplot as plt


def get_data_and():
    X = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]

    y = [0,
         0,
         0,
         1]

    X = np.array(X)
    y = np.array(y)

    return X, y


def get_data_or():
    X = [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]]

    y = [0,
         1,
         1,
         1]

    X = np.array(X)
    y = np.array(y)

    return X, y

def get_normal_distribution2D(class_mean, classs_cov, samples_in_classes=100):
    assert len(class_mean) == len(classs_cov)
    X = []
    y = []
    for i in range(len(class_mean)):
        Xi = np.random.multivariate_normal(class_mean[i], cov=classs_cov[i], size=(samples_in_classes))
        yi = np.ones(samples_in_classes) * i

        X.append(Xi)
        y.append(yi)
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y