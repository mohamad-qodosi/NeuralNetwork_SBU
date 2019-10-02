from get_data import get_data_and as data
import numpy as np
import matplotlib.pyplot as plt

def hardlim(x):
    # computes hardlim function.
    # 1    if   x > 0
    # 0    otherwise
    return (x > 0).astype(np.int)

def plot_decision_boundary(W, xx):
    m = -W[0, 1] / W[0, 2]
    b = -W[0, 0] / W[0, 2]
    yy = m * xx + b
    plt.plot(xx, yy, 'k--')


X, y = data()
X = np.hstack((np.ones((X.shape[0], 1)), X)) # appends '1' to every data vector as "on-neuron"

W = np.array([[-3, 2, 2]])
z = np.dot(X, W.transpose())
h = hardlim(z)

print('predicted :', h.transpose())
print('expected :', y.transpose())
print('accuracy :', (y == h).mean())

for point, label in zip(X, y):
    plt.plot(point[1], point[2], 'bx' if label == 0 else 'ro', markersize=8)

t = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 10)
plot_decision_boundary(W, t)

plt.show()
