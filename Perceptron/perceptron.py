from Data.get_data import get_normal_distribution2D as data
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


classes_mean = [
    [-3, -3],
    [4, 4]
                ]

classes_cov = [
    [[5, 0], [0, 5]],
    [[3, 0], [0, 3]]
                ]

X, d = data(classes_mean, classes_cov)
X = np.hstack((np.ones((X.shape[0], 1)), X)) # appends '1' to every data vector as "on-neuron"

W = np.random.random((1, 3))
learning_rate = 0.1

w_change = True
epoch = 0
while w_change and epoch < 200:
    w_change = False
    for j in range(X.shape[0]):
        z = np.dot(X[j], W.transpose())
        y = hardlim(z)

        if d[j] != y:
            W = W + (learning_rate * (d[j] - y) * X[j])
            w_change = True
    epoch += 1

z = np.dot(X, W.transpose())
y = hardlim(z)[:, 0]

print('weights :', W)
print('predicted :', d)
print('expected :', y)
print('accuracy :', (y == d).mean())

for point, label in zip(X, d):
    plt.plot(point[1], point[2], 'bx' if label == 0 else 'ro', markersize=8)

t = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 10)
plot_decision_boundary(W, t)
plt.ylim([X[:, 2].min() - 1, X[:, 2].max() + 1])
plt.xlim([X[:, 1].min() - 1, X[:, 1].max() + 1])

plt.show()
