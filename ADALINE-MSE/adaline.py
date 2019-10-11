from Data.get_data import get_normal_distribution2D as data
import numpy as np
import matplotlib.pyplot as plt

def hardlim(x):
    # computes hardlim function.
    # 1    if   x > 0
    # -1    otherwise
    return (x > 0).astype(np.int) * 2 - 1

def plot_decision_boundary(W, xx):
    m = -W[0, 1] / W[0, 2]
    b = -W[0, 0] / W[0, 2]
    yy = m * xx + b
    plt.plot(xx, yy, 'k--')

def unison_shuffle(a, b):
    assert a.shape[0] == b.shape[0]
    p = np.random.permutation(a.shape[0])
    return a[p], b[p]

classes_mean = [
    [-1, -1],
    [1, 1]
                ]

classes_cov = [
    [[0.4, 0], [0, 0.4]],
    [[0.2, 0], [0, 0.2]]
                ]

X, d = data(classes_mean, classes_cov)
d[d == 0] = -1
X = np.hstack((np.ones((X.shape[0], 1)), X)) # appends '1' to every data vector as "on-neuron"

W = np.random.random((1, 3))
# for 'AND' and 'OR' gates data, learning rate should be large. like 0.01
# but for normal distribution data, learning rate should be small. like 0.0001
learning_rate = 0.0001
w_change_max = 1
epoch = 0

z = np.dot(X, W.transpose())[:, 0]
y = hardlim(z)
base_error = np.mean((d - z) ** 2) / 2
base_acc = (d == y).mean() * 100
error = [base_error]
acc = [base_acc]

while w_change_max > 0.0001 and epoch < 500:
    w_change_max = 0

    X, d = unison_shuffle(X, d)

    for i in range(X.shape[0]):
        z = np.dot(X[i], W.transpose())
        y = hardlim(z)

        delta_w = (learning_rate * (d[i] - z) * X[i])

        W = W + delta_w
        # print(abs(delta_w))
        if w_change_max < max(abs(delta_w)):
            w_change_max = max(abs(delta_w))

    # end of epoch:
    epoch += 1
    z = np.dot(X, W.transpose())[:, 0]
    y = hardlim(z)

    epoch_error = np.mean((d - z) ** 2) / 2
    epoch_acc = (d == y).mean() * 100

    error.append(epoch_error)
    acc.append(epoch_acc)

z = np.dot(X, W.transpose())
y = hardlim(z)[:, 0]

print('weights :', W)
print('predicted :', y)
print('expected :', d)
print('accuracy :', (y == d).mean() * 100, '%')

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(acc, 'bo--', label='accuracy')
plt.ylabel('accuracy(%)')
plt.xlabel('epoch')
plt.subplot(1, 2, 2)
plt.plot(error, 'ro--', label='MSE')
plt.ylabel('MSE')
plt.xlabel('epoch')

plt.figure()
for point, label in zip(X, d):
    plt.plot(point[1], point[2], 'bx' if label == 1 else 'ro', markersize=8)
t = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 10)
plot_decision_boundary(W, t)
plt.ylim([X[:, 2].min() - 1, X[:, 2].max() + 1])
plt.xlim([X[:, 1].min() - 1, X[:, 1].max() + 1])

plt.show()
