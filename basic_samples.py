import numpy as np
from sklearn import datasets
import random

from network import Network


def train_on_iris():
    iris = datasets.load_iris()
    x = [np.reshape(d, (4, 1)) for d in iris.data]
    y = np.zeros((len(iris.target), 3, 1))
    for i in range(len(y)):
        y[i, iris.target[i] - 1] = 1
    data = list(zip(x, y))
    random.shuffle(data)
    train_data = data[:120]
    test_data = data[120:]

    net = Network([4, 16, 3])
    net.train(train_data, epochs=250, batch_size=120, learning=1, regularization=0.1, valid_data=test_data)


def train_on_xor():
    x = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    y = [0, 1, 1, 0]
    x = np.reshape(x, (4, 2, 1))
    y = np.reshape(y, (4, 1))
    train_data = list(zip(x, y))
    random.shuffle(train_data)

    x1, x2 = np.meshgrid((np.arange(80) - 40) / 13, (np.arange(80) - 40) / 13)
    x1 = np.reshape(x1, (6400, 1))
    x2 = np.reshape(x2, (6400, 1))
    noisy_input = [np.append(a1, a2) for a1, a2 in zip(x1, x2)]
    noisy_input = np.reshape(noisy_input, (6400, 2, 1))
    noisy_target = [(inp[0] > 0 > inp[1])-0 or (inp[0] < 0 < inp[1])-0 for inp in noisy_input]
    test_data = list(zip(noisy_input, noisy_target))

    net = Network([2, 16, 1])
    net.train(train_data, epochs=250, batch_size=4, learning=0.5, regularization=0.01, valid_data=test_data)
    net.plot_results(test_data, 'fig.png')


train_on_xor()
