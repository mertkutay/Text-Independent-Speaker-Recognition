import numpy as np
from sklearn import datasets
import random

from network import Network

iris = datasets.load_iris()
x = iris.data
y = np.zeros((len(iris.target), 3))

for i in range(len(y)):
    y[i, iris.target[i] - 1] = 1

data = list(zip(x, y))
random.shuffle(data)
train_data = data[:120]
test_data = data[120:]

net = Network(4, 10, 3)
net.train(train_data, 150, 1, 0.5, test_data)
