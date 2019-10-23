import numpy as np
import random
import matplotlib.pyplot as plt
import json


class Network(object):

    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(b, 1) for b in sizes[1:]]
        self.weights = [np.random.randn(w1, w2)/(10*w2) for w1, w2 in zip(sizes[1:], sizes[:-1])]

    def forward(self, a):
        for i in range(self.num_layers-1):
            if i == self.num_layers - 2:
                a = sigmoid(np.dot(self.weights[i], a) + self.biases[i])
            else:
                a = relu(np.dot(self.weights[i], a) + self.biases[i])
        return a

    def train(self, train_data, epochs=1, batch_size=32, learning=1, regularization=0, valid_data=None):
        for i in range(1, epochs + 1):
            n = len(train_data)
            random.shuffle(train_data)
            for j in range(int(np.ceil(n / batch_size))):
                batch = train_data[j*batch_size: j*batch_size+batch_size]
                self.update_params(batch, learning, regularization)
                # accuracy = 'Epoch {}: {}/{} loss: {:.4f} acc: {:.4f}'.format(i, j * batch_size, n,
                #                                                              self.total_cost(train_data, regularization),
                #                                                              self.evaluate(train_data))
                # print(accuracy)

            accuracy = 'Epoch {}: loss: {:.4f} acc: {:.4f}'.format(i, self.total_cost(train_data, regularization),
                                                                   self.evaluate(train_data))
            if valid_data:
                accuracy += ', val loss: {:.4f} - val acc: {:.4f}'.format(self.total_cost(valid_data, regularization),
                                                                          self.evaluate(valid_data))
            print(accuracy)

    def update_params(self, batch, learning, regularization):
        gradient_w, gradient_b = self.back_propagation(batch)
        self.weights = [(1 - learning * regularization / len(batch)) * w - learning * g_w / len(batch)
                        for w, g_w in zip(self.weights, gradient_w)]
        self.biases = [b - learning * g_b / len(batch) for b, g_b in zip(self.biases, gradient_b)]

    def back_propagation(self, batch):
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        gradient_b = [np.zeros(b.shape) for b in self.biases]

        for x, y in batch:
            a = [x]
            z = []
            for w, b in zip(self.weights, self.biases):
                z.append(np.dot(w, a[-1]) + b)
                if len(z) != len(self.weights):
                    a.append(sigmoid(z[-1]))
                else:
                    a.append(relu(z[-1]))

            delta = (a[-1] - y) # * sigmoid_prime(z[-1])
            gradient_w[-1] += np.dot(delta, a[-2].transpose())
            gradient_b[-1] += delta

            for i in range(-2, -self.num_layers, -1):
                delta = np.dot(self.weights[i+1].transpose(), delta) * relu_prime(z[i])
                gradient_w[i] += np.dot(delta, a[i-1].transpose())
                gradient_b[i] += delta

        return gradient_w, gradient_b

    def evaluate(self, test_data):
        if self.sizes[-1] == 1:
            return np.sum([y == np.round(self.forward(x)) for x, y in test_data]) / len(test_data)
        else:
            return np.sum([y[np.argmax(self.forward(x))] == 1 for x, y in test_data]) / len(test_data)

    def plot_results(self, test_data, file_name=None):
        for x, y in test_data:
            plt.scatter(x[0], x[1], c='r' if np.around(self.forward(x)) == 1 else 'g', marker='.')
        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name)

    def total_cost(self, data, regularization):
        cost = 0.0
        for x, y in data:
            a = self.forward(x)
            cost += cross_entropy_cost(a, y) / len(data)
        cost += 0.5 * (regularization / len(data)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases]}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    net = Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


def cross_entropy_cost(a, y):
    return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))


def cross_entropy_cost_delta(a, y):
    return a - y


def relu(z):
    return np.maximum(np.zeros((len(z), 1)), z)


def relu_prime(z):
    x = np.sign(z)
    return relu(x)


def sigmoid(z):
    return .5 * (1 + np.tanh(.5 * z))


def sigmoid_prime(z):
    return np.exp(-z)/((1+np.exp(-z))**2)
