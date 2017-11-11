import numpy as np
import random
import matplotlib.pyplot as plt


class Network(object):

    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(b, 1) for b in sizes[1:]]
        self.weights = [np.random.randn(w1, w2) for w1, w2 in zip(sizes[1:], sizes[:-1])]

    def forward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def train(self, train_data, epochs=10, batch_size=1, learning=1, valid_data=None):
        for i in range(epochs):
            random.shuffle(train_data)
            for j in range(int(np.ceil(len(train_data)/batch_size))):
                self.update_params(train_data[j*batch_size: j*batch_size+batch_size], learning)
            accuracy = 'Training Accuracy: {:.2f}%'.format(self.evaluate(train_data)*100)
            if valid_data:
                accuracy = accuracy + ', Validation Accuracy: {:.2f}%'.format(self.evaluate(valid_data)*100)
            print('Epoch {}. {}'.format(i, accuracy))

    def update_params(self, batch, learning):
        gradient_w, gradient_b = self.back_propagation(batch)
        self.weights = [w - learning * g_w / len(batch) for w, g_w in zip(self.weights, gradient_w)]
        self.biases = [b - learning * g_b / len(batch) for b, g_b in zip(self.biases, gradient_b)]

    def back_propagation(self, batch):
        gradient_w = [np.zeros(w.shape) for w in self.weights]
        gradient_b = [np.zeros(b.shape) for b in self.biases]

        for x, y in batch:
            a = [x]
            z = []
            for w, b in zip(self.weights, self.biases):
                z.append(np.dot(w, a[-1]) + b)
                a.append(sigmoid(z[-1]))

            delta = (a[-1] - y) * sigmoid_prime(z[-1])
            gradient_w[-1] += np.dot(delta, a[-2].transpose())
            gradient_b[-1] += delta

            for i in range(-2, -self.num_layers, -1):
                delta = np.dot(self.weights[i+1].transpose(), delta) * sigmoid_prime(z[i])
                gradient_w[i] += np.dot(delta, a[i-1].transpose())
                gradient_b[i] += delta

        return gradient_w, gradient_b

    def evaluate(self, test_data):
        if self.sizes[-1] == 1:
            return np.sum([y == np.around(self.forward(x)) for x, y in test_data]) / len(test_data)
        else:
            return np.sum([y[np.argmax(self.forward(x))] == 1 for x, y in test_data]) / len(test_data)

    def plot_results(self, test_data, file_name=None):
        for x, y in test_data:
            plt.scatter(x[0], x[1], c='r' if np.around(self.forward(x)) == 1 else 'g', marker='.')
        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name)


def sigmoid(z):
    return 1/(1+np.exp(-z))


def sigmoid_prime(z):
    return np.exp(-z)/((1+np.exp(-z))**2)
