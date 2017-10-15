import numpy as np
import random


class Network(object):

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.biases = []
        self.biases.append(np.random.randn(hidden_size, 1))
        self.biases.append(np.random.randn(output_size, 1))
        self.weights = []
        self.weights.append(np.random.randn(hidden_size, input_size))
        self.weights.append(np.random.randn(output_size, hidden_size))

    def forward(self, x):
        y = sigmoid(np.dot(self.weights[0], x) + self.biases[0])
        y = sigmoid(np.dot(self.weights[1], y) + self.biases[1])
        return y

    def train(self, train_data, num_epoch, batch_size, learning, valid_data=None):
        for i in range(num_epoch):
            random.shuffle(train_data)
            for j in range(int(len(train_data)/batch_size)):
                self.update_params(train_data[j: j+batch_size], learning)
            accuracy = ''
            if len(valid_data) > 0:
                accuracy = '{}%'.format(self.test(valid_data)*100)
            print('Epoch {}. {}'.format(i, accuracy))

    def update_params(self, batch, learning):
        gradient_w0, gradient_w1, gradient_b0, gradient_b1 = self.back_propagation(batch)
        self.weights[0] = self.weights[0] - learning * gradient_w0
        self.weights[1] = self.weights[1] - learning * gradient_w1
        self.biases[0] = self.biases[0] - learning * gradient_b0
        self.biases[1] = self.biases[1] - learning * gradient_b1

    def back_propagation(self, batch):
        gradient_w0 = np.zeros(self.weights[0].shape)
        gradient_w1 = np.zeros(self.weights[1].shape)
        gradient_b0 = np.zeros(self.biases[0].shape)
        gradient_b1 = np.zeros(self.biases[1].shape)

        for x, y in batch:
            x = np.reshape(x, (len(x), 1))
            y = np.reshape(y, (len(y), 1))
            a = [x]
            z = [np.dot(self.weights[0], x) + self.biases[0]]
            a.append(sigmoid(z[0]))
            z.append(np.dot(self.weights[1], a[1]) + self.biases[1])
            a.append(sigmoid(z[1]))

            delta1 = (a[2] - y) * sigmoid_prime(z[1])
            gradient_w1 += np.dot(delta1, a[1].transpose())
            gradient_b1 += delta1

            delta2 = np.dot(self.weights[1].transpose(), delta1) * sigmoid_prime(z[0])
            gradient_w0 += np.dot(delta2, a[0].transpose())
            gradient_b0 += delta2

        return gradient_w0, gradient_w1, gradient_b0, gradient_b1

    def test(self, test_data):
        correct_results = 0
        for x, y in test_data:
            x = np.reshape(x, (len(x), 1))
            y = np.reshape(y, (len(y), 1))
            if y[np.argmax(self.forward(x))] == 1:
                correct_results += 1
        return correct_results/len(test_data)


def sigmoid(z):
    return 1/(1+np.exp(-z))


def sigmoid_prime(z):
    return np.exp(-z)/((1+np.exp(-z))**2)
