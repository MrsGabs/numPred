import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# lê o dataset de MNIST
data = pd.read_csv("coloque o path do MNIST.csv")


# usado para facilitar a manipulação dos dados e permitir o uso de algebra
data = np.array(data)
m, n = data.shape  # m é a quantidade de linhas / n é a quantidade de colunas
np.random.shuffle(data)  # embaralha os dados do dataset


# analisa os primeiros 1000 exemplos embaralhados e transpõe a matrix
data_dev = data[0:1000].T
y_dev = data_dev[0]  # primeira linha
x_dev = data_dev[1:n]  # primeira coluna até a ultima
x_dev = x_dev / 255.

data_train = data[1000:m].T
y_train = data_train[0]
x_train = data_train[1:n]
x_train = x_train / 255.
_, m_train = x_train.shape


def init_param():
    W1 = np.random.rand(10, 784) - 0.5
    B1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    B2 = np.random.rand(10, 1) - 0.5
    return W1, B1, W2, B2


def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(W1, B1, W2, B2, x):
    Z1 = W1.dot(x) + B1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def deriv_ReLU(Z):
    return Z > 0


def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


def back_prop(Z1, A1, Z2, A2, W1, W2, x, y):
    one_hot_y = one_hot(y)
    dZ2 = A2 - one_hot_y
    dW2 = 1 / m * dZ2.dot(A1.T)
    dB2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(x.T)
    dB1 = 1 / m * np.sum(dZ1)
    return dW1, dB1, dW2, dB2


def update_params(W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha):
    W1 = W1 - alpha * dW1
    B1 = B1 - alpha * dB1
    W2 = W2 - alpha * dW2
    B2 = B2 - alpha * dB2
    return W1, B1, W2, B2


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size


def gradient_descent(x, y, alpha, iterations):
    W1, B1, W2, B2 = init_param()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, B1, W2, B2, x)
        dW1, dB1, dW2, dB2 = back_prop(Z1, A1, Z2, A2, W1, W2, x, y)
        W1, B1, W2, B2 = update_params(
            W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha)
        if i % 10 == 0:
            print('Iteration: ', i)
            predictions = get_predictions(A2)
            print("Accuracy: ", get_accuracy(predictions, y))
    return W1, B1, W2, B2


W1, B1, W2, B2 = gradient_descent(x_train, y_train, 0.10, 500)


def make_predictions(x, W1, B1, W2, B2):
    _, _, _, A2 = forward_prop(W1, B1, W2, B2, x)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, B1, W2, B2):
    current_image = x_train[:, index, None]
    prediction = make_predictions(x_train[:, index, None], W1, B1, W2, B2)
    label = y_train[index]
    print("Prediction: ", prediction)
    print("Número: ", label)

    current_image = current_image.reshape((28, 28))*255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


test_prediction(28, W1, B1, W2, B2)

# dev_predictions = make_predictions(x_dev, W1, B1, W2, B2)
# get_accuracy(dev_predictions, y_dev)
