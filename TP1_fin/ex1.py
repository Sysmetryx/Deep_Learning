"""
@author: N. Barillot, N. Laporte, N. Thome
"""
from keras.utils import np_utils
import numpy as np
from keras.datasets import mnist

def loader():
    """
    loads the data from Mnist and returns all sets.
    """
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    return X_train, y_train, X_test, y_test


def softmax(X, b):
    """
    Applies a softmax
    """
    num = np.exp(X + b)
    den = np.sum(np.exp(X + b), axis=1)[:, None]
    return num / den


def accuracy(W, b, images, labels):
    """
    get the accuracy of the given model.
    """
    pred = forward(images, W, b)
    return np.where(pred.argmax(axis=1) !=
                    labels.argmax(axis=1), 0., 1.).mean()*100.0


def forward(batch, W, b):
    """
    forward method, returning the predicted label.
    """
    s = np.matmul(batch, W)
    y_hat = softmax(s, b)
    return y_hat


def backward(batch, y_star, y_hat, batch_size):
    """
    back propagation of error gradiant.
    """
    grad_w = (1/batch_size) * np.dot(batch.T, (y_hat - y_star))
    grad_b = (1/batch_size) * np.sum(y_hat - y_star, axis=0)
    return grad_w, grad_b


def update(grad_w, W, grad_b, b, learning_rate):
    """
    returns updated weight matrix
    """
    return W - learning_rate * grad_w, b - learning_rate * grad_b


def main():
    X_train, y_train, X_test, y_test = loader()
    K = 10
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, K)
    Y_test = np_utils.to_categorical(y_test, K)
    d = X_train.shape[1]
    W = np.zeros((d, K))
    b = np.zeros((1, K))
    numEp = 20  # Number of epochs for gradient descent
    eta = 1e-1  # Learning rate
    batch_size = 100
    # for complete dataset
    N = 60000
    nb_batches = int(float(N) / batch_size)
    grad_w = np.zeros((d, K))
    grad_b = np.zeros((1, K))
    for epoch in range(numEp):
        for ex in range(nb_batches):
            batch = X_train[batch_size * ex:batch_size * (ex + 1)]
            labels = Y_train[batch_size * ex:batch_size * (ex + 1)]
            y_hat = forward(batch, W, b)
            grad_w, grad_b = backward(batch, labels, y_hat, batch_size)
            W, b = update(grad_w, W, grad_b, b, eta)
    print("Accuracy on train dataset = " +
          str(accuracy(W, b, X_train, Y_train)))
    print("Accuracy on test dataset = " + str(accuracy(W, b, X_test, Y_test)))


if __name__ == '__main__':
    main()
