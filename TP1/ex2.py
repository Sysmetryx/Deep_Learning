"""
@author: N. Barillot, N. Laporte, N. Thome
"""
from keras.utils import np_utils
import numpy as np
from keras.datasets import mnist


def softmax_batch(batch, W, b):
    num = np.exp(np.dot(batch, W) + b)
    den = np.sum(num, axis=1)
    softmax = np.divide(num.T, den).T
    return softmax


def sigmoid_batch(batch, W, b):
    return 1 / (1 + np.exp(-(np.dot(batch, W) + b)))


def forward_batch(batch, W1, W2, b1, b2):
    sigmoid = sigmoid_batch(batch, W1, b1)
    return sigmoid, softmax_batch(sigmoid, W2, b2)


def backward_batch(y_hat_1, y_hat_2, y_pred, W2, b2, lr, W1, b1, batch, N):
    delta = y_hat_2 - y_pred
    W2_new = 1 / N * np.dot(y_hat_1, delta)
    W2 = W2 - lr * W2_new
    b2_new = 1 / N * sum(delta)
    b2 = b2 - lr * b2_new
    N = len(batch)
    y_hats = (y_hat_1 * (1 - y_hat_1))
    backprop = np.dot(delta, W2.T)
    delta1 = backprop * y_hats
    W1_new = 1 / N * np.dot(batch.T, delta1)
    W1 = W1 - lr * W1_new
    b1_new = 1 / N * sum(delta1)
    b1 = b1 - lr * b1_new
    return W1, W2, b1, b2


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


def zero_weights(d, K, L):
    """
    Sets all weights at 0
    :return: initialized wieghts
    """
    W1 = np.zeros((d, L))
    b1 = np.zeros((1, L))
    W2 = np.zeros((L, K))
    b2 = np.zeros((1, K))
    return W1, b1, W2, b2


def norm_weights(d, K, L, e_type):
    W1 = np.random.randn(d, L) * e_type
    b1 = np.random.randn(1, L) * e_type
    W2 = np.random.randn(L, K) * e_type
    b2 = np.random.randn(1, K) * e_type
    return W1, b1, W2, b2


def xavier_weights(d, K, L, e_type):
    W1 = np.divide(np.random.randn(d, L), np.sqrt(d))
    b1 = np.divide(np.random.randn(1, L), np.sqrt(d))
    W2 = np.divide(np.random.randn(L, K), np.sqrt(d))
    b2 = np.divide(np.random.randn(1, K), np.sqrt(d))
    return W1, b1, W2, b2


def accuracy(W1, W2, b1, b2, images, labels):
    """
    get the accuracy of the given model.
    """
    pred, result = forward_batch(images, W1, W2, b1, b2)
    return np.where(result.argmax(axis=1) != labels.argmax(axis=1), 0.,
                    1.).mean() * 100.0


def main(method):
    """
    trains the MLP and displays it's performances eval.
    :param method: method for weights initialization,
    'zero' -> all weights are 0
    'norm' -> weights chosen by sampling a normal distribution
    'xavier' -> weights chosen using Xavier method
    :return:
    """
    # loading data
    X_train, y_train, X_test, y_test = loader()
    # params for weights
    L = 100
    d = X_train.shape[1]
    K = 10
    e_type = 0.1
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, K)
    Y_test = np_utils.to_categorical(y_test, K)
    numEp = 20  # Number of epochs for gradient descent
    lr = 1  # Learning rate
    batch_size = 100
    # for complete dataset
    N = 60000
    nb_batches = int(float(N) / batch_size)
    # initializing :
    if method == 'norm':
        W1, b1, W2, b2 = norm_weights(d, K, L, e_type)
    elif method == 'xavier':
        W1, b1, W2, b2 = xavier_weights(d, K, L, e_type)
    else:
        W1, b1, W2, b2 = zero_weights(d, K, L)
    for epoch in range(numEp):
        print("Starting EPOCH " + str(epoch) + " With " + str(nb_batches) +
              " batches.")
        for ex in range(nb_batches):
            batch_start = ex * batch_size
            batch_end = ex * batch_size + batch_size
            batch = X_train[batch_start:batch_end]
            Y_batch = Y_train[batch_start:batch_end]
            predict1, predict2 = forward_batch(batch, W1, W2, b1, b2)
            W1, W2, b1, b2 = backward_batch(predict1, predict2, Y_batch, W2,
                                            b2, lr, W1, b1, batch, N)
    print(accuracy(W1,W2, b1, b2, X_test,Y_test))


if __name__ == '__main__':
    """
    Select a method :
    'zero'
    'norm'
    'xavier'
    anything else will use 'zero'
    """
    main('xavier')
