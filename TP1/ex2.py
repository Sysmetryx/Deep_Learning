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


if __name__ == '__main__':
    main()
