"""
@author: N. Barillot, N. Laporte, N. Thome
"""

from keras.datasets import mnist
import matplotlib as mpl
import matplotlib.pyplot as plt

def loader():
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

def plot_data(X):
    mpl.use('TKAgg')
    plt.figure(figsize=(7.195, 3.841), dpi=100)
    # plotting the first 200 data
    for i in range(200):
        plt.subplot(10, 20, i+1)
        plt.imshow(X[i,:].reshape([28, 28]), cmap='gray')
        plt.axis('off')
    plt.show()

def main():
    X_train, y_train, X_test, y_test = loader()
    plot_data(X_train)


if __name__ == '__main__':
    main()
