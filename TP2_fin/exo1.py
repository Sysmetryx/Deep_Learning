"""
@author: N. Barillot, N. Laporte, N. Thome
"""
from keras.datasets import mnist
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Sequential


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


def saveModel(model, savename):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(savename + ".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
        print("Yaml Model ", savename, ".yaml saved to disk")
    # serialize weights to HDF5
    model.save_weights(savename + ".h5")
    print("Weights ", savename, ".h5 saved to disk")


def main():
    # Loading dataset
    X_train, y_train, X_test, y_test = loader()
    # Creating empty model
    model = Sequential()
    # Adding fully connected layer
    model.add(Dense(10, input_dim=784, name='fc1'))
    # Adding an output layer with softmax activation
    model.add(Activation('softmax'))
    # Display network info
    model.summary()
    # Preparing network for train with learning rate
    learning_rate = 0.1
    sgd = SGD(learning_rate)
    # selecting training method
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])
    # formating data
    batch_size = 100
    nb_epoch = 20
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1)
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0] * 100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


if __name__ == '__main__':
    main()
