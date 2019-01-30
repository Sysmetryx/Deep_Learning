"""
@author: N. Barillot, N. Laporte, N. Thome
"""
from keras.optimizers import SGD
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import average_precision_score
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
default_batch_size = 200
default_data_dir = 'VOC2007/'


def main():
    # Loading file from exo2
    outfile = 'DF_ResNet50_VOC2007.npz'
    npzfile = np.load(outfile)
    X_train = npzfile['X_train']
    Y_train = npzfile['Y_train']
    X_test = npzfile['X_test']
    Y_test = npzfile['Y_test']
    print("X_train=", X_train.shape, "Y_train=", Y_train.shape, " X_test=", X_test.shape, "Y_train=", Y_test.shape)
    # Creating model.
    model = Sequential()
    model.add(Dense(20, input_dim=2048, name='fc1', activation='sigmoid'))
    model.summary()
    learning_rate = 0.1
    sgd = SGD(learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['binary_accuracy'])
    batch_size = 32
    nb_epoch = 20
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1)
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("%s TEST: %.2f%%" % (model.metrics_names[0], scores[0] * 100))
    print("%s TEST: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    AP_train = np.zeros(20)
    AP_test = np.zeros(20)
    for c in range(20):
        AP_train[c] = average_precision_score(Y_train[:, c], y_pred_train[:, c])
        AP_test[c] = average_precision_score(Y_test[:, c], y_pred_test[:, c])
    print("MAP TRAIN =", AP_train.mean() * 100)
    print("MAP TEST =", AP_test.mean() * 100)


if __name__ == '__main__':
    main()
