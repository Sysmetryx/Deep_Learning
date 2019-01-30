"""
@author: N. Barillot, N. Laporte, N. Thome
"""
from keras import Model
from keras.applications.resnet50 import ResNet50
from data_gen import PascalVOCDataGenerator
from keras.optimizers import SGD
import numpy as np
import os

# Hyper params, might need to change for specific machine.
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
default_batch_size = 24
default_data_dir = 'VOC2007/'

def getModel():
    """
    loads model from ResNet50.
    is extracted as method for readability
    :return: the model itself
    """
    model = ResNet50(include_top=True, weights='imagenet')
    return model


def main():
    """
    main function for exo 2
    """
    # Retrieving the model (same as Exo 1)
    model = getModel()
    # Delete the last layer of the network.
    model.layers.pop()
    model = Model(input=model.input, output=model.layers[-1].output)
    model.summary()
    # Setting parameters. batch_size is at 24 due to VRAM restriction on GTX970
    lr = 0.1
    batch_size = default_batch_size
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr, momentum=0.9),
                  metrics=['binary_accuracy'])
    data_generator_train = PascalVOCDataGenerator('trainval', default_data_dir)
    generator = data_generator_train.flow(batch_size=batch_size)
    # Initializing Depp Features and Labels Matrixes
    X_train = np.zeros((len(data_generator_train.images_ids_in_subset), 2048))
    Y_train = np.zeros((len(data_generator_train.images_ids_in_subset), 20))
    # Computing number of batches.
    nb_batches = int(len(data_generator_train.images_ids_in_subset)
                     / batch_size) + 1
    for i in range(nb_batches):
        # Extracting images and Label for each images
        X, y = next(generator)
        # Getting Deep Features
        y_pred = model.predict(X)
        X_train[i * batch_size:(i + 1) * batch_size, :] = y_pred
        Y_train[i * batch_size:(i + 1) * batch_size, :] = y
    # Same as before but for Test
    data_generator_test = PascalVOCDataGenerator('test', default_data_dir)
    generator = data_generator_test.flow(batch_size=batch_size)
    # Initilisation des matrices contenant les Deep Features et les labels
    X_test = np.zeros((len(data_generator_test.images_ids_in_subset), 2048))
    Y_test = np.zeros((len(data_generator_test.images_ids_in_subset), 20))
    # Computing number of batches.
    nb_batches = int(len(data_generator_test.images_ids_in_subset) /
                     batch_size) + 1
    for i in range(nb_batches):
        # Extracting images and Label for each images
        X, y = next(generator)
        # Getting Deep Features
        y_pred = model.predict(X)
        X_test[i * batch_size:(i + 1) * batch_size, :] = y_pred
        Y_test[i * batch_size:(i + 1) * batch_size, :] = y
    # Saving model
    outfile = 'DF_ResNet50_VOC2007'
    np.savez(outfile, X_train=X_train, Y_train=Y_train, X_test=X_test,
             Y_test=Y_test)


if __name__ == '__main__':
    main()
