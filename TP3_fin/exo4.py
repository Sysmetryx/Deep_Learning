"""
@author: N. Barillot, N. Laporte, N. Thome
"""
from keras import Model
from keras.applications.resnet50 import ResNet50
from data_gen import PascalVOCDataGenerator
from keras.optimizers import SGD
import numpy as np
from keras.layers import Dense
from sklearn.metrics import average_precision_score
import os


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
default_batch_size = 16
default_data_dir = 'VOC2007/'


def getModel():
    """
    loads model from ResNet50.
    is extracted as method for readability
    :return: the model itself
    """
    model = ResNet50(include_top=True, weights='imagenet')
    return model


def evaluate(model, subset, batch_size=default_batch_size,
             data_dir=default_data_dir):
    """evaluate
    Compute the mean Average Precision metrics on a subset with a given model

    :param model: the model to evaluate
    :param subset: the data subset
    :param batch_size: the batch which will be use in the data generator
    :param data_dir: the directory where the data is stored
    :param verbose: display a progress bar or not, default is no (0)
    """
    # Create the generator on the given subset
    data_generator = PascalVOCDataGenerator(subset, data_dir)
    steps_per_epoch = int(len(data_generator.id_to_label) / batch_size) + 1
    # Get the generator
    generator = data_generator.flow(batch_size=batch_size)
    y_all = []
    y_pred_all = []
    for i in range(steps_per_epoch):
        # Get the next batch
        X, y = next(generator)
        y_pred = model.predict(X)
        # We concatenate all the y and the prediction
        for y_sample, y_pred_sample in zip(y, y_pred):
            y_all.append(y_sample)
            y_pred_all.append(y_pred_sample)
    y_all = np.array(y_all)
    y_pred_all = np.array(y_pred_all)
    # Now we can compute the AP for each class
    AP = np.zeros(data_generator.nb_classes)
    for cl in range(data_generator.nb_classes):
        AP[cl] = average_precision_score(y_all[:, cl], y_pred_all[:, cl])
    return AP


def main():
    model = getModel()
    model.layers.pop()
    x = model.layers[-1].output
    data_dir = default_data_dir
    data_generator_train = PascalVOCDataGenerator('trainval', data_dir)
    x = Dense(data_generator_train.nb_classes, activation='sigmoid', name='predictions')(x)
    model = Model(input=model.input, output=x)
    for i in range(model.layers.__len__()):
        model.layers[i].trainable = True
    lr = 0.1
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=lr),
                  metrics=['binary_accuracy'])
    batch_size = default_batch_size
    nb_epochs = 10
    data_generator_train = PascalVOCDataGenerator('trainval', data_dir)
    steps_per_epoch_train = int(len(data_generator_train.id_to_label) / batch_size) + 1
    model.fit_generator(data_generator_train.flow(batch_size=batch_size),
                        steps_per_epoch=steps_per_epoch_train,
                        epochs=nb_epochs)
    print(evaluate(model, 'test', batch_size=default_batch_size,
              data_dir=default_data_dir))


if __name__ == '__main__':
    main()
