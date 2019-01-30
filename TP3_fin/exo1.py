"""
@author: N. Barillot, N. Laporte, N. Thome
"""
from keras.applications.resnet50 import ResNet50


def getModel():
    """
    loads model from ResNet50.
    is extracted as method for readability
    :return: the model itself
    """
    model = ResNet50(include_top=True, weights='imagenet')
    return model


def main():
    model = getModel()
    model.summary()


if __name__ == '__main__':
    main()
