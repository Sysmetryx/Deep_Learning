"""
@author: N. Barillot, N. Laporte, N. Thome
"""
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from scipy.spatial import ConvexHull


def convexHulls(points, labels):
    # computing convex hulls for a set of points with asscoiated labels
    convex_hulls = []
    for i in range(10):
        convex_hulls.append(ConvexHull(points[labels==i, :]))
    return convex_hulls


def best_ellipses(points, labels):
    # computing best fiiting ellipse for a set of points with asscoiated labels
    gaussians = []
    for i in range(10):
        gaussians.append(GaussianMixture(n_components=1,
                         covariance_type='full').fit(points[labels == i, :]))
    return gaussians


def neighboring_hit(points, labels):
  k = 6
  nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points)
  distances, indices = nbrs.kneighbors(points)
  txs = 0.0
  txsc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  nppts = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  for i in range(len(points)):
    tx = 0.0
    for j in range(1,k+1):
      if (labels[indices[i,j]]== labels[i]):
        tx += 1
    tx /= k
    txsc[labels[i]] += tx
    nppts[labels[i]] += 1
    txs += tx
  for i in range(10):
    txsc[i] /= nppts[i]
  return txs / len(points)


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
    model = Sequential()
    x_train, y_train, x_test, y_test = loader()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    conv = Conv2D(16, kernel_size=(5, 5), activation='relu',
             input_shape=input_shape,
           padding='valid')
    model.add(conv)
    pool = MaxPooling2D(pool_size=(2, 2))
    model.add(pool)
    conv2 = Conv2D(32, kernel_size=(5, 5), activation='relu',
             input_shape=input_shape,
           padding='valid')
    model.add(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))
    model.add(pool2)
    model.add(Flatten())
    model.add(Dense(100, input_dim=784, name='fc1'))
    model.add(Activation('sigmoid'))
    model.add(Dense(10, name='Out'))
    model.add(Activation('softmax'))
    model.summary()
    batch_size = 100
    nb_epoch = 10
    # Preparing network for train with learning rate
    learning_rate = .1
    sgd = SGD(learning_rate)
    # selecting training method
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1)
    scores = model.evaluate(x_test, Y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0] * 100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    saveModel(model, "ex3model")


if __name__ == '__main__':
    main()
