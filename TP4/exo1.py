import _pickle as pickle
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop


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
    SEQLEN = 10
    nb_chars = 60
    outfile = "Baudelaire_len_" + str(SEQLEN) + ".p"
    [index2char, X_train, y_train, X_test, y_test] = pickle.load(open(
            outfile, "rb"))
    model = Sequential()
    HSIZE = 128
    model.add(SimpleRNN(HSIZE, return_sequences=False,
                        input_shape=(SEQLEN, nb_chars), unroll=True))
    model.add(Dense(nb_chars))
    model.add(Activation("softmax"))
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    learning_rate = 0.001
    optim = RMSprop(lr=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=optim,
                  metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)
    scores_train = model.evaluate(X_train, y_train, verbose=1)
    scores_test = model.evaluate(X_test, y_test, verbose=1)
    print("PERFS TRAIN: %s: %.2f%%" % (
    model.metrics_names[1], scores_train[1] * 100))
    print("PERFS TEST: %s: %.2f%%" % (
    model.metrics_names[1], scores_test[1] * 100))
    saveModel(model, "testmodel")

if __name__ == '__main__':
    main()
