import gc
import numpy as np
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
from keras.layers import Dense, Activation, Masking
from keras.optimizers import Adam
from keras.models import model_from_yaml
from keras import backend as K


def saveModel(model, savename):
  # serialize model to YAML
  model_yaml = model.to_yaml()
  with open(savename + ".yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
    print("Yaml Model ", savename, ".yaml saved to disk")
    # serialize weights to HDF5
  model.save_weights(savename + ".h5")
  print("Weights ", savename, ".h5 saved to disk")


def evaluate_model_train():
    data = np.load("train_data_1000.npz")
    X_test = data["X_train"]
    Y_test = data["Y_train"]
    with open("trained_model.yaml", "r") as yaml_file:
        model = model_from_yaml(yaml_file.read())
    model.load_weights("trained_model"+".h5")
    model.compile(loss='categorical_crossentropy', optimizer=Adam(),
                  metrics=['accuracy'])
    print(model.evaluate(X_test, Y_test))


def evaluate_model_test():
    data = np.load("test_data_1000.npz")
    X_test = data["X_test"]
    Y_test = data["Y_test"]
    with open("trained_model.yaml", "r") as yaml_file:
        model = model_from_yaml(yaml_file.read())
    model.load_weights("trained_model" + ".h5")
    model.compile(loss='categorical_crossentropy', optimizer=Adam(),
                  metrics=['accuracy'])
    print(model.evaluate(X_test, Y_test))


def train_model():
    data = np.load("train_data_1000.npz")
    X_train = data["X_train"]
    y_train = data["Y_train"]
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    SEQLEN = 39
    HSIZE = 100
    nb_chars = 1000
    model = Sequential()
    model.add(Masking(mask_value=0.0))
    model.add(SimpleRNN(HSIZE, return_sequences=True, unroll=True,
                        input_shape=(SEQLEN, nb_chars)))
    model.add(Dense(nb_chars))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(),
                  metrics=['accuracy'])
    # model.summary()
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)
    return X_train, model, y_train


def main():
    gc.enable()
    X_train, model, y_train = train_model()
    saveModel(model, 'trained_model')
    # Ugly mess starts here, but VRAM issues were serious when running this
    # one. If not needed (e.g. you have 10GB VRAM or more), comment those
    # lines.
    del model
    del X_train
    del y_train
    # Somehow GC calls sometimes don't register.
    for i in range(10):
        gc.collect()
    # Should flush Keras session memory, thus freeing VRAM
    K.clear_session()
    # Ugly mess ends here... for now.
    print("Eval with Train data : ")
    evaluate_model_train()
    print("Eval with Test data : ")
    evaluate_model_test()


if __name__ == '__main__':
    main()

