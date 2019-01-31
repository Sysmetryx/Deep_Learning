import time
import numpy as np
import _pickle as pickle
from keras.models import model_from_yaml


def loadModel(savename):
    with open(savename+".yaml", "r") as yaml_file:
        model = model_from_yaml(yaml_file.read())
    print("Yaml Model ",savename,".yaml loaded ")
    model.load_weights(savename+".h5")
    print("Weights ",savename,".h5 loaded ")
    return model


def sampling(preds, temperature):
    preds2 = np.asarray(preds).astype('float64')
    preds = np.exp(preds2) / sum(np.exp(preds2))
    temperature = temperature
    predsN = pow(preds2, 1.0 / temperature)
    predsN /= np.sum(predsN)
    value = np.random.multinomial(1, predsN)
    return np.argmax(value)


def main():
    SEQLEN = 10
    outfile = "Baudelaire_len_"+str(SEQLEN)+".p"
    [index2char, X_train, y_train, X_test, y_test] = pickle.load(open(outfile, "rb"))
    model = loadModel("testmodel")
    nb_chars = len(index2char)
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    model.summary()
    # in order to reproduce
    seed = 15
    char_init = ""
    # for i in range(SEQLEN):
    #     char = index2char[np.argmax(X_train[seed, i, :])]
    #     char_init += char
    # print("CHAR INIT: " + char_init)
    test = np.zeros((1, SEQLEN, nb_chars), dtype=np.bool)
    test[0, :, :] = X_train[seed, :, :]
    # preds = np.random.random_sample(5)
    # preds = np.exp(preds) / sum(np.exp(preds))
    # temperature = 0.1
    # predsN = pow(preds, 1.0 / temperature)
    # predsN /= np.sum(predsN)
    nbgen = 400  # number of characters to generate (1,nb_chars)
    gen_char = char_init
    temperature = 0.01
    for i in range(nbgen):
        preds = model.predict(test)[0]  # shape (1,nb_chars)
        next_ind = sampling(preds, temperature)
        next_char = index2char[next_ind]
        gen_char += next_char
        for i in range(SEQLEN - 1):
            test[0, i, :] = test[0, i + 1, :]
        test[0, SEQLEN - 1, :] = 0
        test[0, SEQLEN - 1, next_ind] = 1
    print("Generated text: " + gen_char)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("-------% " + str(time.time() - start_time) + " secondes %-------")

