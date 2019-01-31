from keras.models import model_from_yaml
from keras.optimizers import Adam
import numpy as np
import _pickle as pickle
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import pandas as pd


def loadModel(savename):
    with open(savename + ".yaml", "r") as yaml_file:
        model = model_from_yaml(yaml_file.read())
    print("Yaml Model ", savename, ".yaml loaded ")
    model.load_weights(savename + ".h5")
    print("Weights ", savename, ".h5 loaded ")
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
    nbkeep = 1000
    # LOADING MODEL
    nameModel = 'trained_model'
    model = loadModel(nameModel)
    optim = Adam()
    model.compile(loss="categorical_crossentropy", optimizer=optim,
                  metrics=['accuracy'])
    # LOADING TEST DATA
    outfile = 'test_data_' + str(nbkeep) + '.npz'
    npzfile = np.load(outfile)
    X_test = npzfile['X_test']
    Y_test = npzfile['Y_test']
    outfile = "Caption_Embeddings_" + str(nbkeep) + ".p"
    [listwords, embeddings] = pickle.load(open(outfile, "rb"))
    indexwords = {}
    for i in range(len(listwords)):
        indexwords[listwords[i]] = i
    ind = np.random.randint(X_test.shape[0])
    filename = 'flickr_8k_test_dataset.txt'  # PATH IF NEEDED
    df = pd.read_csv(filename, delimiter='\t')
    iter = df.iterrows()
    for i in range(ind + 1):
        x = iter.__next__()
    imname = x[1][0]
    print("image name=" + imname + " caption=" + x[1][1])
    dirIm = "data/flickr8k/Flicker8k_Dataset/"  # CHANGE WITH YOUR DATASET
    img = mpimg.imread(dirIm + imname)
    plt.figure(dpi=100)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    pred = model.predict(X_test[ind:ind + 1, :, :])
    nbGen = 5
    temperature = 0.1  # Temperature param for peacking soft-max distribution
    for s in range(nbGen):
        wordpreds = "Caption n° " + str(s + 1) + ": "
        indpred = sampling(pred[0, 0, :], temperature)
        wordpred = listwords[indpred]
        wordpreds += str(wordpred) + " "
        X_test[ind:ind + 1, 1, 100:202] = embeddings[indexwords[wordpred], :]
        # COMPLETE WITH YOUR CODE
        cpt = 1
        while (str(wordpred) != '<end>' and cpt < 30):
            pred = model.predict(X_test[ind:ind + 1, :, :])
            indpred = sampling(pred[0, cpt, :], temperature)
            wordpred = listwords[indpred]
            wordpreds += str(wordpred) + " "
            cpt += 1
            X_test[ind:ind + 1, cpt, 100:202] = \
                embeddings[indexwords[wordpred], :]  # COMPLETE WITH YOUR CODE
        print(wordpreds)


if __name__ == '__main__':
    main()
