from keras.optimizers import Adam
from keras.models import model_from_yaml
import pandas as pd
import numpy as np
import nltk
import _pickle as pickle


def loadModel(savename):
    with open(savename+".yaml", "r") as yaml_file:
        model = model_from_yaml(yaml_file.read())
    print("Yaml Model ", savename, ".yaml loaded ")
    model.load_weights(savename + ".h5")
    print("Weights ", savename, ".h5 loaded ")
    return model


def main():
    # LOADING TEST DATA
    nbkeep = 1000
    outfile = ""  # REPLACE WITH YOUR DATA PATH
    outfile += 'test_data_'+str(nbkeep)+'.npz'
    npzfile = np.load(outfile)
    X_test = npzfile['X_test']
    Y_test = npzfile['Y_test']
    # LOADING MODEL
    nameModel = "trained_model"
    model = loadModel(nameModel)
    # COMPILING MODEL
    optim = Adam()
    model.compile(loss="categorical_crossentropy", optimizer=optim,
                  metrics=['accuracy'])
    scores_test = model.evaluate(X_test, Y_test, verbose=1)
    print("PERFS TEST: %s: %.2f%%" % (model.metrics_names[1],
                                      scores_test[1]*100))
    # LOADING TEXT EMBEDDINGS
    outfile = "Caption_Embeddings_"+str(nbkeep)+".p"
    [listwords, embeddings] = pickle.load(open(outfile, "rb"))
    indexwords = {}
    for i in range(len(listwords)):
        indexwords[listwords[i]] = i
    # COMPUTING CAPTION PREDICTIONS ON TEST SET
    predictions = []
    nbTest = X_test.shape[0]
    for i in range(0, nbTest, 5):
        pred = model.predict(X_test[i:i + 1, :, :])
        wordpreds = []
        indpred = np.argmax(pred[0, 0, :])
        wordpred = listwords[indpred]
        wordpreds.append(str(wordpred))
        X_test[i, 1, 100:202] = embeddings[indpred]
        cpt = 1
        while str(wordpred) != '<end>' and cpt < (X_test.shape[1] - 1):
            pred = model.predict(X_test[i: i + 1, :, :])
            indpred = np.argmax(pred[0, cpt, :])
            wordpred = listwords[indpred]
            if wordpred != '<end>':
                wordpreds.append(str(wordpred))
            cpt += 1
            X_test[i, cpt, 100:202] = embeddings[indpred]
        if i % 1000 == 0:
            print("i=" + str(i) + " " + str(wordpreds))
        predictions.append(wordpreds)
    # LOADING GROUD TRUTH CAPTIONS ON TEST SET
    references = []
    filename = 'flickr_8k_test_dataset.txt'
    df = pd.read_csv(filename, delimiter='\t')
    iter = df.iterrows()
    ccpt = 0
    for i in range(int(np.floor(nbTest/5))):
        captions_image = []
        for j in range(5):
            x = iter.__next__()
            ll = x[1][1].split()
            caption = []
            for k in range(1, len(ll)-1):
                caption.append(ll[k])
            captions_image.append(caption)
            ccpt += 1
        references.append(captions_image)
    # COMPUTING BLUE-1, BLUE-2, BLUE-3, BLUE-4
    blue_scores = np.zeros(4)
    weights = np.zeros((4, 4))
    weights[0, 0] = 1
    weights[1, 0] = 0.5
    weights[1, 1] = 0.5
    weights[2, 0] = 1.0/3.0
    weights[2, 1] = 1.0/3.0
    weights[2, 2] = 1.0/3.0
    weights[3, :] = 1.0/4.0
    for i in range(4):
        blue_scores[i] = nltk.translate.bleu_score\
            .corpus_bleu(references, predictions, weights=
            (weights[i, 0], weights[i, 1], weights[i, 2], weights[i, 3]))
        print("blue_score - " + str(i) + "=" + str(blue_scores[i]))


if __name__ == '__main__':
    main()
