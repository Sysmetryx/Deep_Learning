import pandas as pd
import _pickle as pickle
import numpy as np


def load_data(dataset):
    filename = 'flickr_8k_' + dataset + '_dataset.txt'
    df = pd.read_csv(filename, delimiter='\t')
    return df


def save_embeddings(dataset, data, labels, nbkeep):
    outfile = str(dataset) + '_data_' + str(nbkeep)
    if str(dataset) == 'train':
        np.savez(outfile, X_train=data, Y_train=labels)  # Saving tensor
    elif str(dataset) == 'test':
        np.savez(outfile, X_test=data, Y_test=labels)  # Saving tensor
    else:
        print("Undefined dataset : " + str(dataset))


def embedd(dataset):
    df = load_data(dataset)
    nbTrain = df.shape[0]
    iter = df.iterrows()
    caps = []  # Set of captions
    imgs = []  # Set of images
    for i in range(nbTrain):
        x = iter.__next__()
        caps.append(x[1][1])
        imgs.append(x[1][0])
    outfile = 'Caption_Embeddings.p'
    [listwords, embeddings] = pickle.load(open(outfile, "rb"))
    maxLCap = 0
    for caption in caps:
        l = 0
        words_in_caption = caption.split()
        for j in range(len(words_in_caption) - 1):
            current_w = words_in_caption[j].lower()
            if current_w in listwords:
                l += 1
        if (l > maxLCap):
            maxLCap = l
    print("max caption length =" + str(maxLCap))
    nbkeep = 1000
    outfile = "Caption_Embeddings_" + str(nbkeep) + ".p"
    # Loading reduced dictionary
    [listwords, embeddings] = pickle.load(open(outfile, "rb"))
    indexwords = {} # Useful for tensor filling
    for i in range(len(listwords)):
        indexwords[listwords[i]] = i
    # Loading images features
    encoded_images = pickle.load(open("encoded_images_PCA.p", "rb"))
    # Allocating data and labels tensors
    tinput = 202
    tVocabulary = len(listwords)
    X = np.zeros((nbTrain, maxLCap, tinput))
    Y = np.zeros((nbTrain, maxLCap, tVocabulary), bool)
    for i in range(nbTrain):
        words_in_caption = caps[i].split()
        indseq = 0
        # current sequence index (to handle mising words in reduced dictionary)
        for j in range(len(words_in_caption) - 1):
            current_w = words_in_caption[j].lower()
            if (current_w in listwords):
                index = indexwords[current_w]
                img = imgs[i]
                X[i, indseq, 0:100] = encoded_images[img]
                X[i, indseq, 100:202] = embeddings[indexwords[current_w], :]
            next_w = words_in_caption[j + 1].lower()
            if next_w in listwords:
                index_pred = indexwords[next_w]
                Y[i, indseq, index_pred] = next_w
                indseq += 1
                # Increment index if target label present in reduced dictionary
    save_embeddings(dataset, X, Y, nbkeep)


def main():
    embedd('train')


if __name__ == '__main__':
    main()
